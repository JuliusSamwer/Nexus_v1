"""World-model adapters: one seam, two substrates. Each produces paired (imagined vs real)
rollouts and the per-step features / labels / intrinsics the gate consumes.

  EmeraldTorchAdapter — upstream EMERALD + the trained 5M Crafter checkpoint (PRIMARY).
  EmeraldJaxAdapter   — emerald_jax on Craftax-Classic (untrained-runnable; for later).

Per imagined step k=1..H the adapter emits:
  feat_prev (D)  deter of context s_{k-1}      feat_cur (D)  deter of imagined ŝ_k
  action (A)     one-hot a_{k-1}               horizon       k
  L1 token-mismatch frac | L2 KL(true-post‖imagined-prior) | L3 decoded-obs L2 |
  L4 reward-div | L4b value-div   (all: imagined ŝ_k vs real posterior s^true_k)
  entropy (prior categorical entropy) | disagreement (K-sample token disagreement)
  ref_latent (D) real posterior deter, pooled by the caller into the k-NN bank.

Labels are WM-INTERNAL: they compare the open-loop prior ŝ_k against the WM's OWN
posterior encode(real_obs_k), i.e. prior-vs-own-posterior drift, not absolute reality.
Action source is a MIXTURE: w.p. mixture_p the WM's policy, else uniform random.
"""

import numpy as np

A_CRAFTER = 17


class WMAdapter:
    num_actions = A_CRAFTER
    latent_dim = 512

    def generate_rollout(self, H, mixture_p, K, warmup, seed, rng):
        """Return a dict of per-step arrays (length H) for one paired rollout."""
        raise NotImplementedError


# =========================================================================== #
# Torch / upstream-EMERALD / Crafter  (PRIMARY)
# =========================================================================== #
class EmeraldTorchAdapter(WMAdapter):
    def __init__(self, checkpoint, device=None):
        import sys, os
        _THIS = os.path.dirname(os.path.abspath(__file__))
        probe = os.path.join(_THIS, "..", "probe")
        if probe not in sys.path:
            sys.path.insert(0, probe)
        import emerald_api as api
        import torch
        from emerald_torch.env import CrafterEnv  # crafter wrapper (CHW uint8)
        self.api = api
        self.torch = torch
        self.model, self.device = api.load_emerald(checkpoint, device=device, verbose=False)
        self.CrafterEnv = CrafterEnv
        self.num_actions = self.model.policy_network and A_CRAFTER
        self.latent_dim = int(self.model.config.dim_model) if hasattr(self.model, "config") else 512

    def _onehot(self, a):
        v = np.zeros(self.num_actions, np.float32)
        v[int(a)] = 1.0
        return v

    def _policy_action(self, posts, t, greedy=False):
        seed = self.api.seed_state_from_posts(posts, t)
        feat = self.api.feat_of_state(self.model, seed)
        dist = self.model.policy_network(feat)
        a = dist.mode() if greedy else dist.sample()
        return int(a.reshape(-1, self.num_actions)[0].argmax().item())

    def generate_rollout(self, H, mixture_p, K, warmup, seed, rng):
        torch, api = self.torch, self.api
        env = self.CrafterEnv(seed=int(seed))
        img0, _ = env.reset()                                  # (3,64,64) uint8
        frames = [img0]
        actions = [0]                                          # action INTO each state, a0=0
        # online warmup with the policy (growing-window observe for deter)
        for w in range(warmup + H):
            obs_stack = torch.as_tensor(np.stack(frames))      # (T,3,64,64) uint8
            act_oh = torch.as_tensor(np.stack([self._onehot(a) for a in actions]))
            with torch.no_grad():
                posts, _ = api.observe_sequence(self.model, obs_stack, act_oh)
            t = len(frames) - 1
            on_policy = (w < warmup) or (rng.random() < mixture_p)
            a = self._policy_action(posts, t) if on_policy else int(rng.integers(self.num_actions))
            img, _, done, _ = env.step(a)
            frames.append(img)
            actions.append(a)
            if done:
                break
        T = len(frames)
        if T < warmup + 2:
            return None                                        # episode died too early
        # one clean observe over the whole window -> real posteriors (deter+logits)
        obs_stack = torch.as_tensor(np.stack(frames))
        act_oh = torch.as_tensor(np.stack([self._onehot(a) for a in actions]))
        with torch.no_grad():
            posts, priors = api.observe_sequence(self.model, obs_stack, act_oh)
        t0 = min(warmup, T - 2)                                # seed index
        H_eff = min(H, T - 1 - t0)
        seed_state = api.seed_state_from_posts(posts, t0)
        # actions a_{t0..t0+H_eff-1} drive imagined steps 1..H_eff; pad one for the api
        fut_actions = act_oh[t0 + 1:t0 + 1 + H_eff]            # (H_eff,A)
        a_in = torch.cat([fut_actions, fut_actions[-1:]], 0)   # (H_eff+1,A)
        with torch.no_grad():
            img = api.imagine_with_actions(self.model, seed_state, a_in, H_eff)
            # K-sample disagreement: re-imagine K times, compare token argmaxes
            ksamp = [api.tokens_from_logits(
                        api.imagine_with_actions(self.model, seed_state, a_in, H_eff)["logits"])[0]
                     for _ in range(K)]
            return self._records(api, posts, img, ksamp, t0, H_eff)

    def _records(self, api, posts, img, ksamp, t0, H):
        torch = self.torch
        SymL = None
        rec = {k: [] for k in ("feat_prev", "feat_cur", "action", "horizon",
                               "L1", "L2", "L3", "L4", "L4b",
                               "entropy", "disagreement", "ref_latent")}
        # real posterior tokens/logits at imagined targets t0+1..t0+H
        real_logits = posts["logits"]                          # (1,T,4,4,S,V)
        real_deter = posts["deter"]                            # (1,T,512)
        img_logits = img["logits"]                             # (1,1+H,4,4,S,V)
        img_deter = img["deter"]
        img_stoch = img["stoch"]
        img_action = img["action"]                             # (1,1+H,A)
        try:
            self.model.decoder_network
            has_dec = True
        except Exception:
            has_dec = False
        for k in range(1, H + 1):
            tt = t0 + k
            ip = img_logits[:, k]                               # (1,4,4,S,V) prior
            rp = real_logits[:, tt]                             # posterior
            i_tok = ip.argmax(-1); r_tok = rp.argmax(-1)
            rec["L1"].append(float((i_tok != r_tok).float().mean()))
            p = torch.softmax(rp, -1); q = torch.softmax(ip, -1)
            rec["L2"].append(float((p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(-1).mean()))
            # L4 reward / value divergence via heads on (stoch,deter)
            fi = (img_stoch[:, k:k + 1], img_deter[:, k:k + 1])
            fr = (posts["stoch"][:, tt:tt + 1], real_deter[:, tt:tt + 1])
            rec["L4"].append(float((api.reward_pred(self.model, fi) -
                                    api.reward_pred(self.model, fr)).abs().mean()))
            rec["L4b"].append(float((api.value_pred(self.model, fi) -
                                     api.value_pred(self.model, fr)).abs().mean()))
            # L3 decoded-obs distance (best effort)
            if has_dec:
                try:
                    di = self.model.decoder_network(fi).mode()
                    # real frame target: re-decode the posterior (same metric space)
                    dr = self.model.decoder_network(fr).mode()
                    rec["L3"].append(float((di - dr).pow(2).mean()))
                except Exception:
                    rec["L3"].append(np.nan)
            else:
                rec["L3"].append(np.nan)
            qprob = torch.softmax(ip, -1)
            rec["entropy"].append(float(-(qprob * torch.log(qprob + 1e-8)).sum(-1).mean()))
            ks = torch.stack([t[k] for t in ksamp], 0)         # (K,4,4,S)
            mode = ks.mode(0).values
            rec["disagreement"].append(float((ks != mode[None]).float().mean()))
            rec["feat_prev"].append(img_deter[0, k - 1].cpu().numpy())
            rec["feat_cur"].append(img_deter[0, k].cpu().numpy())
            rec["action"].append(img_action[0, k - 1].cpu().numpy())
            rec["horizon"].append(float(k))
            rec["ref_latent"].append(real_deter[0, tt].cpu().numpy())
        return {k: np.asarray(v, np.float32) for k, v in rec.items()}


# =========================================================================== #
# JAX / emerald_jax / Craftax-Classic  (for later; untrained-runnable for smoke)
# =========================================================================== #
class EmeraldJaxAdapter(WMAdapter):
    def __init__(self, cfg=None, checkpoint=None):
        import jax, jax.numpy as jnp
        from emerald_jax import config as cfgmod, env as envmod, model as modelmod, dists
        self.jax, self.jnp, self.dists = jax, jnp, dists
        self.cfg = cfg or cfgmod.tiny()
        self.envmod = envmod
        self.env, self.eparams = envmod.make_env(auto_reset=True)
        self.A = envmod.NUM_ACTIONS
        self.num_actions = self.A
        self.latent_dim = self.cfg.dim_model

        class _ImagineAgent(modelmod.EmeraldAgent):
            def imagine_actions(self, start_stoch, start_deter, actions):
                c = self.cfg
                B = start_stoch.shape[0]
                s, d = start_stoch, start_deter
                stochs, deters, logs, x_seq = [s], [d], [], []
                for h in range(actions.shape[1]):
                    a = actions[:, h:h + 1]
                    x_seq.append(self.tssm.mix(self.tssm.encode_stoch(s), a))
                    x = jnp.concatenate(x_seq, axis=1)
                    d = self.tssm.transformer(x, causal=True)[:, -1:]
                    lg, s_oh = self.tssm.mask_network.sample(
                        self.tssm.deter_to_dec(d), c.num_decoding_steps)
                    s = s_oh.reshape(B, 1, 4, 4, -1)
                    stochs.append(s); deters.append(d); logs.append(lg)
                return {"stoch": jnp.concatenate(stochs, 1),
                        "deter": jnp.concatenate(deters, 1),
                        "logits": jnp.concatenate(logs, 1)}

        self.agent = _ImagineAgent(self.cfg, self.A)
        key = jax.random.PRNGKey(0)
        B, L = 1, max(4, self.cfg.L)
        dummy = {"image": jnp.zeros((B, L, 3, 64, 64)),
                 "action": jax.nn.one_hot(jnp.zeros((B, L), jnp.int32), self.A),
                 "reward": jnp.zeros((B, L)), "cont": jnp.ones((B, L))}
        rngs = {k: key for k in ("params", "sample", "mask", "order")}
        if checkpoint:
            import pickle
            with open(checkpoint, "rb") as f:
                self.params = pickle.load(f)["params"]
        else:
            self.params = self.agent.init(rngs, dummy, 0.0, 0.0,
                                          method=self.agent.compute_losses)

    def _ap(self, method, *a, **kw):
        key = kw.pop("key")
        rngs = {k: key for k in ("sample", "mask", "order")}
        return self.agent.apply(self.params, *a, method=method, rngs=rngs)

    def generate_rollout(self, H, mixture_p, K, warmup, seed, rng):
        jax, jnp = self.jax, self.jnp
        key = jax.random.PRNGKey(int(seed))
        keys = jax.random.split(key, 1)
        img, state = self.envmod.reset(self.env, self.eparams, keys)   # (1,3,64,64)
        frames = [np.asarray(img)[0]]
        acts = [0]
        for w in range(warmup + H):
            a = int(rng.integers(self.A))                              # smoke: random source
            key, sk = jax.random.split(key)
            act = jnp.array([a], jnp.int32)
            img, state, r, d, info = self.envmod.step(
                self.env, self.eparams, jax.random.split(sk, 1), state, act)
            frames.append(np.asarray(img)[0])
            acts.append(a)
        T = len(frames)
        H_eff = min(H, T - 1 - warmup)
        images = jnp.asarray(np.stack(frames))[None]                   # (1,T,3,64,64)
        act_oh = jax.nn.one_hot(jnp.asarray(acts), self.A)[None]       # (1,T,A)
        key, k1 = jax.random.split(key)
        enc = self._ap(lambda s, im: s.encoder(im), images, key=k1)
        post, _ = self._ap(lambda s, st, ac: s.tssm.observe(st, ac),
                           enc["stoch"], act_oh, key=k1)
        t0 = warmup
        seed_stoch = post["stoch"][:, t0:t0 + 1]
        seed_deter = post["deter"][:, t0:t0 + 1]
        fut = act_oh[:, t0 + 1:t0 + 1 + H_eff]
        key, k2 = jax.random.split(key)
        im = self._ap(lambda s, ss, sd, ac: s.imagine_actions(ss, sd, ac),
                      seed_stoch, seed_deter, fut, key=k2)
        ksamp = []
        for _ in range(K):
            key, kk = jax.random.split(key)
            ik = self._ap(lambda s, ss, sd, ac: s.imagine_actions(ss, sd, ac),
                          seed_stoch, seed_deter, fut, key=kk)
            ksamp.append(np.asarray(ik["logits"]).argmax(-1)[0])       # (H,4,4,S)
        return self._records(post, im, np.stack(ksamp), enc, t0, H_eff, act_oh)

    def _records(self, post, im, ksamp, enc, t0, H, act_oh):
        jnp = self.jnp
        SymLog = self.dists.SymLogDiscreteDist
        rec = {k: [] for k in ("feat_prev", "feat_cur", "action", "horizon",
                               "L1", "L2", "L3", "L4", "L4b",
                               "entropy", "disagreement", "ref_latent")}
        post_logits = np.asarray(enc["logits"])[0]                     # (T,4,4,S,V)
        post_deter = np.asarray(post["deter"])[0]
        post_stoch = np.asarray(post["stoch"])[0]
        i_logits = np.asarray(im["logits"])[0]                         # (H,4,4,S,V)
        i_deter = np.asarray(im["deter"])[0]                           # (1+H,512)
        i_stoch = np.asarray(im["stoch"])[0]
        for k in range(1, H + 1):
            tt = t0 + k
            ip = i_logits[k - 1]; rp = post_logits[tt]
            rec["L1"].append(float((ip.argmax(-1) != rp.argmax(-1)).mean()))
            p = _softmax(rp); q = _softmax(ip)
            rec["L2"].append(float((p * (np.log(p + 1e-8) - np.log(q + 1e-8))).sum(-1).mean()))
            fi = (jnp.asarray(i_stoch[k:k + 1][None]), jnp.asarray(i_deter[k:k + 1][None]))
            fr = (jnp.asarray(post_stoch[tt:tt + 1][None]), jnp.asarray(post_deter[tt:tt + 1][None]))
            ri = float(SymLog(self._ap(lambda s, f: s.reward_head(f), fi, key=self.jax.random.PRNGKey(0))).mode().mean())
            rr = float(SymLog(self._ap(lambda s, f: s.reward_head(f), fr, key=self.jax.random.PRNGKey(0))).mode().mean())
            vi = float(SymLog(self._ap(lambda s, f: s.value_head(f), fi, key=self.jax.random.PRNGKey(0))).mode().mean())
            vr = float(SymLog(self._ap(lambda s, f: s.value_head(f), fr, key=self.jax.random.PRNGKey(0))).mode().mean())
            rec["L4"].append(abs(ri - rr)); rec["L4b"].append(abs(vi - vr))
            rec["L3"].append(np.nan)                                   # decoder optional in JAX smoke
            rec["entropy"].append(float(-(q * np.log(q + 1e-8)).sum(-1).mean()))
            ks = ksamp[:, k - 1]                                       # (K,4,4,S)
            mode = _mode0(ks)
            rec["disagreement"].append(float((ks != mode[None]).mean()))
            rec["feat_prev"].append(i_deter[k - 1])
            rec["feat_cur"].append(i_deter[k])
            rec["action"].append(np.asarray(act_oh)[0, t0 + k - 1])
            rec["horizon"].append(float(k))
            rec["ref_latent"].append(post_deter[tt])
        return {k: np.asarray(v, np.float32) for k, v in rec.items()}


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _mode0(a):
    """Mode along axis 0 for integer array (avoids scipy dep)."""
    out = np.zeros(a.shape[1:], a.dtype)
    flat = a.reshape(a.shape[0], -1)
    o = out.reshape(-1)
    for j in range(flat.shape[1]):
        vals, cnts = np.unique(flat[:, j], return_counts=True)
        o[j] = vals[cnts.argmax()]
    return out


def make_adapter(substrate, checkpoint=None, cfg=None):
    if substrate == "torch":
        return EmeraldTorchAdapter(checkpoint)
    if substrate == "jax":
        return EmeraldJaxAdapter(cfg=cfg, checkpoint=checkpoint)
    raise ValueError(substrate)
