#!/usr/bin/env python3
"""Stage-2 decision-aware world-model finetuning (MVP).

Loads a pretrained EMERALD-JAX checkpoint and continues training with an added
MULTI-STEP own-rollout objective, so imagined rollouts become decision-reliable deeper
into the horizon. Actor co-trains; the standard world-model loss stays on as a recon
anchor against collapse.

Arms (--arm):
  A0  recon-only continued training (control for 'just more gradient steps')
  A1  + multi-step token-matching over the model's OWN rollout, UNIFORM      (exposure-bias)
  A2  + multi-step token-matching, weighted by value/policy SENSITIVITY      (decision-aware)
  A3  (later) weighted by the divergence critic

Multi-step term: from the real seed state, roll the WM forward teacher-forced (real
actions), carrying its OWN predicted stoch with stop-gradient between steps (scheduled
sampling — stable, no BPTT). Each step's predicted token logits are matched (CE) to the
real posterior tokens; A2 weights that per token-dim by |dV/ds|+|dpi/ds| (sg).

  python stage2_finetune/finetune.py --checkpoint <W0.pkl> --arm A2 --steps 8000 \
      --out /workspace/ft/full5M_A2 --label full5M-A2
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax

from emerald_jax import model, replay, train
from emerald_jax.dists import OneHotDist, SymLogDiscreteDist

sg = jax.lax.stop_gradient


def load_ckpt(path):
    blob = pickle.load(open(path, "rb"))
    return blob["cfg"], {"params": blob["params"]["params"]}


def rollout_tf_logits(m, seed_stoch, actions):
    """Scheduled-sampling teacher-forced rollout. seed_stoch (B,1,4,4,SV), actions (B,L,A).
    Returns predicted token logits (B,L,4,4,S,V). Carry is stop-gradded between steps so
    gradient flows to each step's dynamics read-out, not through the recurrence (no BPTT)."""
    c = m.cfg
    B, L = seed_stoch.shape[0], actions.shape[1]
    s = seed_stoch
    xbuf = jnp.zeros((B, L, c.dim_model))
    logits_seq = []
    for t in range(L):
        xbuf = xbuf.at[:, t:t + 1].set(m.tssm.mix(m.tssm.encode_stoch(sg(s)), actions[:, t:t + 1]))
        d = m.tssm.transformer(xbuf, causal=True)[:, t:t + 1]
        lg, s_oh = m.tssm.mask_network.sample(m.tssm.deter_to_dec(d), c.num_decoding_steps)
        logits_seq.append(lg)
        s = s_oh.reshape(B, 1, 4, 4, -1)                 # carried prediction (sg'd next step)
    return jnp.concatenate(logits_seq, axis=1)            # (B,L,4,4,S,V)


def vp_sensitivity(m, stoch, deter):
    """Per-token-dim value+policy sensitivity |dV/ds|+|dpi/ds|, sg, normalised to mean 1.
    stoch (B,L,4,4,SV), deter (B,L,Dm) -> weight (B,L,4,4,S)."""
    c = m.cfg
    def scalar(s):
        feat = (s, deter)
        v = SymLogDiscreteDist(m.value_head(feat)).mode().sum()
        p = jnp.abs(m.policy_head(feat)).sum()
        return v + p
    g = jax.grad(scalar)(stoch)                           # (B,L,4,4,SV)
    B, L = stoch.shape[:2]
    w = jnp.abs(g).reshape(B, L, 4, 4, c.stoch_size, c.discrete).sum(-1)   # (B,L,4,4,S)
    return sg(w / (w.mean() + 1e-8))


class FTAgent(model.EmeraldAgent):
    def finetune_losses(self, batch, perc_low, perc_high, arm, decision_w):
        c = self.cfg
        # --- P2: phase-2 actor-critic retrain on the FROZEN (improved) WM. ---
        # actor_critic_loss only routes gradient to policy/value heads (imagined rollout is
        # stop-gradded), so the WM is naturally frozen — just skip the WM/decision terms.
        if arm == "P2":
            enc = self.encoder(batch["image"])
            post, _ = self.tssm.observe(enc["stoch"], batch["action"])
            s = c.img_stride
            detached = {"stoch": sg(post["stoch"][:, ::s]), "deter": sg(post["deter"][:, ::s]),
                        "cont": sg(batch["cont"][:, ::s])}
            a_loss, v_loss, ac_metrics, perc_new = self.actor_critic_loss(detached, perc_low, perc_high)
            total = a_loss + v_loss
            metrics = {**ac_metrics, "decision_loss": jnp.zeros(()), "image_loss": jnp.zeros(()),
                       "total_loss": total}
            return total, (metrics, perc_new)
        # --- standard world-model loss (recon anchor) + posterior states ---
        wm_loss, wm_metrics, detached = self.world_model_loss(batch)
        # --- multi-step own-rollout decision/state term ---
        decision_loss = jnp.zeros(())
        if arm in ("A1", "A2"):
            enc = self.encoder(batch["image"])
            post, _ = self.tssm.observe(enc["stoch"], batch["action"])
            real_tok = sg(jnp.argmax(enc["logits"], -1))         # (B,L,4,4,S) targets
            pred_logits = rollout_tf_logits(self, post["stoch"][:, :1], batch["action"])
            ce = optax.softmax_cross_entropy_with_integer_labels(pred_logits, real_tok)  # (B,L,4,4,S)
            if arm == "A2":
                w = vp_sensitivity(self, sg(post["stoch"]), sg(post["deter"]))
                decision_loss = (w * ce).mean()
            else:
                decision_loss = ce.mean()
        # --- actor-critic co-train (standard, WM frozen via detached) ---
        a_loss, v_loss, ac_metrics, perc_new = self.actor_critic_loss(detached, perc_low, perc_high)
        total = wm_loss + decision_w * decision_loss + a_loss + v_loss
        metrics = {**wm_metrics, **ac_metrics, "decision_loss": decision_loss, "total_loss": total}
        return total, (metrics, perc_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arm", required=True, choices=["A0", "A1", "A2", "P2"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--decision_w", type=float, default=1.0)
    ap.add_argument("--lr_scale", type=float, default=0.3, help="scale pretrain LRs for finetune")
    ap.add_argument("--collect_steps", type=int, default=64)
    ap.add_argument("--grad_per_collect", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--out", default="ft_out")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    # test/perf overrides (don't change model architecture)
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seq_L", type=int, default=None)
    args = ap.parse_args()

    cfg, params = load_ckpt(args.checkpoint)
    for k in ("model_lr", "value_lr", "actor_lr"):
        setattr(cfg, k, getattr(cfg, k) * args.lr_scale)
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_L is not None:
        cfg.L = args.seq_L

    A = train.cenv.NUM_ACTIONS if hasattr(train, "cenv") else None
    # build state via the standard path, then swap in FTAgent + pretrained params
    st = train.init_state(cfg, args.seed)
    A = st["A"]
    agent = FTAgent(cfg, A)
    st["agent"] = agent
    st["params"] = params
    tx, opt_state = train.make_optimizer(cfg, params)
    st["tx"], st["opt_state"] = tx, opt_state

    rollout = train.make_rollout(agent, st["env"], st["eparams"], st["ach_keys"], A)

    def loss_fn(p, batch, perc, rk):
        ks = jax.random.split(rk, 3)
        return agent.apply(p, batch, perc[0], perc[1], args.arm, args.decision_w,
                           rngs={"sample": ks[0], "mask": ks[1], "order": ks[2]},
                           method=agent.finetune_losses)

    @jax.jit
    def train_step(p, opt_state, perc, batch, rk):
        (total, (metrics, perc_new)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, batch, perc, rk)
        updates, opt_state = tx.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        d = cfg.critic_ema_decay
        vt = jax.tree_util.tree_map(lambda t, h: (1 - d) * t + d * h,
                                    p["params"]["value_target"], p["params"]["value_head"])
        p = {"params": {**p["params"], "value_target": vt}}
        return p, opt_state, perc_new, metrics

    os.makedirs(args.out, exist_ok=True)
    lbl = args.label or f"{os.path.basename(args.checkpoint)}-{args.arm}"
    print(f"[ft {lbl}] arm={args.arm} steps={args.steps} decision_w={args.decision_w} "
          f"lr_scale={args.lr_scale} | {sum(x.size for x in jax.tree_util.tree_leaves(params))/1e6:.1f}M params",
          flush=True)

    def collect():
        (st["obs"], st["estate"], st["rings"], st["key"]), outs = rollout(
            st["params"], st["obs"], st["estate"], st["rings"], st["key"], args.collect_steps, True)
        img, a_int, reward, done, _ = outs
        st["buf"] = replay.add_rollout(st["buf"], img, a_int, reward, done)

    # warm the buffer from the pretrained policy
    while int(st["buf"].size) < min(cfg.capacity, 8 * cfg.L):
        collect()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step % args.grad_per_collect == 1:
            collect()
        st["key"], bk, tk = jax.random.split(st["key"], 3)
        batch = replay.sample(st["buf"], bk, cfg.batch_size, cfg.L, A)
        st["params"], st["opt_state"], st["perc"], m = train_step(
            st["params"], st["opt_state"], st["perc"], batch, tk)
        if step % 200 == 0:
            print(f"[{step:>6}] {(step)/(time.time()-t0):.1f} it/s | total {float(m['total_loss']):.2f} "
                  f"img {float(m['image_loss']):.2f} decision {float(m['decision_loss']):.3f} "
                  f"actor {float(m['actor_loss']):+.3f} value {float(m['value_loss']):.2f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            path = os.path.join(args.out, f"ft_step_{step}.pkl")
            with open(path, "wb") as f:
                pickle.dump(jax.device_get({"params": st["params"], "cfg": cfg, "step": step}), f)
            print(f"[ckpt] {path}", flush=True)
    print(f"[done] {lbl} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
