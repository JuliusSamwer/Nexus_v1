"""EMERALD agent in Flax: world model + imagination actor-critic.

Method-for-method port of emerald_torch.model.EmeraldAgent. All torch `.detach()` /
`torch.no_grad()` become jax.lax.stop_gradient so the combined loss has the SAME
gradient structure as training the three optimizers separately:
  * world_model_loss -> encoder/decoder/tssm/reward_head/continue_head
  * actor_critic_loss -> policy_head (actor) and value_head (critic) only;
    everything from `imagine` is frozen (stop_gradient), value_target is frozen.
The percentile return-norm EMA (perc_low/high) is carried as explicit state (args in,
new values out) instead of torch buffers. value_target is EMA'd outside (in train.py).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from . import nets
from .dists import (OneHotDist, SymLogDiscreteDist, BernoulliDist, MSEDist,
                    kl_onehot)

sg = jax.lax.stop_gradient


class EmeraldAgent(nn.Module):
    cfg: object
    num_actions: int

    def setup(self):
        c = self.cfg
        self.encoder = nets.Encoder(c)
        self.decoder = nets.ImageDecoder(c)
        self.tssm = nets.TSSM(c, self.num_actions)
        self.reward_head = nets.Head(c, c.bins, zero_init_out=True)
        self.continue_head = nets.Head(c, 1, zero_init_out=True)
        self.value_head = nets.Head(c, c.bins, zero_init_out=True)
        self.policy_head = nets.Head(c, self.num_actions)
        self.value_target = nets.Head(c, c.bins, zero_init_out=True)

    # ---- world model ---------------------------------------------------- #
    def world_model_loss(self, batch):
        c = self.cfg
        image, action = batch["image"], batch["action"]
        reward, cont = batch["reward"], batch["cont"]
        enc = self.encoder(image)
        post, prior = self.tssm.observe(enc["stoch"], action)
        feats = (post["stoch"], post["deter"])

        loss_image = -self.decoder(feats).log_prob(image).mean()

        post_logits = enc["logits"]
        kl_pr = kl_onehot(sg(post_logits), prior["logits"], c.uniform_mix)
        kl_po = kl_onehot(post_logits, sg(prior["logits"]), c.uniform_mix)
        loss_kl_prior = jnp.clip(kl_pr.sum(-1), c.free_nats).mean()
        loss_kl_post = jnp.clip(kl_po.sum(-1), c.free_nats).mean()
        loss_kl_mask = jnp.zeros(())
        if c.num_decoding_steps > 0:
            m = prior["mask"].astype(jnp.float32)
            kl_m = kl_onehot(sg(post_logits), prior["logits_masked"], c.uniform_mix)
            kl_m = c.discrete * (kl_m * m).sum((-3, -2, -1)) / (m.sum((-3, -2, -1)) + 1e-8)
            loss_kl_mask = jnp.clip(kl_m, c.free_nats).mean()

        loss_reward = -SymLogDiscreteDist(self.reward_head(feats)).log_prob(
            reward[..., None]).mean()
        loss_cont = -BernoulliDist(self.continue_head(feats)).log_prob(
            cont[..., None]).mean()

        loss = (c.loss_decoder_scale * loss_image
                + c.loss_kl_prior_scale * loss_kl_prior
                + c.loss_kl_post_scale * loss_kl_post
                + c.loss_kl_mask_scale * loss_kl_mask
                + c.loss_reward_scale * loss_reward
                + c.loss_continue_scale * loss_cont)
        metrics = {"model_loss": loss, "image_loss": loss_image,
                   "kl_prior": loss_kl_prior, "kl_post": loss_kl_post,
                   "kl_mask": loss_kl_mask, "reward_loss": loss_reward,
                   "cont_loss": loss_cont}
        s = c.img_stride
        detached = {"stoch": sg(post["stoch"][:, ::s]), "deter": sg(post["deter"][:, ::s]),
                    "cont": sg(cont[:, ::s])}
        return loss, metrics, detached

    # ---- imagination ---------------------------------------------------- #
    def imagine(self, start):
        c = self.cfg
        B, n = start["stoch"].shape[:2]
        Bp = B * n
        s = start["stoch"].reshape(Bp, 1, 4, 4, -1)
        d = start["deter"].reshape(Bp, 1, c.dim_model)
        stochs, deters, actions, x_seq = [s], [d], [], []
        for _ in range(c.H):
            a = OneHotDist(self.policy_head((sg(s), sg(d)))).rsample(self.make_rng("sample"))
            actions.append(a)
            x_seq.append(self.tssm.mix(self.tssm.encode_stoch(s), a))
            x = jnp.concatenate(x_seq, axis=1)
            d = self.tssm.transformer(x, causal=True)[:, -1:]
            _, s_oh = self.tssm.mask_network.sample(self.tssm.deter_to_dec(d),
                                                    c.num_decoding_steps)
            s = s_oh.reshape(Bp, 1, 4, 4, -1)
            stochs.append(s)
            deters.append(d)
        actions.append(OneHotDist(self.policy_head((sg(s), sg(d)))).rsample(
            self.make_rng("sample")))
        return sg({"stoch": jnp.concatenate(stochs, axis=1),
                   "deter": jnp.concatenate(deters, axis=1),
                   "action": jnp.concatenate(actions, axis=1),
                   "cont_first": start["cont"].reshape(Bp, 1, 1)})

    def td_lambda(self, reward, value, discount):
        c = self.cfg
        interm = reward + discount * (1 - c.lambda_td) * value
        vals = [value[:, -1]]
        for t in reversed(range(interm.shape[1])):
            vals.append(interm[:, t] + discount[:, t] * c.lambda_td * vals[-1])
        return jnp.stack(list(reversed(vals))[:-1], axis=1)

    def update_perc(self, returns, perc_low, perc_high):
        c = self.cfg
        flat = sg(returns).reshape(-1)
        low = jnp.quantile(flat, c.return_norm_perc_low)
        high = jnp.quantile(flat, c.return_norm_perc_high)
        perc_low = c.return_norm_decay * perc_low + (1 - c.return_norm_decay) * low
        perc_high = c.return_norm_decay * perc_high + (1 - c.return_norm_decay) * high
        invscale = jnp.maximum(perc_high - perc_low, 1.0 / c.return_norm_limit)
        return perc_low, perc_high, sg(perc_low), sg(invscale)

    # ---- actor-critic --------------------------------------------------- #
    def actor_critic_loss(self, start, perc_low, perc_high):
        c = self.cfg
        traj = self.imagine(start)
        feats = (traj["stoch"], traj["deter"])
        rewards = sg(SymLogDiscreteDist(self.reward_head(feats)).mode())
        values = sg(SymLogDiscreteDist(self.value_head(feats)).mode())
        continues = sg(BernoulliDist(self.continue_head(feats)).mode())
        continues = jnp.concatenate([traj["cont_first"], continues[:, 1:]], axis=1)
        weights = sg(jnp.cumprod(c.gamma * continues, axis=1) / c.gamma)
        returns = self.td_lambda(rewards[:, 1:], values[:, 1:], c.gamma * continues[:, 1:])
        perc_low, perc_high, offset, invscale = self.update_perc(returns, perc_low, perc_high)
        normed_returns = (returns - offset) / invscale
        normed_base = (values[:, :-1] - offset) / invscale
        advantage = (normed_returns - normed_base).squeeze(-1)          # (Bp,H)

        fs = (sg(feats[0]), sg(feats[1]))
        w = weights[:, :-1].squeeze(-1)

        policy_dist = OneHotDist(self.policy_head(fs))
        logp = policy_dist.log_prob(sg(traj["action"]))[:, :-1]
        ent = policy_dist.entropy()[:, :-1]
        actor_loss = -((logp * sg(advantage) + c.eta_entropy * ent) * w).mean()

        value_dist = SymLogDiscreteDist(self.value_head((fs[0][:, :-1], fs[1][:, :-1])))
        value_loss = value_dist.log_prob(sg(returns))
        if c.target_value_reg:
            vt = sg(SymLogDiscreteDist(
                self.value_target((fs[0][:, :-1], fs[1][:, :-1]))).mode())
            value_loss = value_loss + c.critic_slow_reg_scale * value_dist.log_prob(vt)
        value_loss = -(value_loss * w).mean()

        metrics = {"actor_loss": actor_loss, "value_loss": value_loss,
                   "imag_reward_mean": rewards.mean(), "returns_mean": returns.mean(),
                   "policy_ent": ent.mean(), "perc_low": perc_low, "perc_high": perc_high}
        return actor_loss, value_loss, metrics, (perc_low, perc_high)

    # ---- combined (single grad target) ---------------------------------- #
    def compute_losses(self, batch, perc_low, perc_high):
        wm_loss, wm_metrics, detached = self.world_model_loss(batch)
        a_loss, v_loss, ac_metrics, perc_new = self.actor_critic_loss(
            detached, perc_low, perc_high)
        total = wm_loss + a_loss + v_loss
        metrics = {**wm_metrics, **ac_metrics, "total_loss": total}
        return total, (metrics, perc_new)

    # ---- acting (single step, online; for eval / data collection) ------- #
    def act(self, image, prev_stoch, prev_action, sample=True):
        """image (N,1,3,64,64); prev_stoch (N,L,4,4,SV); prev_action (N,L,A) actions
        into each past state (a_0=0). Returns (action one-hot (N,A), stoch_t (N,4,4,SV))."""
        c = self.cfg
        enc = self.encoder(image)                                      # (N,1,4,4,SV)
        stoch_t = enc["stoch"]
        stoch_seq = jnp.concatenate([prev_stoch, stoch_t], axis=1)[:, -c.att_context_left:]
        act_seq = prev_action[:, -c.att_context_left:]
        B = stoch_seq.shape[0]
        init = self.tssm.initial_stoch(B, 1)
        prev = jnp.concatenate([init, stoch_seq[:, :-1]], axis=1)
        x = self.tssm.mix(self.tssm.encode_stoch(prev), act_seq)
        deter_t = self.tssm.transformer(x, causal=True)[:, -1:]        # (N,1,512)
        dist = OneHotDist(self.policy_head((stoch_t, deter_t)))
        action = dist.sample(self.make_rng("sample")) if sample else dist.mode()
        return action[:, 0], stoch_t[:, 0]
