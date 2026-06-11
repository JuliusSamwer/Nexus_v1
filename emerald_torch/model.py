"""EMERALD agent: world model + imagination actor-critic.

Faithful to EMERALD's training (nnet/models/emerald.py) with two documented
simplifications for local MPS runs:
  1. World-model training uses full causal attention over the L window — no
     cross-batch KV cache / TBTT. Replay samples WITHIN a single episode, so the
     window's first state is the only reset (init prepend in TSSM.observe).
  2. Imagination rolls each subsampled start forward with intra-rollout context
     only (the transformer attends its own imagined prefix), rather than EMERALD's
     att_context_left real-context KV cache.
Everything else (spatial latent, MaskGIT prior + KL, symlog/twohot heads, TD-lambda,
percentile return norm, slow critic target) follows EMERALD's hyperparameters.
"""

import copy

import torch
import torch.nn as nn

from . import nets
from .dists import kl_onehot


class EmeraldAgent(nn.Module):
    def __init__(self, cfg, num_actions):
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        self.encoder = nets.Encoder(cfg)
        self.decoder = nets.ImageDecoder(cfg)
        self.tssm = nets.TSSM(cfg, num_actions)
        self.reward_head = nets.RewardHead(cfg)
        self.continue_head = nets.ContinueHead(cfg)
        self.value_head = nets.ValueHead(cfg)
        self.policy_head = nets.PolicyHead(cfg, num_actions)
        self.value_target = copy.deepcopy(self.value_head)
        for p in self.value_target.parameters():
            p.requires_grad_(False)
        self.register_buffer("perc_low", torch.tensor(0.0))
        self.register_buffer("perc_high", torch.tensor(0.0))

    # ---- parameter groups ---------------------------------------------- #
    def wm_parameters(self):
        mods = [self.encoder, self.decoder, self.tssm, self.reward_head, self.continue_head]
        return [p for m in mods for p in m.parameters()]

    def actor_parameters(self):
        return list(self.policy_head.parameters())

    def critic_parameters(self):
        return list(self.value_head.parameters())

    # ---- world model ---------------------------------------------------- #
    def world_model_loss(self, batch):
        cfg = self.cfg
        image, action = batch["image"], batch["action"]
        reward, cont = batch["reward"], batch["cont"]
        enc = self.encoder(image)                                   # post stoch + logits
        post, prior = self.tssm.observe(enc["stoch"], action)
        feats = (post["stoch"], post["deter"])

        image_dist = self.decoder(feats)
        loss_image = -image_dist.log_prob(image).mean()

        post_logits = enc["logits"]
        kl_pr = kl_onehot(post_logits.detach(), prior["logits"], cfg.uniform_mix)
        kl_po = kl_onehot(post_logits, prior["logits"].detach(), cfg.uniform_mix)
        loss_kl_prior = kl_pr.sum(-1).clamp(min=cfg.free_nats).mean()
        loss_kl_post = kl_po.sum(-1).clamp(min=cfg.free_nats).mean()
        loss_kl_mask = torch.zeros((), device=image.device)
        if cfg.num_decoding_steps > 0:
            m = prior["mask"].float()
            kl_m = kl_onehot(post_logits.detach(), prior["logits_masked"], cfg.uniform_mix)
            kl_m = cfg.discrete * (kl_m * m).sum((-3, -2, -1)) / (m.sum((-3, -2, -1)) + 1e-8)
            loss_kl_mask = kl_m.clamp(min=cfg.free_nats).mean()

        reward_dist = self.reward_head(feats)
        loss_reward = -reward_dist.log_prob(reward.unsqueeze(-1)).mean()
        cont_dist = self.continue_head(feats)
        loss_cont = -cont_dist.log_prob(cont.unsqueeze(-1)).mean()

        loss = (cfg.loss_decoder_scale * loss_image
                + cfg.loss_kl_prior_scale * loss_kl_prior
                + cfg.loss_kl_post_scale * loss_kl_post
                + cfg.loss_kl_mask_scale * loss_kl_mask
                + cfg.loss_reward_scale * loss_reward
                + cfg.loss_continue_scale * loss_cont)
        metrics = {
            "model_loss": loss.item(), "image_loss": loss_image.item(),
            "kl_prior": loss_kl_prior.item(), "kl_post": loss_kl_post.item(),
            "kl_mask": float(loss_kl_mask.detach()), "reward_loss": loss_reward.item(),
            "cont_loss": loss_cont.item(),
        }
        # Detached posteriors + true continue for imagination, subsampled by img_stride.
        s = cfg.img_stride
        detached = {
            "stoch": post["stoch"][:, ::s].detach(),
            "deter": post["deter"][:, ::s].detach(),
            "cont": cont[:, ::s].detach(),
        }
        return loss, metrics, detached

    # ---- imagination ---------------------------------------------------- #
    def imagine(self, start):
        """start: stoch (B,n,SV,4,4), deter (B,n,512). Returns flattened trajectories
        (B', 1+H, ...)."""
        cfg = self.cfg
        B, n = start["stoch"].shape[:2]
        Bp = B * n
        s = start["stoch"].reshape(Bp, 1, *start["stoch"].shape[2:])
        d = start["deter"].reshape(Bp, 1, cfg.dim_model)
        stochs, deters, actions, x_seq = [s], [d], [], []
        for h in range(cfg.H):
            a = self.policy_head((s.detach(), d.detach())).rsample()
            actions.append(a)
            x_seq.append(self.tssm.mix(self.tssm.encode_stoch(s), a))
            x = torch.cat(x_seq, dim=1)
            d = self.tssm.transformer(x, causal=True)[:, -1:]
            _, s_oh = self.tssm.mask_network.sample(self.tssm.deter_to_dec(d),
                                                    cfg.num_decoding_steps)
            s = s_oh.flatten(-2, -1).permute(0, 1, 4, 2, 3)
            stochs.append(s)
            deters.append(d)
        actions.append(self.policy_head((s.detach(), d.detach())).rsample())
        return {
            "stoch": torch.cat(stochs, dim=1),
            "deter": torch.cat(deters, dim=1),
            "action": torch.cat(actions, dim=1),
            "cont_first": start["cont"].reshape(Bp, 1, 1),
        }

    def td_lambda(self, reward, value, discount):
        cfg = self.cfg
        interm = reward + discount * (1 - cfg.lambda_td) * value
        vals = [value[:, -1]]
        for t in reversed(range(interm.shape[1])):
            vals.append(interm[:, t] + discount[:, t] * cfg.lambda_td * vals[-1])
        return torch.stack(list(reversed(vals))[:-1], dim=1)

    def update_perc(self, returns):
        cfg = self.cfg
        flat = returns.detach().flatten().float().cpu()
        low = torch.quantile(flat, cfg.return_norm_perc_low).to(returns.device)
        high = torch.quantile(flat, cfg.return_norm_perc_high).to(returns.device)
        self.perc_low.mul_(cfg.return_norm_decay).add_((1 - cfg.return_norm_decay) * low)
        self.perc_high.mul_(cfg.return_norm_decay).add_((1 - cfg.return_norm_decay) * high)
        offset = self.perc_low
        invscale = torch.clamp(self.perc_high - self.perc_low, min=1.0 / cfg.return_norm_limit)
        return offset.detach(), invscale.detach()

    # ---- actor-critic --------------------------------------------------- #
    def actor_critic_loss(self, start):
        cfg = self.cfg
        with torch.no_grad():
            traj = self.imagine(start)
            feats = (traj["stoch"], traj["deter"])
            rewards = self.reward_head(feats).mode()                # (B',1+H,1)
            values = self.value_head(feats).mode()
            continues = self.continue_head(feats).mode()
            continues = torch.cat([traj["cont_first"], continues[:, 1:]], dim=1)
            weights = (torch.cumprod(cfg.gamma * continues, dim=1) / cfg.gamma).detach()
            returns = self.td_lambda(rewards[:, 1:], values[:, 1:],
                                     cfg.gamma * continues[:, 1:])   # (B',H,1)
            offset, invscale = self.update_perc(returns)
            normed_returns = (returns - offset) / invscale
            normed_base = (values[:, :-1] - offset) / invscale
            advantage = (normed_returns - normed_base).squeeze(-1)  # (B',H)

        fs = (feats[0].detach(), feats[1].detach())
        w = weights[:, :-1].squeeze(-1)

        # Actor (REINFORCE with normalized advantage + entropy)
        policy_dist = self.policy_head(fs)
        logp = policy_dist.log_prob(traj["action"].detach())[:, :-1]
        ent = policy_dist.entropy()[:, :-1]
        actor_loss = (logp * advantage.detach() + cfg.eta_entropy * ent) * w
        actor_loss = -actor_loss.mean()

        # Critic (twohot value regression + slow target reg)
        value_dist = self.value_head((fs[0][:, :-1], fs[1][:, :-1]))
        value_loss = value_dist.log_prob(returns.detach())
        if cfg.target_value_reg:
            with torch.no_grad():
                vt = self.value_target((fs[0][:, :-1], fs[1][:, :-1])).mode()
            value_loss = value_loss + cfg.critic_slow_reg_scale * value_dist.log_prob(vt)
        value_loss = -(value_loss * weights[:, :-1].squeeze(-1)).mean()

        metrics = {
            "actor_loss": actor_loss.item(), "value_loss": value_loss.item(),
            "imag_reward_mean": rewards.mean().item(),
            "returns_mean": returns.mean().item(),
            "policy_ent": ent.mean().item(),
            "perc_low": self.perc_low.item(), "perc_high": self.perc_high.item(),
        }
        return actor_loss, value_loss, metrics

    def update_target(self):
        d = self.cfg.critic_ema_decay
        for pt, p in zip(self.value_target.parameters(), self.value_head.parameters()):
            pt.mul_(1 - d).add_(d * p.detach())

    # ---- acting (single env, online) ------------------------------------ #
    @torch.no_grad()
    def act(self, image, stoch_hist, action_hist, sample=True):
        """image: (1,1,3,64,64). stoch_hist/action_hist: lists of past posterior stochs
        (SV,4,4) and actions-into-state (A,). Returns (action one-hot (A,), stoch_t)."""
        enc = self.encoder(image)
        stoch_t = enc["stoch"][0, 0]                                # (SV,4,4)
        hist = stoch_hist + [stoch_t]
        deter_t = self.tssm.last_deter(hist, action_hist)           # (1,1,512)
        feats = (stoch_t.unsqueeze(0).unsqueeze(0), deter_t)
        dist = self.policy_head(feats)
        action = dist.sample() if sample else dist.mode()
        return action[0, 0], stoch_t
