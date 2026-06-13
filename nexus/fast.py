"""Fast tier (§2.3) — segment-local TSSM + u-conditioned MaskGIT, with the strict
bottleneck.

The strictness lives in two places, both here:
  1. `attn_mask = segs.attn_mask` — the transformer is block-diagonal + causal, so no
     fast-tier receptive field crosses a boundary.
  2. The state is REBUILT at every boundary by `Init(u_n, z_{t_n}, W·h_{t_n-1})`
     (§2.3): the boundary frame's full latent z_{t_n} always crosses (it is observed),
     the slow token group u_n crosses, and a w-dim learned projection W of the outgoing
     fast state is the ONLY unobservable-memory channel. w=0 => fully strict (default).

Everything else (encode_stoch, mix, deter_to_dec, the MaskGIT network, the two-hot
reward head, the Bernoulli continue head, the image decoder) is EMERALD, reused verbatim
from `emerald_torch.nets` — this tier is "EMERALD's recurrence, unchanged" plus the
boundary rebuild and the u-FiLM conditioning.
"""

import torch
import torch.nn as nn

from emerald_torch import nets as enets
from .common import kl_onehot


def _masked_mean(x, keep):
    """Mean of x (B,T,...) over batch + the positions where keep (T,) is True."""
    while keep.dim() < x.dim():
        keep = keep.unsqueeze(0) if keep.dim() == 1 else keep.unsqueeze(-1)
    w = keep.expand_as(x)
    return (x * w).sum() / w.sum().clamp(min=1.0)


class FastTier(nn.Module):
    def __init__(self, cfg, num_actions):
        super().__init__()
        self.cfg = cfg
        step = cfg.step
        self.tssm = enets.TSSM(step, num_actions)          # encode_stoch/mix/deter_to_dec/
        self.decoder = enets.ImageDecoder(step)            #   transformer/mask_network reused
        self.reward_head = enets.RewardHead(step)
        self.continue_head = enets.ContinueHead(step)

        dm = step.dim_model
        self.cond_dim = 8 * step.dim_cnn                   # MaskGIT conditioning width (256)
        # boundary rebuild Init(u_n, z_{t_n}[, W·h_prev])
        leak_in = dm + cfg.u_emb_dim + cfg.w
        self.init_mlp, _ = enets.mlp(leak_in, [dm, dm])
        self.leak = nn.Linear(dm, cfg.w, bias=False) if cfg.w > 0 else None
        # u enters the fast MaskGIT prior via additive FiLM on the conditioning map
        self.u_film = nn.Linear(cfg.u_emb_dim, self.cond_dim)

    # ---- segment-local dynamics ---------------------------------------- #
    def _dynamics(self, z, action, segs, u_emb, leak_src):
        """z (B,T,SV,4,4) posterior stoch; u_emb (B,N,u_emb_dim). Returns deter (B,T,dm).
        leak_src: detached deter from a prior pass (w>0 two-pass) or None."""
        B, T = z.shape[:2]
        dm = self.cfg.step.dim_model
        # within-segment tokens (boundary slots overwritten below)
        prev = torch.cat([z.new_zeros(B, 1, *z.shape[2:]), z[:, :-1]], dim=1)
        x = self.tssm.mix(self.tssm.encode_stoch(prev), action)           # (B,T,dm)

        zb_emb = self.tssm.encode_stoch(segs.gather_starts(z))            # (B,N,dm)
        parts = [zb_emb, u_emb]
        if self.leak is not None:
            if leak_src is None:
                leak = zb_emb.new_zeros(B, segs.N, self.leak.out_features)
            else:
                prev_idx = (segs.starts - 1).clamp(min=0)                 # (N,)
                leak = self.leak(leak_src[:, prev_idx])                   # (B,N,w)
                leak = leak * (segs.starts > 0).to(leak.dtype).view(1, -1, 1)
            parts.append(leak)
        init_tok = self.init_mlp(torch.cat(parts, dim=-1))               # (B,N,dm)

        x = x.index_copy(1, segs.starts, init_tok)
        return self.tssm.transformer(x, attn_mask=segs.attn_mask, pos_ids=segs.pos_ids)

    def run(self, z, action, segs, u_emb):
        """Full dynamics with the (detached) leak two-pass when w>0."""
        deter = self._dynamics(z, action, segs, u_emb, leak_src=None)
        if self.leak is not None:
            deter = self._dynamics(z, action, segs, u_emb, leak_src=deter.detach())
        return deter

    # ---- training loss -------------------------------------------------- #
    def loss(self, image, z, logits_post, action, reward, cont, segs, u_emb):
        cfg, step = self.cfg, self.cfg.step
        B, T = z.shape[:2]
        u_t = u_emb[:, segs.seg_id]                                       # (B,T,u_emb_dim)
        deter = self.run(z, action, segs, u_emb)
        feats = (z, deter)

        # decoder reconstruction over ALL frames
        loss_image = -self.decoder(feats).log_prob(image).mean()

        # u-FiLM conditioned MaskGIT prior
        cond = self.tssm.deter_to_dec(deter) + self.u_film(u_t)[..., None, None]
        logits, logits_masked, mmask = self.tssm.mask_network.train_logits(z, cond)

        # KL (EMERALD's 0.5/0.1 split, free bits) — boundary frames excluded from the
        # fast prior (their z_{t_n} crossed into Init; the slow grounding head predicts them).
        nb = ~segs.is_boundary                                           # (T,)
        kl_pr = kl_onehot(logits_post.detach(), logits, step.uniform_mix).sum(-1)  # (B,T,4,4)
        kl_po = kl_onehot(logits_post, logits.detach(), step.uniform_mix).sum(-1)
        loss_kl_prior = _masked_mean(kl_pr.clamp(min=step.free_nats), nb)
        loss_kl_post = _masked_mean(kl_po.clamp(min=step.free_nats), nb)
        loss_kl_mask = z.new_zeros(())
        if logits_masked is not None:
            m = mmask.float()
            kl_m = kl_onehot(logits_post.detach(), logits_masked, step.uniform_mix)
            kl_m = step.discrete * (kl_m * m).sum((-3, -2, -1)) / (m.sum((-3, -2, -1)) + 1e-8)
            loss_kl_mask = _masked_mean(kl_m.clamp(min=step.free_nats), nb)

        loss_reward = -self.reward_head(feats).log_prob(reward.unsqueeze(-1)).mean()
        loss_cont = -self.continue_head(feats).log_prob(cont.unsqueeze(-1)).mean()

        loss = (step.loss_decoder_scale * loss_image
                + step.loss_kl_prior_scale * loss_kl_prior
                + step.loss_kl_post_scale * loss_kl_post
                + step.loss_kl_mask_scale * loss_kl_mask
                + step.loss_reward_scale * loss_reward
                + step.loss_continue_scale * loss_cont)

        # ---- diagnostic: per-position fast-prior NLL (§8.1) ------------- #
        with torch.no_grad():
            tgt = logits_post.argmax(-1)                                  # (B,T,4,4,S)
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum((-3, -2, -1))  # (B,T)

        metrics = {
            "fast_loss": loss.item(), "fast_image": loss_image.item(),
            "fast_kl_prior": loss_kl_prior.item(), "fast_kl_post": loss_kl_post.item(),
            "fast_kl_mask": float(loss_kl_mask.detach()), "fast_reward": loss_reward.item(),
            "fast_cont": loss_cont.item(),
        }
        return loss, metrics, {"deter": deter, "fast_nll": nll}
