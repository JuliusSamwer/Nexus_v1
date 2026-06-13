"""NexusWM — the segment-native world model (strict bottleneck), Phases N0/N1.

Ties together: the shared frame encoder (EMERALD, the encoder "serves two masters" — both
clocks backprop into it), the fast tier (§2.3, segment-local with the boundary rebuild +
w-leak), the slow posterior (§2.4), and the slow prior / jumpy model (§2.5). Computes the
§4 loss and the §8 self-auditing diagnostics (post-boundary NLL spike, grounding-vs-copy
advantage, u-stream health).

N1 = full WM with seg=scheduled. No actor yet (that is N2); this trains the world model
only and reports whether (u_n, z_{t_n}) suffices to re-init prediction across a boundary.
"""

import torch
import torch.nn as nn

from emerald_torch import nets as enets
from .fast import FastTier
from .slow import SlowPosterior, SlowPrior
from .common import kl_onehot


def _grid(z, S, V):
    """stoch (B,N,S*V,4,4) -> grid (B,N,4,4,S,V) (S-major, matching the encoder)."""
    return z.permute(0, 1, 3, 4, 2).reshape(*z.shape[:2], 4, 4, S, V)


class NexusWM(nn.Module):
    def __init__(self, cfg, num_actions):
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        self.encoder = enets.Encoder(cfg.step)             # shared frame encoder
        self.fast = FastTier(cfg, num_actions)
        self.post = SlowPosterior(cfg, num_actions)
        self.prior = SlowPrior(cfg, num_actions)

    # ---- grounding CE (direct + masked), reusing EMERALD's MaskNetwork ---- #
    def _grounding(self, z_next, cond):
        S, V = self.prior.ground.S, self.prior.ground.V
        cond_map = self.prior.ground_cond(cond)
        logits, logits_masked, mmask = self.prior.ground.train_logits(z_next, cond_map)
        tgt = _grid(z_next, S, V)
        direct = -(tgt * torch.log_softmax(logits, dim=-1)).sum(-1).sum((-3, -2, -1))  # (B,N)
        if logits_masked is not None:
            ce = -(tgt * torch.log_softmax(logits_masked, dim=-1)).sum(-1)
            m = mmask.float()
            masked = (ce * m).sum((-3, -2, -1)) / (m.sum((-3, -2, -1)) + 1e-8)
        else:
            masked = torch.zeros_like(direct)
        return direct, masked

    # ---- full §4 loss --------------------------------------------------- #
    def loss(self, batch, segs):
        cfg, step = self.cfg, self.cfg.step
        image, action = batch["image"], batch["action"]
        reward, cont = batch["reward"], batch["cont"]
        B, T = reward.shape

        enc = self.encoder(image)
        z, logits_post = enc["stoch"], enc["logits"]                  # (B,T,SV,4,4), grid

        # slow posterior: u_n per segment
        q = self.post(z, action, segs)
        u_emb = q["emb"]                                             # (B,N,emb_dim)

        # fast tier (segment-local, all losses within segments)
        fast_loss, fast_m, fast_aux = self.fast.loss(
            image, z, logits_post, action, reward, cont, segs, u_emb)

        # ---- slow targets ------------------------------------------------ #
        zb = segs.gather_starts(z)                                   # (B,N,SV,4,4)
        zb_logits = segs.gather_starts(logits_post)                 # (B,N,4,4,S,V)
        tau = segs.lengths.to(z.dtype).unsqueeze(0).expand(B, -1)    # (B,N)
        a_summary = segs.seg_mean_feat(action)                      # (B,N,A)
        reward_sum = segs.seg_sum(reward)                          # (B,N)
        cont_seg = segs.seg_min(cont)                             # (B,N)
        z_next = torch.cat([zb[:, 1:], zb[:, -1:]], dim=1)          # (B,N,SV,4,4)
        valid_next = (torch.arange(segs.N, device=z.device) < segs.N - 1).to(z.dtype)  # (N,)

        # ---- slow prior + heads ----------------------------------------- #
        ctx = self.prior.context(u_emb, zb, tau, a_summary)         # (B,N,d)
        u_prior_logits = self.prior.u_prior_dist(ctx)              # (B,N,G,C)
        cond = self.prior.cond(ctx, u_emb, zb)
        heads = self.prior.outcome_heads(cond)

        # slow KL (EMERALD's 0.5/0.1 balanced split, free bits, unimix)
        kl_pr = kl_onehot(q["logits"].detach(), u_prior_logits, cfg.slow_uniform_mix)  # (B,N,G)
        kl_po = kl_onehot(q["logits"], u_prior_logits.detach(), cfg.slow_uniform_mix)
        loss_u_kl = (step.loss_kl_prior_scale * kl_pr.clamp(min=cfg.slow_free_bits).mean()
                     + step.loss_kl_post_scale * kl_po.clamp(min=cfg.slow_free_bits).mean())

        # grounding (weighted highest); mask the last segment (no next boundary frame)
        g_direct, g_masked = self._grounding(z_next, cond)
        gden = valid_next.sum().clamp(min=1) * B
        loss_ground = ((g_direct + g_masked) * valid_next).sum() / gden

        loss_tau = -heads["tau"].log_prob(tau.unsqueeze(-1)).mean()
        loss_r = -heads["sigma_r"].log_prob(reward_sum.unsqueeze(-1)).mean()
        loss_cont = -heads["cont"].log_prob(cont_seg.unsqueeze(-1)).mean()

        slow_loss = (loss_u_kl
                     + cfg.ground_scale * loss_ground
                     + cfg.slow_tau_scale * loss_tau
                     + cfg.slow_r_scale * loss_r
                     + cfg.slow_cont_scale * loss_cont)

        loss = fast_loss + cfg.lambda_slow * slow_loss

        metrics = dict(fast_m)
        metrics.update({
            "loss": loss.item(), "slow_loss": slow_loss.item(),
            "u_kl": (kl_pr.mean().item()), "ground_nll": loss_ground.item(),
            "slow_tau": loss_tau.item(), "slow_r": loss_r.item(),
            "slow_cont": loss_cont.item(),
        })
        # diagnostics (§8.1, §8.2, §8.4)
        with torch.no_grad():
            metrics.update(self._diagnostics(
                segs, fast_aux["fast_nll"], g_direct, valid_next, z_next, zb_logits,
                q["logits"]))
        return loss, metrics

    # ---- §8 diagnostics ------------------------------------------------- #
    def _diagnostics(self, segs, fast_nll, g_direct, valid_next, z_next, zb_logits, u_logits):
        out = {}
        # 8.1 post-boundary NLL spike: fast-prior NLL by within-segment offset
        pos = segs.pos_ids
        mid = pos >= max(8, int(segs.lengths.float().mean().item() // 2))
        mid_nll = fast_nll[:, mid].mean().item() if mid.any() else float("nan")
        out["fast_nll_mid"] = mid_nll
        for k in range(1, 6):
            sel = pos == k
            out[f"fast_nll_off{k}"] = fast_nll[:, sel].mean().item() if sel.any() else float("nan")
        if out.get("fast_nll_off1") == out.get("fast_nll_off1") and mid_nll == mid_nll:
            out["post_boundary_spike"] = out["fast_nll_off1"] - mid_nll

        # 8.2 slow-prior advantage: grounding NLL vs copy-last-boundary-frame baseline
        S, V = self.prior.ground.S, self.prior.ground.V
        tgt = _grid(z_next, S, V)                                   # (B,N,4,4,S,V) one-hot
        copy_logp = torch.log_softmax(zb_logits, dim=-1)
        copy_nll = -(tgt * copy_logp).sum(-1).sum((-3, -2, -1))     # (B,N)
        den = valid_next.sum().clamp(min=1) * z_next.shape[0]
        out["ground_nll_diag"] = (g_direct * valid_next).sum().item() / den.item()
        out["copy_nll_diag"] = (copy_nll * valid_next).sum().item() / den.item()
        out["slow_advantage"] = out["copy_nll_diag"] - out["ground_nll_diag"]

        # 8.4 u-stream health: per-token perplexity (collapse check)
        probs = torch.softmax(u_logits, dim=-1).mean(dim=(0, 1))    # (G,C)
        ent = -(probs * torch.log(probs + 1e-8)).sum(-1)           # (G,)
        out["u_perplexity"] = torch.exp(ent).mean().item()
        return out
