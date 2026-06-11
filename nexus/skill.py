"""Skill codebook (VQ) + skill encoder q(k|segment) — the POSTERIOR over skills.

The skill encoder reads a whole segment in hindsight (bidirectional, sees the future),
contextualizes per-step latents, and a segment's representation is the masked mean of
those features. Vector-quantizing that representation names the skill kₙ — giving an
inspectable, countable skill inventory (CompILE's segment→code, with a VQ-VAE/FSQ
codebook). FSQ is noted as the collapse fallback; v1 uses EMA-VQ + dead-code restart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from emerald_torch import nets as enets
from .common import StochEmbed


class VectorQuantizer(nn.Module):
    def __init__(self, K, dim, decay=0.99, commit=0.25, restart_thresh=1.0, eps=1e-5):
        super().__init__()
        self.K, self.dim = K, dim
        self.decay, self.commit, self.restart_thresh, self.eps = decay, commit, restart_thresh, eps
        embed = torch.randn(K, dim) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.zeros(K))

    def lookup(self, idx):
        return F.embedding(idx, self.embed)

    def forward(self, z):
        """z: (..., dim). Returns z_q (straight-through), idx, commit_loss, perplexity."""
        flat = z.reshape(-1, self.dim)
        d = torch.cdist(flat, self.embed)                  # (N, K)
        idx = d.argmin(dim=1)                              # (N,)
        onehot = F.one_hot(idx, self.K).type_as(flat)
        z_q = self.lookup(idx).reshape_as(z)
        z_q_st = z + (z_q - z).detach()                    # straight-through
        commit_loss = self.commit * F.mse_loss(z, z_q.detach())

        if self.training:
            with torch.no_grad():
                n = onehot.sum(dim=0)
                self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                dw = onehot.t() @ flat
                self.embed_avg.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (total + self.K * self.eps) * total
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))
                # dead-code restart: reseed unused codes from current batch
                dead = self.cluster_size < self.restart_thresh
                if dead.any() and flat.shape[0] > 0:
                    ridx = torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)
                    self.embed[dead] = flat[ridx].detach()
                    self.embed_avg[dead] = flat[ridx].detach()
                    self.cluster_size[dead] = 1.0

        probs = onehot.mean(dim=0)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        return z_q_st, idx.reshape(z.shape[:-1]), commit_loss, perplexity


class SkillEncoder(nn.Module):
    def __init__(self, cfg, num_actions):
        super().__init__()
        feat = cfg.step.stoch_size * cfg.step.discrete
        self.embed = StochEmbed(feat, cfg.skill_enc_dim)
        self.action_proj = nn.Linear(num_actions, cfg.skill_enc_dim)
        self.transformer = enets.Transformer(
            cfg.skill_enc_dim, cfg.skill_enc_blocks, cfg.skill_enc_heads,
            ff_ratio=2, drop=0.1, pos_emb=True)
        self.to_code = nn.Linear(cfg.skill_enc_dim, cfg.code_dim)
        self.vq = VectorQuantizer(cfg.K, cfg.code_dim, decay=cfg.vq_ema,
                                  commit=cfg.vq_commit, restart_thresh=cfg.vq_restart_thresh)

    def features(self, stoch, action):
        """Contextualized per-step features over a length-T window. (B,T,...) -> (B,T,dim)."""
        return self.transformer(self.embed(stoch) + self.action_proj(action), causal=False)

    @staticmethod
    def prefix_sums(g):
        """Cumulative sums with a leading zero, for O(1) segment means. (B,T,d)->(B,T+1,d)."""
        z = g.new_zeros(g.shape[0], 1, g.shape[2])
        return torch.cat([z, torch.cumsum(g, dim=1)], dim=1)

    def segment_mean(self, ps, a, b):
        """Mean of features over [a,b) from prefix sums ps (B,T+1,d). a,b broadcastable."""
        return (ps.gather(1, b) - ps.gather(1, a)) / (b - a).clamp(min=1)

    def code_of(self, seg_feat):
        """Pooled segment feature (..., dim) -> (z_q, idx, commit_loss, perplexity)."""
        return self.vq(self.to_code(seg_feat))
