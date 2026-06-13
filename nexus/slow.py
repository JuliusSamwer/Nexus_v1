"""Slow tier — the segment-level streams.

  * SlowPosterior (§2.4): "watch the segment, summarize it." A bidirectional transformer
    over a segment's (z, a) sequence; G learned query tokens cross-attend and pool to the
    logits for the G categorical slow tokens u_n. Straight-through sample + unimix.

  * SlowPrior / jumpy model (§2.5): the outer generative loop, and the ONLY module whose
    receptive field crosses boundaries. A causal transformer over the interleaved slow
    history (u_m, z_{t_m}, tau_m, a-summary_m). Heads:
        p(u_n | history)              — segment-token prior (KL target for the posterior)
        p(tau_n | history, u_n)       — duration, two-hot on log tau
        p(z_{t_{n+1}} | history, u_n) — grounding: jumpy MaskGIT over the next boundary frame
        p(Sigma_r_n | history, u_n)   — symexp two-hot
        p(c_n | history, u_n)         — continue

Reuses EMERALD's MaskNetwork (grounding), two-hot heads, and categorical-ST machinery.
"""

import copy
import math

import torch
import torch.nn as nn

from emerald_torch import nets as enets
from . import common
from .common import (StochEmbed, SlowTokenEmbed, OneHotDist, SymLogDiscreteDist,
                     BernoulliDist, kl_onehot)


class SlowPosterior(nn.Module):
    def __init__(self, cfg, num_actions):
        super().__init__()
        self.cfg = cfg
        feat = cfg.step.stoch_size * cfg.step.discrete
        d = cfg.post_dim
        self.z_embed = StochEmbed(feat, d)
        self.a_proj = nn.Linear(num_actions, d)
        self.tr = enets.Transformer(d, cfg.post_blocks, cfg.post_heads, pos_emb=True)
        self.queries = nn.Parameter(torch.randn(cfg.G, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, cfg.post_heads, batch_first=True)
        self.head_w = nn.Parameter(torch.randn(cfg.G, d, cfg.u_classes) * (d ** -0.5))
        self.head_b = nn.Parameter(torch.zeros(cfg.G, cfg.u_classes))
        self.token_embed = SlowTokenEmbed(cfg.G, cfg.u_classes, cfg.u_emb_dim)

    def forward(self, z, action, segs):
        cfg = self.cfg
        B = z.shape[0]
        logits = []
        for n in range(segs.N):
            a, b = int(segs.starts[n]), int(segs.ends[n])
            tok = self.z_embed(z[:, a:b]) + self.a_proj(action[:, a:b])   # (B,tau,d)
            h = self.tr(tok, causal=False)                               # (B,tau,d)
            q = self.queries.unsqueeze(0).expand(B, -1, -1)              # (B,G,d)
            pooled, _ = self.pool(q, h, h)                              # (B,G,d)
            lg = torch.einsum("bgd,gdc->bgc", pooled, self.head_w) + self.head_b
            logits.append(lg)
        u_logits = torch.stack(logits, dim=1)                           # (B,N,G,C)
        dist = OneHotDist(u_logits, cfg.slow_uniform_mix)
        u_oh = dist.rsample()                                           # ST (B,N,G,C)
        u_emb = self.token_embed(u_oh)                                  # (B,N,emb_dim)
        return {"logits": u_logits, "onehot": u_oh, "emb": u_emb}


class SlowPrior(nn.Module):
    """The jumpy model — slow prior + grounding + outcome heads."""

    def __init__(self, cfg, num_actions):
        super().__init__()
        self.cfg = cfg
        feat = cfg.step.stoch_size * cfg.step.discrete
        d = cfg.slow_dim
        self.cond_dim = 8 * cfg.step.dim_cnn                            # 256, MaskGIT cond
        self.G, self.C = cfg.G, cfg.u_classes

        # history-token embedding: (u_m, z_{t_m}, tau_m, a-summary_m)
        self.zb_embed = StochEmbed(feat, d)
        self.u_proj = nn.Linear(cfg.u_emb_dim, d)
        self.tau_proj = nn.Linear(1, d)
        self.a_proj = nn.Linear(num_actions, d)
        self.jump_in, _ = enets.mlp(4 * d, [d])
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.tr = enets.Transformer(d, cfg.slow_blocks, cfg.slow_heads, pos_emb=True)

        # u prior head p(u_n | history<n)
        self.u_prior = nn.Linear(d, cfg.G * cfg.u_classes)

        # u-conditioned context for the outcome heads (history<n, u_n, z_{t_n})
        self.cond_mlp, _ = enets.mlp(d + d + d, [d])
        self.tau_head = nn.Linear(d, cfg.step.bins)
        self.r_head = nn.Linear(d, cfg.step.bins)
        self.cont_head = nn.Linear(d, 1)
        self.ground_map = nn.Linear(d, self.cond_dim * 4 * 4)
        mcfg = copy.copy(cfg.step)
        mcfg.num_decoding_steps = cfg.ground_decoding_steps
        mcfg.num_blocks_mask = cfg.ground_mask_blocks
        self.ground = enets.MaskNetwork(mcfg, dim_model=self.cond_dim)
        self.tau_lo, self.tau_hi = 0.0, math.log(1.0 + cfg.tau_max) + 0.5

    # ---- history rollup ------------------------------------------------- #
    def context(self, u_emb, zb, tau, a_summary):
        """All (B,N,...). Returns ctx (B,N,d): ctx[:,n] has seen history strictly < n."""
        tau_feat = common.symlog(tau).unsqueeze(-1)                    # (B,N,1)
        tok = self.jump_in(torch.cat([
            self.u_proj(u_emb), self.zb_embed(zb),
            self.tau_proj(tau_feat), self.a_proj(a_summary)], dim=-1))  # (B,N,d)
        B, N = tok.shape[:2]
        inp = torch.cat([self.bos.expand(B, -1, -1), tok], dim=1)      # (B,N+1,d)
        ctx = self.tr(inp, causal=True)                               # (B,N+1,d)
        return ctx[:, :N]                                             # (B,N,d) predicts seg n

    def cond(self, ctx, u_emb, zb):
        return self.cond_mlp(torch.cat([ctx, self.u_proj(u_emb), self.zb_embed(zb)], dim=-1))

    def ground_cond(self, c):
        B, N = c.shape[:2]
        return self.ground_map(c).reshape(B, N, self.cond_dim, 4, 4)

    def u_prior_dist(self, ctx):
        B, N = ctx.shape[:2]
        lg = self.u_prior(ctx).reshape(B, N, self.G, self.C)
        return lg

    def outcome_heads(self, c):
        return {
            "tau": SymLogDiscreteDist(self.tau_head(c), low=self.tau_lo, high=self.tau_hi),
            "sigma_r": SymLogDiscreteDist(self.r_head(c)),
            "cont": BernoulliDist(self.cont_head(c)),
        }
