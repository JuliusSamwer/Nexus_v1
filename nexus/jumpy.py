"""Jumpy world model — the option model p(z_term, Σr, τ, continue | Sₙ, kₙ), one level
up from EMERALD. Sₙ = (Hₙ, z_{tₙ}, kₙ).

  * Hₙ  — causal transformer over boundaries (TSSM one level up), context = hl_ctx jumps.
          Reads the step-tier h at the boundary as an INPUT (never a target), with
          h-input dropout so the imagination path (h absent) is trained, not first-seen.
  * terminal-latent head — jumpy MaskGIT over the 4×4 grid (EMERALD's MaskNetwork, one
          level up): answers skill-outcome multimodality, absorbs irrelevant detail as
          high-entropy tokens.
  * Σr / τ heads — symexp two-hot (Σr) and two-hot-on-log-τ; HL continue — Bernoulli.
  * HL actor p(kₙ|Sₙ) — the skill PRIOR (commits before execution); HL critic discounts
          by γ^τ (semi-MDP Bellman).
"""

import copy

import torch
import torch.nn as nn

from emerald_torch import nets as enets
from . import common
from .common import StochEmbed, mlp


def direct_prior_logits(masknet, cond):
    """Direct MaskGIT prior logits over the latent grid from a conditioning map
    cond (B,N,dim_model,4,4). (B,N,4,4,S,V). No sampling — for fast NLL scoring."""
    deter = cond.permute(0, 1, 3, 4, 2)
    B, N = deter.shape[:2]
    return masknet.dynamics_predictor(deter).reshape(B, N, 4, 4, masknet.S, masknet.V)


class JumpyWM(nn.Module):
    def __init__(self, cfg, num_actions):
        super().__init__()
        self.cfg = cfg
        feat = cfg.step.stoch_size * cfg.step.discrete
        H, code = cfg.hl_dim, cfg.code_dim
        cond_dim = 8 * cfg.step.dim_cnn                              # 256, MaskGIT cond width
        self.cond_dim = cond_dim

        # embeddings
        self.z_embed = StochEmbed(feat, H)
        self.h_proj = nn.Linear(cfg.step.dim_model, H)              # step-tier h as INPUT
        self.h_drop = nn.Dropout(cfg.h_dropout)

        # Hₙ recurrence over jumps (causal)
        self.jump_in, _ = enets.mlp(H + code + H, [H])
        self.Hn = enets.Transformer(H, cfg.hl_blocks, cfg.hl_heads, ff_ratio=2,
                                    drop=0.1, pos_emb=True)

        # context Sₙ -> conditioning for the outcome heads
        self.ctx, _ = enets.mlp(H + H + code, [H])                  # (Hn, z_embed, k_embed)
        self.cond_map = nn.Linear(H, 4 * 4 * cond_dim)             # -> jumpy MaskGIT cond

        # jumpy terminal-latent MaskGIT (EMERALD MaskNetwork, jumpy decode budget)
        mcfg = copy.copy(cfg.step)
        mcfg.num_decoding_steps = cfg.jumpy_decoding_steps
        mcfg.num_blocks_mask = cfg.jumpy_mask_blocks
        self.terminal = enets.MaskNetwork(mcfg, dim_model=cond_dim)

        # outcome regression heads
        self.sigma_r = nn.Linear(H, cfg.bins)                      # Σr  (symexp two-hot)
        self.tau = nn.Linear(H, cfg.bins)                         # τ   (two-hot on log τ)
        self.hl_continue = nn.Linear(H, 1)

        # HL actor (skill prior) + critic over Sₙ⁻ = (Hn, z_embed)  [prior commits w/o k]
        self.actor, _ = enets.mlp(H + H, [H]); self.actor_out = nn.Linear(H, cfg.K)
        self.critic, _ = enets.mlp(H + H, [H]); self.critic_out = nn.Linear(H, cfg.bins)

    # ---- embeddings / recurrence ---------------------------------------- #
    def embed_z(self, stoch):
        return self.z_embed(stoch)

    def jump_token(self, z_emb, k_emb, h_emb, drop_h=True):
        h = self.h_drop(h_emb) if (drop_h and self.training) else h_emb
        return self.jump_in(torch.cat([z_emb, k_emb, h], dim=-1))

    def roll_Hn(self, tokens):
        """tokens (B,N,H) -> Hₙ (B,N,H), causal over jumps."""
        return self.Hn(tokens, causal=True)

    # ---- conditioning / heads ------------------------------------------- #
    def context(self, Hn, z_emb, k_emb):
        return self.ctx(torch.cat([Hn, z_emb, k_emb], dim=-1))

    def cond(self, c):
        B, N = c.shape[:2]
        return self.cond_map(c).reshape(B, N, self.cond_dim, 4, 4)

    def outcome_heads(self, c):
        return {
            "sigma_r": common.SymLogDiscreteDist(self.sigma_r(c)),
            "tau": common.SymLogDiscreteDist(self.tau(c), low=self.cfg.tau_low, high=self.cfg.tau_high),
            "continue": common.BernoulliDist(self.hl_continue(c)),
        }

    def actor_dist(self, Hn, z_emb):
        x = self.actor(torch.cat([Hn, z_emb], dim=-1))
        return common.OneHotDist(self.actor_out(x), uniform_mix=self.cfg.step.uniform_mix)

    def critic_dist(self, Hn, z_emb, target=False):
        x = self.critic(torch.cat([Hn, z_emb], dim=-1))
        return common.SymLogDiscreteDist(self.critic_out(x))

    # ---- terminal latent NLL (fast, direct prior) for the MDL emission --- #
    def terminal_nll(self, c, z_term_stoch):
        """-log p(z_term | c) under the direct MaskGIT prior. c (B,N,H),
        z_term_stoch (B,N,SV,4,4). Returns (B,N)."""
        logits = direct_prior_logits(self.terminal, self.cond(c))   # (B,N,4,4,S,V)
        tgt = (z_term_stoch.permute(0, 1, 3, 4, 2)
               .reshape(*logits.shape[:4], self.terminal.S, self.terminal.V))
        logp = torch.log_softmax(logits, dim=-1)
        return -(tgt * logp).sum(dim=-1).sum(dim=(-3, -2, -1))       # sum over grid + S
