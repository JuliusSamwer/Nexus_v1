"""Boundary discovery — THE core. Segmentation is a latent variable inferred by an
exact semi-Markov forward–backward DP under an MDL objective, run over a top-M proposal
set so the DP stays tractable at scale.

  description length  L = Σ_segments [ jumpy-NLL + code-rate + switch-cost ]

A cut earns its place only if splitting there makes the two pieces cheaper to describe
than gluing them. Reward enters ONLY because Σr is one of the things each segment must
predict — its gradient never touches the boundary positions.

v1 decoupling (documented): the DP scores a segment with the jumpy heads conditioned on
(z_a, k) with Hₙ NULL (=0); the full Hₙ-conditioned jumpy WM is trained on the CHOSEN
segments in Stage 3. Segmentation runs under no_grad — it picks cuts; the heads improve
across stages, not through the DP. This breaks the chicken-and-egg within one backward
while keeping the MDL loop across the curriculum.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from emerald_torch import nets as enets
from .common import StochEmbed, mlp

NEG = -1e9


def switch_cost(tau, ell_bar):
    """-log of a geometric length prior with mean ell_bar (pins both degenerate
    solutions: all-length-1 pays the per-segment base, one-giant pays the tail)."""
    rho = 1.0 / ell_bar
    return -((tau - 1.0) * math.log(1.0 - rho) + math.log(rho))


class BoundaryProposer(nn.Module):
    """Cheap per-step boundary scorer; nominates the top-M candidate cuts the exact DP
    runs over. Trained to imitate the DP posterior marginals."""

    def __init__(self, cfg, num_actions, dim=128):
        super().__init__()
        feat = cfg.step.stoch_size * cfg.step.discrete
        self.embed = StochEmbed(feat, dim)
        self.act_proj = nn.Linear(num_actions, dim)
        self.net = mlp(dim, dim, 1, layers=2)

    def forward(self, stoch, action):
        x = self.embed(stoch) + self.act_proj(action)
        return self.net(x).squeeze(-1)                              # (B,T) boundary logits


def _emission(cfg, skill_enc, jumpy, z_a, z_b, seg_feat, sumr, tau):
    """MDL emission cost for a batch of P candidate segments. All inputs (P,...).
    Returns (cost (P,), k_idx (P,))."""
    P = z_a.shape[0]
    z_emb_a = jumpy.embed_z(z_a.unsqueeze(1))                      # (P,1,H)
    _, k_idx, _, _ = skill_enc.code_of(seg_feat)                   # (P,)
    k_emb = skill_enc.vq.lookup(k_idx).unsqueeze(1)               # (P,1,code)
    Hn0 = torch.zeros_like(z_emb_a)
    c = jumpy.context(Hn0, z_emb_a, k_emb)                         # (P,1,H)

    nll_term = jumpy.terminal_nll(c, z_b.unsqueeze(1)).squeeze(1)  # (P,)
    heads = jumpy.outcome_heads(c)
    nll_r = -heads["sigma_r"].log_prob(sumr.view(P, 1, 1)).reshape(P)
    nll_tau = -heads["tau"].log_prob(tau.view(P, 1, 1)).reshape(P)
    actor = jumpy.actor_dist(Hn0, z_emb_a)                         # (P,1,K)
    k_oh = F.one_hot(k_idx, cfg.K).type_as(z_emb_a).unsqueeze(1)
    code_rate = -actor.log_prob(k_oh).reshape(P)
    switch = switch_cost(tau, cfg.ell_bar)

    cost = (cfg.jumpy_nll_scale * (nll_term + nll_r + nll_tau)
            + cfg.code_rate_scale * code_rate
            + cfg.switch_cost_scale * switch)
    return cost, k_idx


def _logsumexp(vals):
    if not vals:
        return torch.tensor(NEG)
    return torch.logsumexp(torch.stack(vals), dim=0)


def _dp(Em, positions, L_max):
    """Semi-Markov forward–backward + Viterbi over proposal positions.
    Em: (Q,Q) cost, Em[i,j] valid for i<j. Returns (segments[list of (a,b)], marginals
    over interior proposal indices as a dict idx->prob)."""
    Q = len(positions)
    valid = lambda i, j: i < j and (positions[j] - positions[i]) <= L_max
    # forward (logprob = -cost)
    alpha = [torch.tensor(0.0)] + [torch.tensor(NEG) for _ in range(Q - 1)]
    for j in range(1, Q):
        alpha[j] = _logsumexp([alpha[i] - Em[i, j] for i in range(j) if valid(i, j)])
    logZ = alpha[Q - 1]
    beta = [torch.tensor(NEG) for _ in range(Q)]
    beta[Q - 1] = torch.tensor(0.0)
    for i in range(Q - 2, -1, -1):
        beta[i] = _logsumexp([-Em[i, j] + beta[j] for j in range(i + 1, Q) if valid(i, j)])
    marg = {}
    for i in range(1, Q - 1):
        marg[i] = float(torch.exp(alpha[i] + beta[i] - logZ).clamp(0, 1))
    # Viterbi (min cost)
    delta = [0.0] + [-NEG for _ in range(Q - 1)]
    back = [-1] * Q
    for j in range(1, Q):
        best, bi = float("inf"), -1
        for i in range(j):
            if valid(i, j):
                v = delta[i] + float(Em[i, j])
                if v < best:
                    best, bi = v, i
        delta[j], back[j] = best, bi
    # backtrack
    cuts, j = [], Q - 1
    while j > 0 and back[j] >= 0:
        cuts.append((back[j], j)); j = back[j]
    cuts = list(reversed(cuts))
    segments = [(positions[i], positions[k]) for i, k in cuts]
    return segments, marg


@torch.no_grad()
def segment(cfg, proposer, skill_enc, jumpy, batch):
    """Discover boundaries on a length-T HL batch. batch: stoch (B,T,SV,4,4),
    action (B,T,A), reward (B,T), cont (B,T). Returns a dict with per-element segments
    [(a,b,k)], proposer marginal targets (B,T), and stats. no_grad (picks cuts)."""
    proposer.eval(); skill_enc.eval(); jumpy.eval()
    stoch, action, reward = batch["stoch"], batch["action"], batch["reward"]
    B, T = reward.shape
    device = reward.device

    g = skill_enc.features(stoch, action)                          # (B,T,d)
    ps = skill_enc.prefix_sums(g)                                  # (B,T+1,d)
    rsum = torch.cat([reward.new_zeros(B, 1), torch.cumsum(reward, dim=1)], dim=1)  # (B,T+1)
    blogits = proposer(stoch, action)                             # (B,T)

    all_segments, marg_target = [], torch.zeros(B, T, device=device)
    seg_lens, n_segs = [], []
    for b in range(B):
        # top-M interior proposals (positions 1..T-1)
        M = min(cfg.top_M, T - 1)
        interior = (1 + torch.topk(blogits[b, 1:T], M).indices).sort().values.tolist() if M > 0 else []
        positions = [0] + interior + [T]
        Q = len(positions)
        # build all valid (i,j) pairs
        pij = [(i, j) for i in range(Q) for j in range(i + 1, Q)
               if positions[j] - positions[i] <= cfg.L_max]
        if not pij:
            all_segments.append([(0, T, 0)]); seg_lens.append(T); n_segs.append(1); continue
        ai = torch.tensor([positions[i] for i, _ in pij], device=device)
        bj = torch.tensor([positions[j] for _, j in pij], device=device)
        z_a = stoch[b, ai]                                         # (P,SV,4,4)
        z_b = stoch[b, (bj - 1).clamp(max=T - 1)]                  # terminal latent
        seg_feat = (ps[b, bj] - ps[b, ai]) / (bj - ai).clamp(min=1).unsqueeze(-1)
        sumr = rsum[b, bj] - rsum[b, ai]
        tau = (bj - ai).float()
        cost, k_idx = _emission(cfg, skill_enc, jumpy, z_a, z_b, seg_feat, sumr, tau)
        Em = torch.full((Q, Q), float("inf"), device=device)
        kmap = {}
        for p, (i, j) in enumerate(pij):
            Em[i, j] = cost[p]; kmap[(i, j)] = int(k_idx[p])
        segments, marg = _dp(Em.cpu(), positions, cfg.L_max)
        # attach codes to MAP segments
        pos2idx = {pos: ix for ix, pos in enumerate(positions)}
        seg_with_k = [(a, bb, kmap.get((pos2idx[a], pos2idx[bb]), 0)) for a, bb in segments]
        all_segments.append(seg_with_k)
        for ix, pr in marg.items():
            marg_target[b, positions[ix]] = pr
        seg_lens += [bb - a for a, bb, _ in seg_with_k]
        n_segs.append(len(seg_with_k))

    stats = {
        "mean_seg_len": float(sum(seg_lens) / max(len(seg_lens), 1)),
        "mean_n_segs": float(sum(n_segs) / max(len(n_segs), 1)),
    }
    proposer.train(); skill_enc.train(); jumpy.train()
    return {"segments": all_segments, "marg_target": marg_target, "stats": stats}
