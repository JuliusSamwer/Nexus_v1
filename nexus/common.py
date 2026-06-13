"""Shared building blocks for the segment-native tiers (reusing emerald_torch parts)."""

import torch
import torch.nn as nn

from emerald_torch import nets as enets
from emerald_torch import dists as edists


class StochEmbed(nn.Module):
    """Embed a spatial categorical latent stoch (..., S*V, 4, 4) -> (..., out_dim).
    Same shape trick as EMERALD's TSSM encoder, standalone for the slow tier."""

    def __init__(self, feat_size, out_dim, reduced=128):
        super().__init__()
        self.feat_size = feat_size
        self.conv = nn.Conv2d(feat_size, reduced, 1)
        self.norm = enets.ChLayerNorm(reduced)
        self.act = nn.SiLU()
        self.lin = nn.Linear(4 * 4 * reduced, out_dim)

    def forward(self, stoch):
        lead = stoch.shape[:-3]
        x = stoch.reshape(-1, self.feat_size, 4, 4)
        x = self.act(self.norm(self.conv(x)))
        return self.lin(x.flatten(1)).reshape(*lead, -1)


def mlp(dim_in, hidden, out_dim, layers=2):
    trunk, d = enets.mlp(dim_in, [hidden] * layers)
    return nn.Sequential(trunk, nn.Linear(d, out_dim))


class SlowTokenEmbed(nn.Module):
    """Embed the G categorical slow tokens u_n (one-hot, B,N,G,C) -> (B,N,emb_dim).
    Per-group codebook, summed over groups (the slow-tier analog of a stoch embed)."""

    def __init__(self, G, num_classes, emb_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(G, num_classes, emb_dim) * 0.02)

    def forward(self, onehot):
        return torch.einsum("bngc,gce->bne", onehot, self.weight)


# Re-export distributions the slow tier needs.
OneHotDist = edists.OneHotDist
SymLogDiscreteDist = edists.SymLogDiscreteDist
BernoulliDist = edists.BernoulliDist
MSEDist = edists.MSEDist
kl_onehot = edists.kl_onehot
symlog, symexp = edists.symlog, edists.symexp
