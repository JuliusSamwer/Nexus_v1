"""Shared building blocks for the Nexus skill tier (reusing emerald_torch primitives)."""

import torch
import torch.nn as nn

from emerald_torch import nets as enets
from emerald_torch import dists as edists


class StochEmbed(nn.Module):
    """Embed a spatial categorical latent stoch (..., S*V, 4, 4) -> (..., out_dim).
    Same shape trick as EMERALD's TSSM encoder, but standalone for the skill tier."""

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


# Re-export the distributions the skill tier needs.
OneHotDist = edists.OneHotDist
SymLogDiscreteDist = edists.SymLogDiscreteDist
BernoulliDist = edists.BernoulliDist
kl_onehot = edists.kl_onehot
symlog, symexp = edists.symlog, edists.symexp
