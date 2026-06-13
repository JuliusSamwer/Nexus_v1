"""N0 — segmentation utilities for the segment-native world model.

Phase N1 uses `seg=scheduled` (§2.2): a boundary every ~ell_bar steps with +-jitter.
Boundaries are sampled ONCE per batch and shared across batch items — a documented N1
simplification that makes every segment-local op (block-diagonal attention, per-segment
pooling, the slow streams) cleanly batched. (N3's `seg=learned` replaces this module's
`sample_schedule` with a per-sequence posterior over boundaries.)

A `Segments` object carries everything the fast/slow tiers need:
  * starts/ends/lengths     — (N,) segment geometry
  * seg_id                  — (T,) segment index per timestep
  * pos_ids                 — (T,) within-segment offset (so position embeddings are
                              segment-relative and generalize across segments)
  * is_boundary             — (T,) True at each segment's first frame t_n
  * attn_mask               — (T,T) bool, True == BLOCKED. Block-diagonal + causal: i may
                              attend j iff same segment and j<=i. THIS is the strictness:
                              no fast-tier receptive field crosses a boundary.
"""

import numpy as np
import torch


def sample_schedule(T, ell_bar, jitter, rng=None):
    """Segment-start indices in [0,T) for a scheduled segmentation (§2.2)."""
    rng = rng or np.random
    starts, t = [0], 0
    while True:
        step = int(ell_bar + rng.randint(-jitter, jitter + 1)) if jitter > 0 else ell_bar
        t += max(1, step)
        if t >= T:
            break
        starts.append(t)
    return starts


class Segments:
    def __init__(self, starts, T, device="cpu"):
        starts = sorted({int(s) for s in starts if 0 <= s < T})
        if not starts or starts[0] != 0:
            starts = [0] + starts
        ends = starts[1:] + [T]
        self.N, self.T = len(starts), T
        self.starts = torch.tensor(starts, dtype=torch.long, device=device)   # (N,)
        self.ends = torch.tensor(ends, dtype=torch.long, device=device)        # (N,)
        self.lengths = self.ends - self.starts                                 # (N,)

        ar = torch.arange(T, device=device)
        seg_id = torch.zeros(T, dtype=torch.long, device=device)
        for n, (a, b) in enumerate(zip(starts, ends)):
            seg_id[a:b] = n
        self.seg_id = seg_id                                                   # (T,)
        self.pos_ids = ar - self.starts[seg_id]                               # (T,)
        self.is_boundary = self.pos_ids == 0                                   # (T,)

        same = seg_id[:, None] == seg_id[None, :]
        causal = ar[None, :] <= ar[:, None]                                    # j<=i
        self.attn_mask = ~(same & causal)                                      # True==blocked
        self.device = device

    # ---- per-segment pooling over a (B,T,...) tensor --------------------- #
    def gather_starts(self, x):
        """x (B,T,...) -> (B,N,...) sampled at each segment's first frame t_n."""
        idx = self.starts.view(1, -1, *([1] * (x.dim() - 2))).expand(
            x.shape[0], -1, *x.shape[2:])
        return x.gather(1, idx)

    def seg_sum(self, x):
        """x (B,T) -> (B,N) sum within each segment (for Sigma_r)."""
        B = x.shape[0]
        out = x.new_zeros(B, self.N)
        return out.scatter_add_(1, self.seg_id.unsqueeze(0).expand(B, -1), x)

    def seg_min(self, x):
        """x (B,T) -> (B,N) min within each segment (continue = product of 0/1 flags)."""
        B = x.shape[0]
        out = x.new_full((B, self.N), float("inf"))
        out = out.scatter_reduce(
            1, self.seg_id.unsqueeze(0).expand(B, -1), x, reduce="amin",
            include_self=False)
        return out

    def seg_mean_feat(self, x):
        """x (B,T,D) -> (B,N,D) mean within each segment (a-summary etc.)."""
        B, _, D = x.shape
        idx = self.seg_id.view(1, -1, 1).expand(B, -1, D)
        out = x.new_zeros(B, self.N, D).scatter_add_(1, idx, x)
        return out / self.lengths.view(1, -1, 1).clamp(min=1)


def make_segments(cfg, device, rng=None):
    starts = sample_schedule(cfg.T, cfg.ell_bar, cfg.seg_jitter, rng)
    return Segments(starts, cfg.T, device)
