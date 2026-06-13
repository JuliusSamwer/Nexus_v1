"""Nexus — Segment-Native World Model (Strict Bottleneck), build outline v1.

The trajectory's latent is two token streams:
  * FAST frame tokens z_t (EMERALD's 4x4 spatial categoricals), generated segment-locally;
  * SLOW segment tokens u_n (G categoricals), the ONLY information path across boundaries.

Strictness (§2.3): no component except the slow stream has a receptive field crossing a
segment boundary, except through the bounded leak channel W (width `w`, default 0). The
fast state is never zeroed — it is REBUILT at each boundary from (u_n, z_{t_n} [, W·h_prev]).

Modules:
  config.py    — composes EMERALD parts-library sizes with slow-tier params; w/G dials.
  segments.py  — N0: scheduled boundaries, seg masks, per-segment pooling.
  common.py    — shared embeds / slow-token codebook / dist re-exports.
  fast.py      — §2.3 fast tier (segment-local TSSM + u-FiLM MaskGIT + boundary rebuild).
  slow.py      — §2.4 slow posterior (segment encoder) + §2.5 slow prior / jumpy model.
  model.py     — NexusWM: ties the tiers, the §4 loss, the §8 diagnostics.
  replay.py    — stream replay (cross-episode length-T windows).
  train.py     — N1 loop (WM-only, seg=scheduled), the {w}x{G} grid.

Self-contained except for importing `emerald_torch` (the parts library).
"""
