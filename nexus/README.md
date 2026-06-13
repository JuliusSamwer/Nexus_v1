# `nexus/` — Segment-Native World Model (Strict Bottleneck)

The trajectory's latent is **two token streams**: fast frame tokens `z_t` generated
*segment-locally*, and slow segment tokens `u_n` that are the **only** information path
across boundaries. Generation runs coarse-to-fine in time: segments first, frames on
demand.

**Strictness commitment (§2.3):** no component except the slow stream has a receptive
field that crosses a segment boundary — except through the bounded leak channel `W`, whose
width `w` is a config dial with `w=0` (fully strict) as the default. The fast state is
never zeroed; it is **rebuilt** at each boundary from `(u_n, z_{t_n} [, W·h_prev])`. The
boundary frame's full latent `z_{t_n}` always crosses; what the bottleneck cuts is
unobservable internal memory only.

Self-contained except for importing `emerald_torch` (the parts library: encoder, decoder,
TSSM recurrence, MaskGIT, two-hot heads, categorical-ST machinery — reused verbatim).

## Modules

| File | Role |
|------|------|
| `config.py` | Composes EMERALD parts-library sizes with slow-tier params. Headline dials: `w` (leak width, §2.3) and `G` (slow tokens/segment, §2.4). |
| `segments.py` | **N0.** `seg=scheduled`: boundary every `ell_bar`±jitter. Builds `seg_id`, within-segment `pos_ids`, the block-diagonal-causal `attn_mask` (this tensor *is* the strictness), and per-segment pooling (Σr, continue, a-summary). |
| `common.py` | Shared stoch embed, slow-token codebook, dist re-exports. |
| `fast.py` | **§2.3 fast tier.** Segment-local TSSM with the boundary rebuild `Init(u_n, z_{t_n}, W·h_prev)` + w-leak (detached two-pass when `w>0`), u-FiLM on the MaskGIT prior, EMERALD reward/continue/decoder. Boundary frames are excluded from the fast prior (the slow grounding head owns them). |
| `slow.py` | **§2.4 slow posterior** (bidirectional segment encoder; `G` query tokens cross-attend → `G` categorical-256 tokens `u_n`; ST + unimix) and **§2.5 slow prior / jumpy model** (causal transformer over slow history; heads for `u`, `τ`, grounding MaskGIT `z_{t_{n+1}}`, `Σr`, continue). The only boundary-crossing module. |
| `model.py` | `NexusWM`: shared encoder + the tiers, the §4 loss, and the §8 diagnostics. |
| `replay.py` | Stream replay — length-T windows across episode seams (random Crafter episodes are ~170 steps; T=256 needs streaming). |
| `train.py` | **N1** loop (WM-only, `seg=scheduled`). Random-policy collection; the `{w}×{G}` grid via `--w`/`--G`. |

## Run

**Smoke (shape shakeout, CPU/MPS):**
```bash
python3 -m nexus.train --configs tiny --device cpu --logdir logdir/nexus_smoke --w 0 --G 4 --steps 200
```

**N1 run (GPU). Start with the two corners of the §11.4 grid:**
```bash
# strict reference
python3 -m nexus.train --configs crafter --device cuda --logdir logdir/nexus_w0_G4  --w 0  --G 4 --steps 100000
# leak reference
python3 -m nexus.train --configs crafter --device cuda --logdir logdir/nexus_w64_G4 --w 64 --G 4 --steps 100000
```
Then fill the grid `{w∈0,16,64} × {G∈2,4,8}`. The **post-boundary spike per cell** is the
N1 deliverable.

## What to watch (§8 — logged every `log_every` to `metrics.jsonl`)

- **`post_boundary_spike`** *(§8.1, the headline)* — fast-prior NLL at offset 1 after a
  boundary minus mid-segment NLL. Small at `w=0` ⇒ strictness is free, `(u,z_b)` suffices.
  Shrinking as `w` grows ⇒ unobservable environmental memory genuinely needed — *that's the
  figure*. Persisting large at `w=64` ⇒ an Init/`G` problem, not a leak problem (F1).
  (Per-offset `fast_nll_off1..5` and `fast_nll_mid` are also logged.)
- **`slow_advantage`** *(§8.2)* — `copy_nll_diag − ground_nll_diag`: grounding's next-
  boundary-frame NLL vs copying the current boundary frame. The tier-earns-its-existence
  number; should climb positive.
- **`u_perplexity`** *(§8.4)* — per-token perplexity of the slow posterior (collapse alarm,
  out of `u_classes`); **`u_kl`** — code rate (posterior actually informs the prior).

## N1 simplifications (honest, flagged not hidden)

1. **Boundaries shared across a batch** (one jittered schedule per batch) so every
   segment-local op is cleanly batched. N3's `seg=learned` replaces this with a per-
   sequence boundary posterior.
2. **`w>0` leak is detached across segments** (two forward passes; the leak source is the
   previous pass's deter, stop-grad). A narrow *environmental-memory* channel, not a
   gradient bypass — consistent with the bottleneck's intent.
3. **Stream replay crosses episode seams** inside a window (cont-flagged). Random episodes
   are too short for within-episode T=256.
4. **No actor yet** — N1 is WM-only and collects with a random policy. The frozen-recipe AC
   (N2) replaces `collect_episode` and reads the new state `(h_t, z_t, u_n)`.

Deferred to later phases (out of N1, per §10): learned segmenter (N3), `γ^τ` slow critic +
jumpy value backup (N4), rollout-depth curve (§8.3), the soft-bottleneck flag (§9 F1).
