# Nexus — Long-Horizon Latent World Models

Research toward a **fundamental, embodiment-agnostic improvement to the latent
world-model reasoning core** — specifically *long-horizon latent-rollout fidelity*
(controlling compounding error) — validated on Crafter/Craftax and, as a stretch,
toward a JEPA + MPC manipulation loop. Full plan: [`emerald_research_plan.md`](emerald_research_plan.md).

The reference point is **EMERALD** (Burchi & Timofte, ICML 2025): strong overall on
Crafter (achievement score ≈ 57) but it falls off a *cliff* on the deepest tech-tree
tasks (`collect_diamond` ≈ 0.9%, `eat_plant` ≈ 0.06%). Closing that cliff is the thesis.

---

## What's in this repo

| Path | What |
|------|------|
| `emerald_torch/` | **Standalone, MPS/CUDA-runnable reimplementation of EMERALD** in plain PyTorch — a clean model arm to compare against, and the substrate to grow new architectures from. ~1200 lines, no external framework. |
| `harness/` | Phase-0 instrumentation + evaluation. `eval_overlay.py` is the multi-run **comparison dashboard** (Crafter score, per-achievement, the cliff, score-vs-step trajectory) with published baselines as reference lines. |
| `tools/` | Utilities: `plot_training.py`, `watch_crafter.py` (record an episode to mp4), `build_master_doc.py` (lab-notebook generator). |
| `emerald_research_plan.md` | The 5-phase research plan. |

External code (DreamerV3-torch dev baseline, EMERALD reference, Craftax, …) is **not
vendored** — see [External dependencies](#external-dependencies).

---

## Status

- ✅ **Dreamer baseline** trains locally on Crafter (DreamerV3-torch, MPS).
- ✅ **Standalone EMERALD** (`emerald_torch`) trains on MPS and CUDA — full pipeline
  verified (prefill → collect → train → eval → checkpoint), losses finite, achievements
  unlocking.
- ✅ **`eval_overlay`** compares any number of runs against each other and against
  EMERALD's published numbers.
- ⏭ **Next:** a faithful-dimensions 20k–50k-step EMERALD run on an A100, overlaid
  against the Dreamer baseline, to read the early-stage delta — then iterate on a new
  long-horizon mechanism (MaskGIT-confidence-aware rollout is the lead candidate).

---

## Run it

### Local (Apple Silicon / MPS)

```bash
# Dreamer baseline (dev baseline; see third_party/dreamerv3-torch)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m emerald_torch.train \
    --configs crafter_fast --device mps --logdir logdir/emerald_smoke --steps 20000
```

`crafter_fast` keeps EMERALD's architecture but uses a smaller batch/window so 20k
steps finish in ~4–5h on an M3 Max. `crafter_smoke` is EMERALD's true B=16/L=64 dims
(~15h locally — prefer a GPU). `tiny` is a shape-test.

### Colab / A100 (recommended for the real run)

A transformer world model is ~10–20× faster on an A100 than on MPS — the full-fidelity
20k run drops from ~15h to roughly ~1h. No code changes; just `--device cuda`.

```python
!pip install -q crafter torch torchvision numpy
# (upload/copy this folder, e.g. from Drive, then:)
%cd /content/nexus
!python3 -m emerald_torch.train --configs crafter_smoke --device cuda \
    --logdir /content/drive/MyDrive/nexus/emerald_smoke --steps 20000
```

### Compare

```bash
python3 -m harness.eval_overlay \
    --run dreamer=logdir/crafter_smoke \
    --run emerald=logdir/emerald_smoke \
    --ref EMERALD=results/EMERALD.json \
    --out results/eval_overlay
```

Produces `results/eval_overlay.png` (4 panels) + a summary JSON, and prints a
per-achievement table with the head-to-head score delta.

---

## `emerald_torch` design notes

Architecturally faithful to EMERALD: spatial **4×4 × 32×32** categorical latent,
**transformer dynamics (TSSM)**, **MaskGIT** iterative prior, symlog/twohot reward &
value heads, imagination actor-critic (TD-λ, percentile return normalization, slow
critic target) — all with EMERALD's published hyperparameters (`config.py`).

Two **documented simplifications** make it runnable locally without EMERALD's CUDA
framework (see `model.py` docstring):
1. World-model training uses **full causal attention over the L window** (no
   cross-batch KV cache / TBTT). Replay samples within a single episode.
2. Imagination uses **intra-rollout context** (each start attends its own imagined
   prefix), not EMERALD's `att_context_left` real-context KV cache.

Both preserve the distinguishing mechanisms; they trade some training efficiency, not
the science.

---

## External dependencies

Not vendored (external code, some non-commercially licensed). Clone alongside as
`third_party/` or copy from your working tree:

| Dir | Upstream | Role |
|-----|----------|------|
| `third_party/dreamerv3-torch` | NM512/dreamerv3-torch | **dev baseline** (PyTorch, MPS). Local shims applied: gym `Discrete` wrap, inf/uint8 bound fix, `crafter_smoke` config block. |
| `third_party/EMERALD` | burchim/EMERALD | reference baseline (CUDA). `results/EMERALD.json` (20-seed scores) is vendored here for the overlay. CC BY-NC-SA. |
| `third_party/Craftax` | MichaelTMatthews/Craftax | full Craftax benchmark (Phase 3). |

> EMERALD is licensed CC BY-NC-SA 4.0 (academic, non-commercial). `emerald_torch/` is a
> clean-room reimplementation from the paper/spec for research comparison.
