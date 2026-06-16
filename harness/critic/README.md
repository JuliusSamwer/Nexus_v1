# Week-one conditional-critic go/no-go

A learned **conditional critic** `C(s_t, a_t, ŝ_{t+1})` that predicts world-model rollout
divergence, vs. the baselines it must beat or match. Substrate-agnostic: runs on the trained
**5M Crafter EMERALD** (torch, PRIMARY) and on **emerald_jax / Craftax-Classic** (for later).

## The question
Can one learned object predict *when the world model is wrong* better than a marginal critic
and intrinsic signals — and, unlike an ensemble, expose a usable **∇C**?

- **Detection = parity check.** If `C` merely *matches* K-sample disagreement at a fraction of
  the params, that's a **PASS** — because the ensemble can't do legs 2–3 cheaply.
- **The kill** is failing the floor: not beating the **marginal** (no context/action) and
  **horizon-only** baselines. If you can't beat those, the idea is broken.
- `C` is kept **differentiable wrt ŝ** (`critic.grad_wrt_shat`) — the leg-2 correction hook an
  ensemble disagreement scalar structurally lacks.

## Predictors compared
| | |
|---|---|
| `conditional` | full `C([s_{t-1}, a_{t-1}, ŝ_t, t])` |
| `marginal` | `C([ŝ_t])` — no context/action (ablation) |
| `horizon` | `C([t])` — trivial floor |
| `disagreement` | K-sample imagination disagreement (cheap single-WM ensemble proxy) |
| `entropy` | prior/token categorical entropy |
| `knn_density` | latent k-NN distance to a reference bank (OOD-ness) |

## Divergence labels (report ALL — don't rest on one)
- **L1** token-mismatch fraction (argmax disagreement over the 4×4×32 categorical grid)
- **L2** `KL(true-posterior ‖ imagined-prior)` summed over cells
- **L3** decoded-observation L2 (torch only; optional)
- **L4 / L4b** reward- / value-head divergence on imagined vs real state

> **Labels are WM-internal.** They compare the open-loop prior `ŝ_k` to the WM's *own*
> posterior `encode(real_obs_k)` — prior-vs-own-posterior drift, **not** absolute ground
> truth. State this in the paper; report C under multiple labels so a result doesn't hinge
> on one (non-metric) definition.

## Other design choices
- **Mixture action source** (`--mixture_p`): w.p. p the WM policy, else uniform random — so C
  generalizes to the rollout regimes you actually query at correction time (on-policy alone →
  tiny divergence; random alone → off-regime).
- **Seed-disjoint splits**: train/val/test never share a rollout.
- **Disagreement caveat**: the K-sample variant is the *cheap proxy*. A true K-model deep
  ensemble is the stronger bar and an explicit out-of-scope follow-up.

## Run it

**Primary — the 5M Crafter checkpoint** (two phases; generation uses a torch+crafter python,
analysis uses the JAX venv):
```bash
# 1. generate paired-rollout features (torch; ~CPU-bound, run overnight for big N)
/opt/anaconda3/bin/python3 -m harness.critic.run_gate --phase generate --substrate torch \
    --checkpoint /Users/juliius/Nobel30/emerald_runs/crafter_s0_5M/checkpoints_epoch_25_step_312124.ckpt \
    --out runs/critic/c5m.npz --n_starts 300 --H 15 --K 4 --mixture_p 0.5 --warmup 8

# 2. analyze (JAX venv): train C + ablations, score baselines, print the GO/NO-GO table
.venv_jax/bin/python -m harness.critic.run_gate --phase analyze --data runs/critic/c5m.npz
```

**Craftax (when emerald_jax is trained)** — point at a pickle checkpoint and use `--substrate jax`
(drop `--tiny` for the real config):
```bash
.venv_jax/bin/python -m harness.critic.run_gate --phase all --substrate jax \
    --checkpoint <emerald_jax ckpt> --out runs/critic/craftax.npz --n_starts 300 --H 15
```

## Status
Pipeline validated end-to-end on JAX (untrained → null AUROC, as expected) and the torch
adapter validated on the real 5M checkpoint (one paired rollout: L1 `[.023 .027 .021 .041]`,
L2 rising with horizon — the compounding-error signature). `grad_wrt_shat` returns a finite
∇C. Ready for the real generate run on the 5M checkpoint.

## Files
`wm_adapter.py` (interface + torch/jax adapters) · `paired_rollout.py` (dataset + k-NN bank) ·
`critic.py` (C-head + ablations + ∇C) · `baselines.py` (intrinsic predictors + isotonic cal) ·
`metrics.py` (AUROC/Spearman/ECE, numpy) · `run_gate.py` (CLI orchestrator).
