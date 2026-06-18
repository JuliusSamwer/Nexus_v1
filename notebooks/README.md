# Notebooks

Colab notebooks, grouped by topic. Each clones the repo fresh and runs from there, so
their location here doesn't affect execution.

## Craftax-JAX (fast MBRL proxy — `emerald_jax/`, `craftax_sweep/`)
- **`emerald_jax_benchmark_and_10M.ipynb`** — speed benchmark (`craftax()` vs
  `craftax_fast()`) + full 10M training with Drive checkpointing/auto-resume.
- **`emerald_jax_craftax_colab.ipynb`** — standard single-run Craftax-Classic training.

## EMERALD Crafter baseline (torch — `emerald_torch/`, `repro/emerald_baseline/`)
- **`nexus_emerald_colab.ipynb`** — EMERALD on Crafter, baseline repro.

## N1 segment-bottleneck (shelved — `nexus/`)
- **`nexus_n1_colab.ipynb`** — N1 segment-bottleneck experiment.
- **`nexus_kill_experiment_colab.ipynb`** — N1 go/no-go "kill" experiment
  (results in `results/kill_week1/`).
