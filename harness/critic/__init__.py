"""Week-one conditional-critic go/no-go experiment.

Tests whether a learned conditional critic C(s_t, a_t, ŝ_{t+1}) predicts world-model
rollout divergence better than a marginal critic C(ŝ_{t+1}), a horizon-only baseline,
and intrinsic signals (K-sample disagreement, prior/token entropy, latent k-NN density).

Substrate-agnostic via WMAdapter (harness.critic.wm_adapter):
  * EmeraldTorchAdapter — the trained 5M Crafter EMERALD (PRIMARY; the only trained WM).
  * EmeraldJaxAdapter   — emerald_jax on Craftax-Classic (for when that model is trained).

Two-phase pipeline so the critic never depends on the WM's framework:
  1. GENERATE (substrate venv): paired_rollout dumps per-step features + 4 divergence
     labels + intrinsic signals to an .npz, tagged by seed.
  2. ANALYZE (JAX venv): critic + baselines + metrics consume the .npz, print the
     GO/NO-GO table (AUROC / Spearman / calibration) on held-out seeds.

See README.md for design decisions and the exact command to run on the 5M checkpoint.
"""
