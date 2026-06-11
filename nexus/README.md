# `nexus/` — Nexus_v1: discovered-boundary semi-MDP world model

A second world model on top of EMERALD that pays its error budget **per discovered
skill** instead of per environment step. The contribution is **boundary discovery as
compression**: where skills begin and end is a latent variable inferred by an exact
semi-Markov DP under an MDL objective — not a heuristic, not reward backprop.

Self-contained except for importing `emerald_torch` (the step tier).

## Modules

| File | Role |
|------|------|
| `config.py` | §10 starting card; composes the (untouched) EMERALD step-tier config with skill-tier params. |
| `skill.py` | **VQ codebook** (EMA + dead-code restart) + **skill encoder** `q(k\|segment)` — the posterior over skills. |
| `jumpy.py` | **Option-model WM** `p(z_term, Σr, τ, continue \| Sₙ, kₙ)`: `Hₙ` causal transformer, jumpy terminal-latent **MaskGIT**, `Σr`/`τ`/continue heads, HL actor (prior) + HL critic (`γ^τ`). |
| `segment.py` | **THE core.** Boundary proposer → top-M; exact **semi-Markov forward–backward DP** + Viterbi under the **MDL** emission (jumpy-NLL + code-rate + switch-cost). |
| `model.py` | `NexusAgent`: step tier (EMERALD) + skill tier; jumpy WM loss; HL actor-critic. |
| `train.py` | The 5-stage loop (§8). Reuses EMERALD collect/eval; same `metrics.jsonl`/`eval_eps`. |

## Run (smoke)

```bash
python3 -m nexus.train --configs tiny --device mps --logdir logdir/nexus_tiny --steps 200
```

`tiny` is a shape-test preset. A real preset (full dims, GPU) is a follow-up once v1
training dynamics are tuned. Verified end-to-end: both tiers train, segmentation yields
variable-length segments, the VQ codebook activates, eval/checkpoint written.

## v1 scope — what's built vs deferred (be honest in the paper)

**Built & running:** the full skill tier — VQ skills, skill encoder, jumpy option-model
WM (terminal MaskGIT + `Σr`/`τ`/continue + `Hₙ` with h-input dropout 0.3), HL actor/critic
with `γ^τ`, the boundary proposer, and the **exact semi-Markov DP under MDL** (forward–
backward marginals + Viterbi over a top-M proposal set).

**Documented v1 simplifications / gaps** (the fragile joints §9 names — flagged, not hidden):
1. **Step actor not yet skill-conditioned** (Stage 4 "close the loop"). The step tier
   runs as EMERALD verbatim; passing `e(kₙ)` into the step AC is the one deferred change.
2. **DP emission conditions on `(z_a, k)` with `Hₙ`=0** (v1 decoupling); the full
   `Hₙ`-conditioned jumpy WM is trained on the *chosen* segments in Stage 3. This breaks
   the chicken-and-egg within one backward while keeping the MDL loop across stages.
3. **Segment representation = mean of globally-contextualized per-step features** (one
   bidirectional pass over the window), rather than a per-segment transformer pass —
   an efficiency choice for batched DP scoring.
4. **Training dynamics are not tuned.** The MDL fixed point (avoid all-length-1 / one
   giant segment), proposer quality, and codebook collapse are the live risks §9 calls
   out — the switch-cost + code-rate regularizers are in place but need empirical work.

**Metrics to watch** (logged): `vq_perplexity` (codebook collapse alarm), `mean_seg_len`
/ `mean_n_segs` (degenerate-fixed-point alarm), `hl_*` losses. Boundary↔achievement F1
(the H2 payoff figure) is eval-only and not yet wired.
