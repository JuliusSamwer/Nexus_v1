# Stage-2: decision-aware world-model finetuning

Thesis core. Phase 1 (done) = pretrain a general reconstruction WM (the craftax_sweep
arms). Phase 2 (this) = **finetune the pretrained WM toward decision-awareness, localized
by a learned divergence critic**, and show imagined rollouts become decision-reliable
deeper into the horizon — on a budget, without collapse.

## Contribution (what must be isolated)
> Two-phase WM learning: pretrain general reconstruction WM → finetune toward
> decision-awareness, **localized by a learned divergence critic** (the RLHF-reward-model
> role). Novel vs VAML/value-equivalence: (1) the two-phase pretrain→finetune recipe,
> (2) the critic as the localization signal. The critic-as-localizer is THE contribution.

## Success metric — decision-relevant horizon (NOT token accuracy)
Imagined-rollout vs real, t=1..15 (reuse harness/critic/horizon_curves.py + the gate):
- **policy agreement** KL[π(·|ŝ_t) ‖ π(·|s_t)] — headline (pretrained crosses 50% at ~5-8 steps)
- **value error** |V(ŝ_t) − V(s_t)|
- **reward-event recall** (token-acc hides this)

PRIMARY NUMBER: **usable decision horizon** = where policy-agreement crosses 50%. Goal
push ~6 → ~10+. GUARDRAIL: **real Crafter score must not drop** (else WM sharpened into a
fantasy = model-exploitation, the recurring villain we watched at full-6M).

## Checkpoints (W0)
- **full-5M** (clean healthy peak, score 26, pre-exploitation) — primary.
- **tiny-10M** (weak WM, score 13.8) — on-hypothesis: weak WM has MORE *removable*
  decision-relevant error, so should gain MORE from the finetune. Ties Stage-2 to the
  divergence/capacity result.

## Axis A — objective (the science). Finetune on the model's OWN multi-step rollouts,
actor co-training, small recon anchor.
| arm | objective | role |
|-----|-----------|------|
| A0 | recon-only continued training | control for "just more gradient steps" |
| A1 | multi-step rollout, decision-AGNOSTIC (scheduled sampling / latent overshoot) | control for "any horizon fix" (exposure bias) |
| A2 | multi-step + value-gradient weighting (∂V/∂ŝ + ∂π/∂ŝ) | robust decision-aware baseline (critic-free) |
| A3 | multi-step + **critic-localized** weighting | THE contribution |
Contribution lives in **A3 > A2 > A1 > A0**. A3≈A1 → only an exposure-bias fix (not
decision-aware). A3≈A2 → critic adds nothing over value-grads (weaker, still publishable).
A0 + A1 are NON-NEGOTIABLE controls; match total gradient steps across arms.

## Axis B — parameter scope (the freeze ablation). Run SECOND.
adapter-only → LoRA → unfreeze-top-N → full. MVP uses **full-finetune** (max signal);
spectrum answers "does frozen+adapter recover it" (the north-star pitch). Nearly free in
the emerald_jax proxy (no foundation to freeze — the freeze is artificially imposed here).
Objective ≠ scope: same reweighted+anchored loss, applied to adapter OR whole WM.

## Risks + controls
1. **Model exploitation** — recon anchor + online critic co-train + the score guardrail.
   Horizon-curves "improve" while score drops = exploitation; flag + stop.
2. **More-training confound** — A0 controls it; match gradient steps.
3. **Critic staleness** — for A3 the critic must be co-trained ONLINE (WM/actor shift).
   And the critic must first be validated as a reliable localizer (vectorized n_starts=200
   multi-seed gate run) — prerequisite for A3, not A0/A1/A2.

## MVP (run first)
full-5M + tiny-10M, full-finetune scope, arms A0/A1/A2 (A3 folded in once critic
validated), matched steps → 3 horizon curves + Crafter score. One plot (usable-horizon by
arm, with score guardrail) decides the contribution at minimum compute. Then Axis B.

## Why it closes the crossover thesis
Decision-aware finetune makes the latent more decision-sufficient → in a toy world shrinks
the WM-world gap (conditioning even less needed — matches the Crafter result). In the real
world the gap is irreducible → the residual decision-relevant divergence the finetune
CAN'T remove is exactly where the conditional trust critic stays load-bearing. Arc:
**detect** decision-relevant divergence (done) → **finetune** to remove what's removable
(this) → **irreducible residual** = the critic's permanent job (robotics).

## Build plan (parallel tracks)
- TRACK 1 (now): finetune harness — load W0, decision-aware finetune driver with arm flag
  (A0/A1/A2), actor co-train, recon anchor, multi-step own-rollout objective; reuse
  emerald_jax model + horizon-curve/gate eval; before/after on full-5M + tiny-10M.
- TRACK 2 (parallel): vectorize harness/critic generate_rollout (vmap rollouts + scan
  steps) → run n_starts=200 multi-seed gate to validate the critic as localizer → enables A3.

DECIDED 2026-06-18: MVP = full-5M + tiny-10M; localizers A2 (value-grad) + A3 (critic);
build harness in parallel with critic validation.
