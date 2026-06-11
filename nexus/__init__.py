"""Nexus_v1 — a discovered-boundary semi-MDP world model built on EMERALD.

Two tiers from one buffer:
  * STEP tier  — EMERALD (reused verbatim from `emerald_torch`, world model untouched),
                 with an active-skill embedding added as an input to the actor/critic only.
  * SKILL tier — NEW: a VQ skill codebook + skill encoder (posterior over skills),
                 a jumpy option-model world model `p(z_term, Σr, τ, continue | Sₙ, kₙ)`
                 (Sutton–Precup–Singh, one level up), an HL actor (skill prior) and an
                 HL critic discounted by γ^τ.

The contribution is the boundary discovery: segmentation is a latent variable inferred
by an exact semi-Markov forward–backward DP (over a top-M proposal set) under an MDL
objective — jumpy-NLL + code-rate + switch-cost. Reward never touches the boundaries
through its gradient; it shapes them only by being one of the things each segment must
predict (`Σr`). See `segment.py`.

This package is self-contained except for importing `emerald_torch` (the step tier).
"""
