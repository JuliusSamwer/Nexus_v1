# Long-Horizon Latent World Models — Summer Research Plan

## North star
Contribute a **fundamental, embodiment-agnostic improvement to the latent world-model
reasoning core** — specifically *long-horizon latent-rollout fidelity* (controlling
compounding error) — and show it transfers from a benchmark toward robotics
planning/control.

**This summer (Shape 1):** reproduce the current SOTA latent world model (EMERALD) on
Crafter, diagnose *precisely why* it fails on long-horizon tasks, fix it with a new
architectural mechanism, validate on the unsolved **Craftax (full)**, and — stretch —
show the same mechanism helps a **JEPA + MPC** manipulation loop in simulation.

---

## The thesis, made concrete by EMERALD's own numbers

EMERALD (Burchi & Timofte, ICML 2025) is a Dreamer-lineage latent world model whose win
is *within-step* coherence: a spatial `H×W` grid of categorical latent tokens predicted
with **MaskGIT iterative refinement** (`num_decoding_steps = 3`), so each imagined latent
frame is internally consistent. It reproduces to ~57% Crafter achievement score.

But its **imagination horizon is `H = 15`** (from `configs`/`nnet/models/emerald.py`), and
its per-achievement success rates (20 seeds, `results/EMERALD.json`) show a clean cliff
as a function of causal-chain depth:

| Tier (short → long causal chain) | Achievement | Success |
|---|---|---|
| Short | collect_wood | 0.999 |
| Short | place_table | 0.996 |
| Short | make_wood_pickaxe | 0.992 |
| Mid | make_stone_pickaxe | 0.937 |
| Mid | collect_iron | 0.664 |
| **Long** | make_iron_pickaxe | **0.316** |
| **Long** | make_iron_sword | **0.324** |
| **Very long** | collect_diamond | **0.009** |
| **Very long (delayed reward)** | eat_plant | **0.0006** |

**Hypothesis:** EMERALD solved single-step latent fidelity but reasons over only ~15
imagined steps. Long-causal-chain achievements require coherent prediction + credit
assignment over horizons far beyond that, and naively increasing `H` fails because
compounding error degrades the rollout before the distant reward is reachable. The open
problem — and the robotics-relevant one — is **bounded-error long-horizon latent rollout.**

---

## Phase 0 — Reproduce + instrument (weeks 1–2)

Goal: trust the baseline AND have the failure-measurement harness live before you change anything.

- **Setup:** `git clone https://github.com/burchim/EMERALD && cd EMERALD && ./install.sh`
  (PyTorch, single-GPU; CC BY-NC-SA, research use only.)
- **Train:** `run_name=crafter python3 main.py` — logs/replay/checkpoints under `callbacks/run_name`; `tensorboard --logdir ./callbacks`.
- **Eval:** `run_name=crafter python3 main.py --load_last --mode evaluation`.
- **Reproduction bar:** match within noise over **3 seeds** (not 20). Success = achievement
  score in the mid-50s% AND the per-achievement breakdown matches the table above —
  especially that diamond ≈ 0.01 and eat_plant ≈ 0.0006. Wildly different per-achievement
  numbers = a setup bug, not noise.
- **Instrument now (this is the real work of Phase 0):**
  1. **Per-achievement-vs-causal-depth logger** — reproduce the cliff as your baseline figure.
  2. **Open-loop imagination-fidelity curve** — from real states, roll the world model
     forward `k` steps under the logged actions; measure latent prediction error (and
     decoded-frame error) vs `k`. This is the curve your architecture must bend, and it's
     your paper's opening figure.
  3. **MaskGIT confidence logger** — dump the per-token confidence scores `sample()`
     already computes; you'll likely reuse these as an uncertainty signal later.

---

## Phase 1 — Diagnose: *why* does it fail? (weeks 3–4)

Do not design the fix yet. Establish the mechanism with experiments:

- **Horizon sweep:** retrain (or fine-tune) with `H ∈ {15, 30, 60, 120}`. Expected: deep
  achievements improve slightly then plateau/regress as compounding error dominates.
  This is the experiment that proves "more horizon isn't enough" and motivates an
  architectural change rather than a hyperparameter.
- **Fidelity vs reward reach:** overlay the open-loop fidelity curve with the env-step
  depth each achievement requires. Show the horizon at which imagination becomes unreliable
  sits *below* the depth the failing achievements need.
- **Localize the error source:** is divergence driven by (a) accumulating token errors
  across steps, (b) the deterministic state drifting, or (c) compounding stochastic-branch
  errors? Ablate by teacher-forcing subsets of the latent during rollout. The answer
  dictates the fix.

Deliverable of this phase: a one-sentence, evidence-backed failure mechanism.

---

## Phase 2 — New architecture: fix the mechanism (weeks 5–8)

Pick the lever the diagnosis points to. Candidates (pre-diagnosis, ranked by risk/payoff):

1. **Confidence-/uncertainty-aware rollout** — aggregate MaskGIT token confidence into a
   rollout-level reliability signal; truncate, branch, or down-weight imagined trajectories
   before they diverge. Cheap, novel, uses a signal already computed, and directly relevant
   to robotics ("know when the world model is hallucinating"). Strongest first-paper bet.
2. **Cross-temporal refinement / consistency** — extend masked refinement from within-frame
   to across-time; add a cycle/consistency objective that explicitly bounds accumulated error
   over the rollout.
3. **Structured / object-centric spatial latent** — make the `H×W` token grid compositional
   so it chains more stably across long horizons (your "how to represent latents" interest).

Discipline: change **one** mechanism, ablate it cleanly, keep all else equal to the
reproduced baseline. A partial win on the deep achievements (e.g. iron tier 0.32 → 0.6,
diamond 0.01 → meaningfully nonzero) is already a publishable result on Crafter.

---

## Phase 3 — Validate on the open frontier (weeks 8–10)

- **Target:** Craftax (full) — JAX, single-GPU, currently unsolved (~18% of max reward,
  only the first 4 of 9 floors ever reached). This is where a long-horizon fix has real
  headroom and "beat what" is legible.
- **Engineering decision:** EMERALD is PyTorch + slow Python Crafter; Craftax is JAX and
  far faster. Start by running the agent against Craftax through a Python interface (simpler;
  eat the throughput cost). Port the mechanism to a JAX MBRL stack only if env speed becomes
  the bottleneck for seeds/ablations.
- **Baselines to position against (cite explicitly):** EMERALD, Δ-IRIS, DIAMOND, DreamerV3
  on the classic score; the TransformerXL ~18% and the floors-reached ceiling on Craftax-full.
  Report at matched env-step budgets.

---

## Phase 4 — Stretch: JEPA + MPC manipulation transfer (if time)

Frame: the contribution is the *dynamics-model* mechanism, which is portable. Show it also
improves a **JEPA + MPC** control loop (the real-robotics planning loop: encode current &
goal → predict latent under candidate actions → minimize goal-conditioned energy via CEM →
execute → replan).

- **Build on the released, frozen V-JEPA 2 encoder** (don't pretrain a video model). Train a
  small action-conditioned latent predictor + run CEM-MPC. Feasible on the budget because
  only the small predictor + planning cost money.
- **Start trivially simple:** one pushing or pick task (PushT-style) to get
  encode→predict→CEM→execute working end-to-end before scaling task complexity.
- **The robotics-relevant claim:** V-JEPA 2-AC plans well at *short* horizons (locally convex
  energy landscape); long-horizon JEPA planning is open precisely because of latent drift —
  i.e. your mechanism is aimed at the actual frontier of JEPA-for-control.

---

## Compute & discipline (budget: strong laptop + ~$2k rental)

- **Estimate empirically:** time **one epoch** (of 50) on your rented GPU, multiply by 50 ×
  seeds. Ballpark sanity check: MBRL Crafter runs are typically ~1–2 GPU-days/seed on a
  single modern GPU.
- **3 seeds, not 20**, for reproduction and each ablation. Kill diverging runs early.
- **Use spot/interruptible instances** (RunPod / Vast / Lambda). A 3-seed reproduction is
  ~$100–250; the full program fits comfortably in $2k if you don't run 20-seed sweeps.
- Config anchors: `batch_size 16`, `L 64`, `H 15`, `num_envs 16`, `epochs 50`,
  `epoch_length 12500`, `dim_model 512`, `num_decoding_steps 3`, `img_stride 4`.

## Risks & gotchas

- MBRL reproductions are seed-sensitive; the achievement score has real variance — judge by
  the per-achievement breakdown, not one number.
- Crafter's Python env is slow; with `num_envs 16` the env stepping can bottleneck a GPU.
- Resist designing the fix before Phase 1 finishes. The diagnosis is the contribution's spine.
- Matching a tuned baseline is itself hard — budget Phase 0 fully; a "win" over an
  under-tuned baseline isn't a win.

## What "on the map" looks like for this
A clean arXiv preprint + reproducible open code + one sharp opening figure (the fidelity /
achievement-depth cliff) + a mechanism that bends it, positioned against current SOTA, with a
credible robotics-transfer argument (ideally one corroborating sim experiment). Target a top-venue
workshop as the realistic, controllable outcome; broader citation is downstream of rigor + clarity.
