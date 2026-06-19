# Nexus — Decision-Aware World-Model Finetuning: Progress Report
_As of 2026-06-19. Proxy: EMERALD-JAX on Craftax-Classic (4090). North star: V-JEPA-2 (not started)._

## 1. The research question (north star)
A lightweight, **decision-aware adaptation layer** that aligns a frozen foundation
representation for control — "the RLHF moment for world models." Two-phase recipe:
**pretrain a general reconstruction world model → finetune it toward decision-awareness,
localized by a learned divergence critic** (the critic = the RLHF reward-model analogue).
Everything here is the *fast proxy* to validate the mechanism cheaply before paying for V-JEPA-2.

## 2. What we built (code, all on origin/main)
- `emerald_jax/` — JAX/Flax EMERALD on Craftax-Classic. + fixed-buffer `imagine` (killed the
  O(H²) compile blow-up). `craftax_fast` throughput preset; `craftax_fast_tiny_wm` (small-WM arm).
- `craftax_sweep/` — training driver (milestone ckpts + metrics/eval JSONL + resume) + RunPod runner.
- `harness/critic/` — conditional-critic gate; fixed to read JAX ckpts (`blob["cfg"]`),
  policy-driven rollouts, fixed-buffer `imagine_actions`.
- `stage2_finetune/` — `DESIGN.md`, `decision_eval.py` (decision-relevant horizon curves),
  `finetune.py` (arms A0/A1/A2/P2, `roll_H`).

## 3. Models trained (on the persistent pod volume + partially local)
| model | params | env-steps | final Crafter score | notes |
|---|---|---|---|---|
| **full WM** (`craftax_fast`) | 32.6M | 10M | **23.9** (peaked ~26) | 10 milestones; mid-run exploitation turbulence |
| **tiny WM** (`craftax_fast_tiny_wm`) | 5.4M | 10M | **13.8** | 10 milestones; 6× smaller, much weaker |
| full5M-A0 | 32.6M | +4k ft | 26.1 | control (recon-only continued training) |
| **full5M-A2** | 32.6M | +4k ft | 22.5 | **decision-aware finetune — the key artifact (horizon 7)** |
| P2 runs (improved/baseline/ext) | — | — | 18.4 / 21.9 / 10 | frozen-actor retrain — all negative (exploitation) |

Trained at batch 16 (4090 OOM-forced from 32), `gpc=1` (low replay ratio → undertrained for
speed; this is the main reason score≈24 vs EMERALD's ~58 on real Crafter), single seed, ~360 env/s.

## 4. Results by experiment

### (a) Capacity sweep + the exploitation collapse
- Full: climbs to ~26 by 3–5M, **turbulent collapse 5.4–8.6M (down to ~5–8)**, recovers to ~24 by 10M.
- **Diagnosed (6M checkpoint):** value-head pred 3.1→7.0 (inflated) while real reward 10× lower;
  policy collapsed to spamming one action (90%), entropy 1.66→0.45. **Model exploitation /
  critic divergence**, transient. Losses looked healthy throughout → invisible to training loss.
- Tiny far weaker (13.8 vs 23.9): **weaker WM → weaker control**, as expected.

### (b) Divergence / learned-trust gate (DIRECTIONAL — n=24, single seed, noisy)
- **Supports the capacity hypothesis on reward-divergence:** the weak (tiny) WM's
  reward-divergence is far more detectable by the learned critic (AUROC **0.94**, beats all
  baselines) than the strong WM's (**0.72**, ≈ marginal). Weaker WM → stronger *learnable*
  decision-relevant divergence.
- value-divergence goes the other way (favors healthy strong full-3M). Not yet confirmed at scale.

### (c) Decision-relevant horizon baselines (the "before")
| | Crafter score | usable decision horizon | token-acc |
|---|---|---|---|
| full-5M | 27.0 | **1 step** | ~66% (flat) |
| tiny-10M | 13.8 | **0 steps** | ~33% |
- The **decision cliff**: imagination is decision-usable for ~1 step, while **token accuracy looks
  fine the whole way** → token-acc is misleading; the decision-relevant curve is what matters.

### (d) ⭐ Stage-2 decision-aware finetune — THE headline result
Full-5M, after finetuning (decision_w=1.0):
| horizon k | baseline | A0 (more training) | **A2 (decision-aware)** |
|---|---|---|---|
| pol_agree @3 | 13% | 15% | **58%** |
| pol_agree @7 | 5% | 3% | **58%** |
| pol_agree @15 | 24% | 13% | **46%** |
| val_div @15 | 0.65 | 0.73 | **0.28** |
| **usable horizon** | 1 | 1 | **7** |
| Crafter score | 27.0 | 26.1 | 22.5 |
- **A2 flattens the decision cliff (1→7 steps), halves val_div — A0 (just more training) does
  nothing.** The gain is from the decision-aware objective, not extra steps. Cost: ~17% score.

### (e) Score-increase attempts (NEGATIVE but informative)
- **Freeze WM + retrain actor (P2): degrades score on BOTH WMs** (improved 22.5→18.4,
  baseline 27→19.9). Extended run got monotonically worse (→10). Losses stable → **model
  exploitation**, WM-agnostic. **Conclusion: freeze-and-retrain is a dead end.**
- The score goal needs **co-training** (the grounded recipe): unfreeze WM + recon anchor +
  gentler actor LR. A2 *is* the co-trained version (lost only 4.5 pts vs frozen's 9+).

## 5. Headline findings (what's solid)
1. **Decision-aware finetuning makes a pretrained WM's imagination decision-reliable ~7× deeper
   into the horizon (1→7), where more training does nothing.** ← the publishable core.
2. **Token accuracy is misleading**; the decision-relevant horizon is the right metric.
3. **Model exploitation is the recurring villain** — appears in from-scratch training (the collapse)
   and in naive actor retraining (P2). The recon anchor / grounding is what controls it.
4. **(Directional) weaker WM → stronger learnable decision-relevant divergence** (capacity hypothesis).

## 6. Open / unconfirmed
- **Score increase** — the open frontier. Path: tune the co-trained A2 (`decision_w` sweep:
  0.1/0.3/1.0) to keep horizon while recovering score. (Sweep was launched, then stopped.)
- **A1 control** (multi-step but decision-agnostic) — needed to attribute the 1→7 gain to
  *decision-awareness* vs generic exposure-bias. Not yet run.
- **Full-scale divergence gate** (n=200, multi-seed, vectorized rollout) — to confirm (b).
- **Capacity tie-in on finetune** — does the weak (tiny) WM gain *more* from A2? (tiny finetune
  arm was started then killed.)
- **V-JEPA-2 north star** — not started.

## 7. Artifacts & locations
- **Local** (`runs/pod_pull/`): all logs + `ckpts/{full,tiny}/{metrics,eval}.jsonl` + partial
  full milestones (7M–10M). `/tmp/diag/`: full 3M/6M/10M + tiny-10M ckpts + A2 horizon evals.
- **Pod volume (stopped, persistent):** all 10 full + 10 tiny milestones, full5M-A2 (improved WM),
  A0, P2 runs. **Re-accessible by restarting the pod — do not delete the volume.**

## 8. Decision points / realistic paths forward
- **A. Lock the horizon result** — run the A1 control + write it up. Cheapest path to a
  defensible contribution. (The 1→7 result is the strongest thing we have.)
- **B. Chase the score** — `decision_w` sweep of co-trained A2 (grounded). Higher-risk; the
  score may be a genuinely hard problem (exploitation-gated).
- **C. Confirm the capacity/divergence story** — vectorize the gate rollout, run n=200 multi-seed.
- **D. Scale up the base model** — higher replay ratio / bigger GPU to get a stronger base WM
  (score toward EMERALD's level) before finetuning, so the finetune story rides on a credible base.
- **E. Start the V-JEPA-2 north star** — the real artifact, biggest effort.
