# Phase 0 instrumentation harness

Three instruments that turn EMERALD into a *measured* baseline. None of them change
EMERALD's training — they observe it. They are written against the real EMERALD code
(`third_party/EMERALD`); exact hook points are cited below and in each module.

> Status: **skeletons**. The hook points are confirmed by reading the source, but the
> instruments must be run against a loaded checkpoint on a CUDA GPU (the local M3 Max
> has no CUDA). Each module raises `NotImplementedError` where a live model/checkpoint
> is required, with the precise call documented.

## 1. Achievement-vs-causal-depth logger — `achievement_depth.py`
Reproduce the cliff. Reads `results/EMERALD.json` (or a fresh eval dump), maps each
achievement to a causal-chain depth (Crafter tech tree), and emits the
success-vs-depth curve that is the project's *baseline* figure.
- Input: an eval results JSON in EMERALD's format (`achievements_<name>` keys).
- Output: `results/achievement_depth.json` + a plot.

## 2. Open-loop imagination-fidelity curve — `open_loop_fidelity.py`
**The paper's opening figure.** From real observed states, roll the world model forward
`k` steps under the *logged* actions and measure how fast imagination diverges from
ground truth as a function of `k`.
- Hook: `TSSM.imagine(p_net, prev_state, img_steps=k, actions=<logged>, return_stoch_steps=True)`
  — `third_party/EMERALD/nnet/modules/emerald/tssm.py:219`.
- Seed states: `TSSM.observe(...)` posteriors — `tssm.py:193`.
- Metrics: latent error (per-token argmax disagreement and/or categorical KL vs the
  posterior at step k) and decoded-frame error (MSE; optionally LPIPS) via the decoder.
- Output: `results/open_loop_fidelity.json` (error vs k) + curve.

## 3. MaskGIT confidence logger — `maskgit_confidence.py`
Dump the per-token confidence EMERALD already computes during sampling, so it can later
become a rollout-level reliability signal (Phase 2, Candidate 1).
- Hook: `MaskNetwork.sample(...)`, `selected_probs = sum(softmax(logits)*stoch, -1)`
  — `third_party/EMERALD/nnet/modules/emerald/mask_network.py:170`. This is the clean
  per-token confidence *before* Gumbel noise (`conf_scores`, line 173, is the noised
  ordering used internally). Capture `selected_probs`, not `conf_scores`.
- Output: `results/maskgit_confidence.npz` (per-token confidence over a rollout).

## How these compose into the diagnosis (Phase 1)
Overlay (2) with the per-achievement env-step depth derived alongside (1): if imagination
becomes unreliable *below* the depth a failing achievement needs, that is the mechanism.
(3) gives the signal a Phase-2 fix can act on.

## Running (later, on the GPU box)
EMERALD loads via its config/checkpoint machinery (`run_name=... python3 main.py
--load_last --mode evaluation`). The harness is intended to import EMERALD's model and
call the hooks above on a loaded checkpoint; see each module's `TODO(checkpoint)`.
