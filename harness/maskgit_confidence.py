"""Instrument 3 — MaskGIT confidence logger.

EMERALD already computes a per-token confidence during MaskGIT sampling. We dump it so it
can later become a rollout-level reliability signal (Phase 2, Candidate 1: confidence-aware
rollout — "know when the world model is hallucinating").

Hook point (third_party/EMERALD/nnet/modules/emerald/mask_network.py):
  - sample() at :135
  - selected_probs = sum(softmax(logits) * stoch, -1)   at :170
      -> the CLEAN per-token confidence (max-prob of the sampled token), BEFORE noise.
  - conf_scores = log(selected_probs) - log(-log(rand)) at :173
      -> Gumbel-noised version used only for the masking ORDER. Do NOT log this as
         confidence; log selected_probs.

Two ways to capture it without forking EMERALD's logic:
  (A) Monkeypatch / forward-hook MaskNetwork.sample to stash selected_probs each call.
  (B) Add `return_stoch_steps=True` plumbing (already supported) and recompute
      selected_probs from the returned per-step logits/stoch.

Aggregations worth recording per imagined step:
  - mean / min token confidence over the HxW*stoch grid (min = weakest link),
  - entropy of the token distribution,
  - how confidence decays as the open-loop rollout deepens (overlay with Instrument 2).

Output: results/maskgit_confidence.npz  (per-token confidence over a rollout).
"""

import argparse


def attach_confidence_hook(mask_network, sink):
    """Register a hook that appends per-call `selected_probs` to `sink` (a list).

    TODO(checkpoint): on the GPU box, wrap MaskNetwork.sample so that each call records
    selected_probs = (softmax(logits) * stoch).sum(-1) computed at mask_network.py:170.
    A forward_pre/forward hook can't see the local `selected_probs`, so the simplest
    robust approach is a thin monkeypatch:

        orig = mask_network.sample
        def patched(deter, num_steps=3, return_stoch_steps=False):
            out = orig(deter, num_steps, return_stoch_steps)
            # recompute from the predictor head on `deter` if needed, or capture inside
            # a lightly-edited sample() that yields selected_probs.
            return out
        mask_network.sample = patched

    Prefer a small, clearly-marked edit to a copy of sample() that *returns* selected_probs
    over a fragile patch — this is research code we control via the submodule diff.
    """
    raise NotImplementedError(
        "Needs the loaded EMERALD MaskNetwork on CUDA. Capture selected_probs from "
        "mask_network.py:170 (NOT the Gumbel-noised conf_scores at :173)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=False)
    ap.add_argument("--out", default="results/maskgit_confidence.npz")
    args = ap.parse_args()
    print(__doc__)
    if not args.checkpoint:
        print("\nNo --checkpoint given. Runs on the GPU box; capture selected_probs "
              "(mask_network.py:170) during sampling.")


if __name__ == "__main__":
    main()
