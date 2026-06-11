"""Standalone, MPS-runnable reimplementation of EMERALD (Burchi & Timofte, ICML 2025).

Clean-room reimplementation in plain PyTorch — does NOT import EMERALD's `nnet`
framework and is fully separate from dreamerv3-torch, so the two can be compared
modularly. Architecturally faithful (spatial 4x4x32x32 latent, transformer dynamics,
MaskGIT prior, symlog/twohot heads, imagination actor-critic, EMERALD hyperparameters)
with two documented simplifications for local runs (see model.py):
  1. world-model training uses full causal attention over the L window (no cross-batch
     KV cache / TBTT);
  2. imagination uses intra-rollout context (no 64-frame real-context KV cache).

Outputs dreamerv3-torch-compatible logs (metrics.jsonl + eval_eps/*.npz) so
harness/eval_overlay.py reads it directly.
"""
