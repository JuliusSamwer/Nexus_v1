"""EMERALD in JAX/Flax, ported from emerald_torch for fast model-based RL on Craftax.

Target: Craftax-Classic-Pixels (63x63x3 -> padded 64x64), 17 actions, 22 achievements
— near-identical to Crafter, but the env runs on-GPU and vmapped, so the whole
train loop (world-model + MaskGIT + imagination actor-critic) can be jit-compiled
and run without a Python env loop. Architecture is faithful to emerald_torch.
"""
