"""Nexus config — Segment-Native World Model (Strict Bottleneck), build outline v1.

Composes the EMERALD parts-library config (`emerald_torch.config`, reused for the frame
encoder/decoder/TSSM/MaskGIT/heads sizes) with the segment-native (slow-tier) parameters.

The headline N1 dials are `w` (leak width, §2.3) and `G` (slow tokens per segment, §2.4);
the N1 grid is {w in 0,16,64} x {G in 2,4,8}. Defaults: w=0 (fully strict), G=4.
"""

from types import SimpleNamespace

from emerald_torch import config as ecfg


def base():
    step = ecfg.base()                       # EMERALD parts-library sizes, reused verbatim
    return SimpleNamespace(
        step=step,
        # --- trajectory / segmentation (§1, §2.2 seg=scheduled) ---
        T=256, ell_bar=32, seg_jitter=8, L_max=128,
        # --- slow tokens u_n (§2.4) ---
        G=4, u_classes=256, u_emb_dim=256,
        slow_uniform_mix=0.01, slow_free_bits=1.0,
        # --- fast-tier bottleneck (§2.3) ---
        w=0,                                  # leak width: 0 (strict) | 16 | 64
        # --- slow posterior (segment encoder, §2.4) ---
        post_blocks=2, post_dim=256, post_heads=8,
        # --- slow prior / jumpy model (§2.5) ---
        slow_blocks=4, slow_dim=512, slow_heads=8, slow_ctx=16,
        ground_mask_blocks=2, ground_decoding_steps=3,
        tau_max=128,                          # τ two-hot over log-bins 1..tau_max
        # --- loss weights (§4) ---
        lambda_slow=1.0, ground_scale=2.0,    # grounding weighted highest
        slow_tau_scale=1.0, slow_r_scale=1.0, slow_cont_scale=1.0,
        # --- optim (slow tier; fast tier reuses EMERALD's) ---
        slow_lr=1e-4, slow_eps=1e-8, slow_grad_clip=1000.0, weight_decay=0.0,
        # --- loop ---
        batch_size=8, prefill=2000, train_ratio=0.5,
        eval_every=5000, eval_episodes=3, log_every=250, save_every=5000,
    )


def crafter():
    """Real N1 preset (single GPU). Full EMERALD sizes, T=256, ~8 segments/window."""
    c = base()
    c.batch_size = 8
    return c


def tiny():
    """Shape-test preset (CPU/MPS shakeout)."""
    c = base()
    c.step = ecfg.tiny()
    c.T = 48
    c.ell_bar = 8
    c.seg_jitter = 2
    c.L_max = 16
    c.G = 4
    c.u_classes = 32
    c.u_emb_dim = 64
    c.post_dim = 64
    c.post_blocks = 1
    c.slow_dim = 64
    c.slow_blocks = 2
    c.slow_ctx = 8
    c.ground_mask_blocks = 1
    c.ground_decoding_steps = 2
    c.tau_max = 16
    c.batch_size = 2
    c.prefill = 60
    c.eval_every = 40
    c.eval_episodes = 1
    c.log_every = 5
    return c


PRESETS = {"base": base, "crafter": crafter, "tiny": tiny}
