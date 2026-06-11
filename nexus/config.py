"""Nexus_v1 config — §10 starting card. Composes the (untouched) EMERALD step-tier
config with the new skill-tier parameters."""

from types import SimpleNamespace

from emerald_torch import config as ecfg


def base():
    step = ecfg.base()                      # EMERALD step tier — inherited untouched
    return SimpleNamespace(
        step=step,
        # --- skill codebook (VQ) ---
        K=64, code_dim=128, vq_ema=0.99, vq_commit=0.25, vq_restart_thresh=1.0,
        # --- skill encoder q(k|segment) (posterior) ---
        skill_enc_blocks=2, skill_enc_dim=256, skill_enc_heads=8,
        # --- HL recurrent state Hₙ ---
        hl_dim=512, hl_blocks=2, hl_heads=8, hl_ctx=16, h_dropout=0.3,
        # --- jumpy terminal-latent MaskGIT ---
        jumpy_mask_blocks=2, jumpy_decoding_steps=3,
        # --- regression heads (Σr, τ, HL value) ---
        bins=255, tau_low=0.0, tau_high=6.0,    # τ head is two-hot on log τ
        # --- segmentation / MDL ---
        L_max=128, T=256, ell_bar=50, top_M=32,
        switch_cost_scale=1.0, code_rate_scale=1.0, jumpy_nll_scale=1.0,
        # --- HL actor / critic (γ^τ) ---
        gamma=step.gamma, lambda_td=step.lambda_td, eta_entropy=step.eta_entropy,
        critic_ema_decay=step.critic_ema_decay,
        return_norm_decay=step.return_norm_decay, return_norm_limit=step.return_norm_limit,
        return_norm_perc_low=step.return_norm_perc_low,
        return_norm_perc_high=step.return_norm_perc_high,
        hl_H=15,                                  # jumpy imagination horizon (in jumps)
        # --- optim ---
        hl_lr=1e-4, hl_eps=1e-8, hl_grad_clip=100.0, weight_decay=0.0,
        # --- loop ---
        prefill=1000, train_ratio=0.5, eval_every=2000, eval_episodes=3, log_every=250,
    )


def tiny():
    """Shape-test preset."""
    c = base()
    c.step = ecfg.tiny()
    c.K = 16; c.code_dim = 32
    c.skill_enc_dim = 64; c.skill_enc_blocks = 1
    c.hl_dim = 64; c.hl_blocks = 1; c.hl_ctx = 4
    c.jumpy_mask_blocks = 1; c.jumpy_decoding_steps = 2
    c.T = 48; c.L_max = 16; c.top_M = 8; c.ell_bar = 8; c.hl_H = 5
    c.prefill = 40; c.eval_every = 40; c.eval_episodes = 1; c.log_every = 5
    return c


PRESETS = {"base": base, "tiny": tiny}
