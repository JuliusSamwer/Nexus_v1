"""EMERALD configuration (defaults straight from nnet/models/emerald.py) plus local
override presets. Access fields as attributes."""

from types import SimpleNamespace


def base():
    return SimpleNamespace(
        # latent / encoder
        image_channels=3, dim_cnn=32, stoch_size=32, discrete=32, dim_model=512,
        reduced_channels=128, num_res_layers=1, uniform_mix=0.01, bins=255,
        free_nats=1.0, learn_initial=True,
        # heads
        num_layers=2,
        # tssm / maskgit
        num_blocks_trans=4, ff_ratio_trans=2, num_heads_trans=8, drop_rate_trans=0.1,
        num_blocks_mask=2, num_decoding_steps=3, att_context_left=64, img_stride=4,
        # rollout
        L=64, H=15, batch_size=16,
        # loss scales
        loss_decoder_scale=1.0, loss_kl_prior_scale=0.5, loss_kl_post_scale=0.1,
        loss_kl_mask_scale=0.5, loss_reward_scale=1.0, loss_continue_scale=1.0,
        # actor-critic
        gamma=0.997, lambda_td=0.95, eta_entropy=0.0003, target_value_reg=True,
        critic_ema_decay=0.02, critic_slow_reg_scale=1.0,
        return_norm_decay=0.99, return_norm_limit=1.0,
        return_norm_perc_low=0.05, return_norm_perc_high=0.95,
        # optim
        model_lr=1e-4, value_lr=3e-5, actor_lr=3e-5,
        model_eps=1e-8, value_eps=1e-5, actor_eps=1e-5,
        model_grad_max_norm=1000.0, value_grad_max_norm=100.0, actor_grad_max_norm=100.0,
        weight_decay=0.0,
        # training loop
        prefill=1000, train_ratio=1.0,          # train steps per env step
        eval_every=2000, eval_episodes=3, log_every=250, save_every=5000,
    )


def crafter_smoke():
    """Local MPS preset — comparable to the dreamerv3-torch crafter_smoke run."""
    c = base()
    c.batch_size = 16
    c.L = 64
    c.train_ratio = 0.5
    c.eval_every = 2000
    c.eval_episodes = 3
    c.log_every = 250
    return c


def crafter_fast():
    """Lighter local preset — SAME architecture, smaller batch/window so a 20k-step
    comparison finishes locally in a few hours instead of ~15h. Reduces data per
    update, not the model."""
    c = base()
    c.batch_size = 8
    c.L = 32
    c.att_context_left = 32
    c.train_ratio = 0.5
    c.eval_every = 2000
    c.eval_episodes = 3
    c.log_every = 250
    return c


def tiny():
    """Tiny correctness-test preset (CPU/MPS shape shakeout)."""
    c = base()
    c.batch_size = 2
    c.L = 16
    c.H = 5
    c.att_context_left = 16
    c.num_blocks_trans = 2
    c.num_decoding_steps = 2
    c.prefill = 40
    c.train_ratio = 1.0
    c.eval_every = 30
    c.eval_episodes = 1
    c.log_every = 5
    return c


PRESETS = {"base": base, "crafter_smoke": crafter_smoke, "crafter_fast": crafter_fast,
           "tiny": tiny}
