"""Train the segment-native world model — Phase N1 (full WM, seg=scheduled).

WM-only: this trains the world model and reports the §8 diagnostics that answer the
bottleneck bet — does (u_n, z_{t_n}) suffice to re-init prediction across a boundary?
There is no actor yet (that is N2; collection here is random-policy, and the loop is
structured so an actor can replace `collect_episode`).

The N1 headline experiment is the {w} x {G} grid (§11.4); pass --w / --G to set a cell and
tag the logdir. Start with the corners: w=0/G=4 (strict reference), w=64/G=4 (leak ref).

    python3 -m nexus.train --configs crafter --device cuda \
        --logdir logdir/nexus_w0_G4 --w 0 --G 4 --steps 100000
"""

import argparse
import os

import numpy as np
import torch

from emerald_torch import env as envmod
from emerald_torch.train import prefill_episode
from . import config as config_mod
from . import segments as segmod
from .model import NexusWM
from .replay import StreamReplay


def train_step(model, opt, batch, segs, cfg):
    model.train()
    loss, metrics = model.loss(batch, segs)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.step.model_grad_max_norm)
    opt.step()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="crafter", choices=list(config_mod.PRESETS))
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=100000, help="env steps")
    ap.add_argument("--w", type=int, default=None, help="leak width override (0/16/64)")
    ap.add_argument("--G", type=int, default=None, help="slow tokens per segment override")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    cfg = config_mod.PRESETS[args.configs]()
    if args.w is not None:
        cfg.w = args.w
    if args.G is not None:
        cfg.G = args.G

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    os.makedirs(args.logdir, exist_ok=True)
    print(f"NexusWM (N1) | preset={args.configs} device={dev} "
          f"T={cfg.T} ell_bar={cfg.ell_bar} w={cfg.w} G={cfg.G} steps={args.steps}")

    env = envmod.CrafterEnv(seed=args.seed)
    model = NexusWM(cfg, env.num_actions).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: total={n_params:,} "
          f"(fast={sum(p.numel() for p in model.fast.parameters()):,}, "
          f"slow={sum(p.numel() for p in model.post.parameters()) + sum(p.numel() for p in model.prior.parameters()):,})")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.step.model_lr,
                           eps=cfg.step.model_eps, weight_decay=cfg.weight_decay)
    replay = StreamReplay(capacity=min(args.steps + cfg.prefill, 200000),
                          num_actions=env.num_actions)
    writer = envmod.MetricsWriter(args.logdir)

    print(f"Prefilling {cfg.prefill} steps (random policy)...")
    while replay.num_steps < max(cfg.prefill, cfg.T):
        replay.add_episode(prefill_episode(env, cfg.step).to_replay())

    env_steps, train_acc, last_log = 0, 0.0, 0
    while env_steps < args.steps:
        ep = prefill_episode(env, cfg.step).to_replay()      # N1: random-policy collection
        replay.add_episode(ep)
        env_steps += len(ep["reward"])
        train_acc += cfg.train_ratio * len(ep["reward"])
        metrics = None
        while train_acc >= 1.0 and replay.can_sample(cfg.T):
            batch = replay.sample(cfg.batch_size, cfg.T, dev)
            segs = segmod.make_segments(cfg, dev, rng)        # fresh schedule per batch
            metrics = train_step(model, opt, batch, segs, cfg)
            train_acc -= 1.0

        if metrics is not None and env_steps - last_log >= cfg.log_every:
            metrics["env_steps"] = env_steps
            metrics["w"], metrics["G"] = cfg.w, cfg.G
            writer.write(env_steps, metrics)
            last_log = env_steps
            print(f"[{env_steps:>7}] loss={metrics['loss']:.1f} "
                  f"img={metrics['fast_image']:.1f} "
                  f"spike={metrics.get('post_boundary_spike', float('nan')):.2f} "
                  f"ground_nll={metrics['ground_nll_diag']:.1f} "
                  f"slow_adv={metrics['slow_advantage']:.1f} "
                  f"u_ppl={metrics['u_perplexity']:.1f}")

        if env_steps // cfg.save_every > (env_steps - len(ep["reward"])) // cfg.save_every:
            torch.save({"model": model.state_dict(), "step": env_steps, "cfg_w": cfg.w,
                        "cfg_G": cfg.G}, os.path.join(args.logdir, "latest.pt"))

    torch.save({"model": model.state_dict(), "step": env_steps, "cfg_w": cfg.w,
                "cfg_G": cfg.G}, os.path.join(args.logdir, "latest.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
