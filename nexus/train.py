"""Train Nexus_v1 on Crafter — the 5-stage loop (§8).

  Stage 0  collect          — step actor acts; replay fills (one buffer, two sample lengths)
  Stage 1  step WM + step AC — EMERALD on len-step.L (reused verbatim)
  Stage 2  segment          — semi-Markov DP on len-T → boundaries → skill codes
  Stage 3  jumpy WM + HL AC  — option-model heads + Hₙ; HL actor/critic (γ^τ)
  Stage 4  close             — step actor conditioned on k  [v1 GAP: deferred]

Reuses emerald_torch's collect/eval/step-train wholesale; writes the same
metrics.jsonl / eval_eps so harness/eval_overlay.py reads it.

    python3 -m nexus.train --configs tiny --device mps --logdir logdir/nexus_tiny --steps 200
"""

import argparse
import os

import numpy as np
import torch

from emerald_torch import env as envmod
from emerald_torch.replay import ReplayBuffer
from emerald_torch.train import (make_optims, run_episode, prefill_episode,
                                 evaluate, train_step as step_train)
from . import config as config_mod
from .model import NexusAgent


def hl_train_step(agent, opt, hl, cfg):
    agent.train()
    wm_loss, m1, starts = agent.hl_world_model_loss(hl)
    a_loss, v_loss, m2 = agent.hl_actor_critic_loss(starts)
    loss = wm_loss + a_loss + v_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.skill_tier_parameters(), cfg.hl_grad_clip)
    opt.step()
    return {**m1, **m2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="tiny", choices=list(config_mod.PRESETS))
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--hl_every", type=int, default=4, help="step-trains between HL trains")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = config_mod.PRESETS[args.configs]()
    cfg.step.att_context_left = min(cfg.step.att_context_left, cfg.step.L)
    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    os.makedirs(args.logdir, exist_ok=True)
    print(f"Nexus_v1 | preset={args.configs} device={device} steps={args.steps}")

    env = envmod.CrafterEnv(seed=args.seed)
    eval_env = envmod.CrafterEnv(seed=args.seed + 10000)
    agent = NexusAgent(cfg, env.num_actions).to(device)
    print(f"params: total={sum(p.numel() for p in agent.parameters()):,} "
          f"(step={sum(p.numel() for p in agent.step.parameters()):,}, "
          f"skill-tier={sum(p.numel() for p in agent.skill_tier_parameters()):,})")
    step_optims = make_optims(agent.step, cfg.step)
    hl_opt = torch.optim.Adam(agent.skill_tier_parameters(), lr=cfg.hl_lr, eps=cfg.hl_eps,
                              weight_decay=cfg.weight_decay)
    replay = ReplayBuffer(capacity=int(1e6), num_actions=env.num_actions)
    writer = envmod.MetricsWriter(args.logdir)

    print(f"Prefilling {cfg.prefill} steps...")
    while replay.num_steps < cfg.prefill:
        replay.add_episode(prefill_episode(env, cfg.step).to_replay())

    env_steps, train_acc, n_train, last_log = 0, 0.0, 0, 0
    while env_steps < args.steps:
        ep, ret = run_episode(agent.step, env, cfg.step, device, sample=True)
        rep = ep.to_replay()
        replay.add_episode(rep)
        env_steps += len(rep["reward"])
        train_acc += cfg.train_ratio * len(rep["reward"])
        metrics = None
        while train_acc >= 1.0 and replay.can_sample(cfg.step.L):
            batch = replay.sample(cfg.step.batch_size, cfg.step.L, device)
            metrics = step_train(agent.step, step_optims, batch, cfg.step)
            n_train += 1
            # Stage 2-3: HL train on a len-T window, less frequently
            if n_train % args.hl_every == 0 and replay.can_sample(cfg.T):
                img = replay.sample(max(2, cfg.step.batch_size // 4), cfg.T, device)
                hl = agent.encode_hl(img)
                hlm = hl_train_step(agent, hl_opt, hl, cfg)
                metrics.update(hlm)
            train_acc -= 1.0

        if metrics is not None and env_steps - last_log >= cfg.log_every:
            metrics["train_return"] = ret
            writer.write(env_steps, metrics)
            last_log = env_steps
            seglen = metrics.get("mean_seg_len", float("nan"))
            print(f"[{env_steps:>7}] step_model={metrics['model_loss']:.0f} "
                  f"| hl_term={metrics.get('hl_terminal', float('nan')):.2f} "
                  f"hl_tau={metrics.get('hl_tau', float('nan')):.2f} "
                  f"vq_ppl={metrics.get('vq_perplexity', float('nan')):.1f} "
                  f"seglen={seglen:.1f} | ep_ret={ret:.1f}")

        if env_steps // cfg.eval_every > (env_steps - len(rep["reward"])) // cfg.eval_every:
            evaluate(agent.step, eval_env, cfg.step, device, args.logdir, env_steps, writer)
            torch.save({"agent": agent.state_dict(), "step": env_steps},
                       os.path.join(args.logdir, "latest.pt"))

    evaluate(agent.step, eval_env, cfg.step, device, args.logdir, env_steps, writer)
    torch.save({"agent": agent.state_dict(), "step": env_steps},
               os.path.join(args.logdir, "latest.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
