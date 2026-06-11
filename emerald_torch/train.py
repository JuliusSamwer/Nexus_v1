"""Train standalone EMERALD on Crafter, writing dreamerv3-torch-compatible logs.

    python3 -m emerald_torch.train --configs crafter_smoke --device mps \
        --logdir logdir/emerald_smoke --steps 20000

Compare against the dreamer run with:
    python3 -m harness.eval_overlay \
        --run dreamer=logdir/crafter_smoke --run emerald=logdir/emerald_smoke \
        --ref EMERALD=third_party/EMERALD/results/EMERALD.json --out results/eval_overlay
"""

import argparse
import os

import numpy as np
import torch

from . import config as config_mod
from . import env as envmod
from .model import EmeraldAgent
from .replay import ReplayBuffer


def make_optims(agent, cfg):
    return {
        "wm": torch.optim.Adam(agent.wm_parameters(), lr=cfg.model_lr, eps=cfg.model_eps,
                               weight_decay=cfg.weight_decay),
        "actor": torch.optim.Adam(agent.actor_parameters(), lr=cfg.actor_lr,
                                  eps=cfg.actor_eps, weight_decay=cfg.weight_decay),
        "critic": torch.optim.Adam(agent.critic_parameters(), lr=cfg.value_lr,
                                   eps=cfg.value_eps, weight_decay=cfg.weight_decay),
    }


class EpisodeAcc:
    def __init__(self):
        self.image, self.action, self.reward = [], [], []
        self.cont, self.is_terminal, self.ach = [], [], []

    def add(self, image, action, reward, terminal, ach_counts):
        self.image.append(image)
        self.action.append(action)
        self.reward.append(reward)
        self.cont.append(1.0 - float(terminal))
        self.is_terminal.append(bool(terminal))
        self.ach.append(dict(ach_counts))

    def to_replay(self):
        ach = {k: [a[k] for a in self.ach] for k in envmod.ACH_NAMES}
        return {
            "image": np.stack(self.image).astype(np.uint8),
            "action": np.stack(self.action).astype(np.float32),
            "reward": np.array(self.reward, np.float32),
            "cont": np.array(self.cont, np.float32),
            "is_terminal": self.is_terminal, "ach": ach,
        }


def run_episode(agent, env, cfg, device, sample=True):
    """Roll one full episode with the current policy. Returns (episode_acc, return)."""
    ctx = cfg.att_context_left
    img, info = env.reset()
    stoch_hist, action_hist = [], [torch.zeros(env.num_actions, device=device)]
    ep = EpisodeAcc()
    prev_action = np.zeros(env.num_actions, np.float32)
    reward_in, terminal_in = 0.0, False
    total_return, steps = 0.0, 0
    agent.eval()
    while True:
        img_t = torch.from_numpy(img).float().div(255).sub(0.5).to(device).view(1, 1, *img.shape)
        action, stoch_t = agent.act(img_t, stoch_hist, action_hist, sample=sample)
        ep.add(img, prev_action, reward_in, terminal_in, info["achievements"])
        a_idx = int(action.argmax().item())
        img, reward, done, info = env.step(a_idx)
        total_return += reward
        steps += 1
        prev_action = action.detach().cpu().numpy()
        reward_in = reward
        terminal_in = info.get("discount", 1.0) == 0
        stoch_hist.append(stoch_t.detach())
        action_hist.append(action.detach())
        stoch_hist[:] = stoch_hist[-ctx:]
        action_hist[:] = action_hist[-(ctx + 1):]
        if done or steps >= 10000:
            ep.add(img, prev_action, reward_in, terminal_in, info["achievements"])
            break
    agent.train()
    return ep, total_return


def prefill_episode(env, cfg):
    """Random-policy episode for prefill (no model needed)."""
    img, info = env.reset()
    ep = EpisodeAcc()
    prev_action = np.zeros(env.num_actions, np.float32)
    reward_in, terminal_in = 0.0, False
    steps = 0
    while True:
        ep.add(img, prev_action, reward_in, terminal_in, info["achievements"])
        a_idx = np.random.randint(env.num_actions)
        img, reward, done, info = env.step(a_idx)
        prev_action = envmod.onehot(a_idx, env.num_actions)
        reward_in = reward
        terminal_in = info.get("discount", 1.0) == 0
        steps += 1
        if done or steps >= 10000:
            ep.add(img, prev_action, reward_in, terminal_in, info["achievements"])
            break
    return ep


def train_step(agent, optims, batch, cfg):
    agent.train()
    # World model
    loss, metrics, detached = agent.world_model_loss(batch)
    optims["wm"].zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.wm_parameters(), cfg.model_grad_max_norm)
    optims["wm"].step()
    # Actor + critic (imagination with dropout off in the dynamics)
    agent.tssm.eval()
    actor_loss, value_loss, ac_metrics = agent.actor_critic_loss(detached)
    agent.tssm.train()
    optims["actor"].zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor_parameters(), cfg.actor_grad_max_norm)
    optims["actor"].step()
    optims["critic"].zero_grad(set_to_none=True)
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.critic_parameters(), cfg.value_grad_max_norm)
    optims["critic"].step()
    agent.update_target()
    metrics.update(ac_metrics)
    return metrics


def evaluate(agent, eval_env, cfg, device, logdir, step, writer):
    returns, lengths = [], []
    ach_unlocked = {k: 0 for k in envmod.ACH_NAMES}
    for _ in range(cfg.eval_episodes):
        ep, ret = run_episode(agent, eval_env, cfg, device, sample=True)
        rep = ep.to_replay()
        envmod.save_eval_episode(os.path.join(logdir, "eval_eps"), rep)
        returns.append(ret)
        lengths.append(len(rep["reward"]))
        for k in envmod.ACH_NAMES:
            if max(rep["ach"][k]) > 0:
                ach_unlocked[k] += 1
    row = {"eval_return": float(np.mean(returns)), "eval_length": float(np.mean(lengths)),
           "eval_episodes": cfg.eval_episodes}
    for k in envmod.ACH_NAMES:
        row[f"eval_achievement_{k}"] = ach_unlocked[k] / cfg.eval_episodes
    writer.write(step, row)
    print(f"[eval @ {step}] return={row['eval_return']:.2f} "
          f"len={row['eval_length']:.0f} unlocked={sum(v>0 for v in ach_unlocked.values())}/22")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="crafter_smoke", choices=list(config_mod.PRESETS))
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=20000, help="env steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = config_mod.PRESETS[args.configs]()
    cfg.att_context_left = min(cfg.att_context_left, cfg.L)
    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    os.makedirs(args.logdir, exist_ok=True)
    print(f"EMERALD-torch | preset={args.configs} device={device} steps={args.steps}")

    env = envmod.CrafterEnv(seed=args.seed)
    eval_env = envmod.CrafterEnv(seed=args.seed + 10000)
    agent = EmeraldAgent(cfg, env.num_actions).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {n_params:,}")
    optims = make_optims(agent, cfg)
    replay = ReplayBuffer(capacity=int(1e6), num_actions=env.num_actions)
    writer = envmod.MetricsWriter(args.logdir)

    # Prefill
    print(f"Prefilling {cfg.prefill} steps (random policy)...")
    while replay.num_steps < cfg.prefill:
        replay.add_episode(prefill_episode(env, cfg).to_replay())

    env_steps = 0
    train_acc = 0.0
    last_log = 0
    while env_steps < args.steps:
        ep, ret = run_episode(agent, env, cfg, device, sample=True)
        rep = ep.to_replay()
        replay.add_episode(rep)
        env_steps += len(rep["reward"])

        # Train steps proportional to env steps collected this episode
        train_acc += cfg.train_ratio * len(rep["reward"])
        metrics = None
        while train_acc >= 1.0 and replay.can_sample(cfg.L):
            batch = replay.sample(cfg.batch_size, cfg.L, device)
            metrics = train_step(agent, optims, batch, cfg)
            train_acc -= 1.0

        if metrics is not None and env_steps - last_log >= cfg.log_every:
            metrics["train_return"] = ret
            metrics["env_steps"] = env_steps
            writer.write(env_steps, metrics)
            last_log = env_steps
            print(f"[{env_steps:>7}] model_loss={metrics['model_loss']:.1f} "
                  f"image={metrics['image_loss']:.1f} kl_post={metrics['kl_post']:.2f} "
                  f"actor={metrics['actor_loss']:.2f} value={metrics['value_loss']:.2f} "
                  f"ep_ret={ret:.1f}")

        if env_steps // cfg.eval_every > (env_steps - len(rep["reward"])) // cfg.eval_every:
            evaluate(agent, eval_env, cfg, device, args.logdir, env_steps, writer)
            torch.save({"agent": agent.state_dict(), "step": env_steps},
                       os.path.join(args.logdir, "latest.pt"))

    evaluate(agent, eval_env, cfg, device, args.logdir, env_steps, writer)
    torch.save({"agent": agent.state_dict(), "step": env_steps},
               os.path.join(args.logdir, "latest.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
