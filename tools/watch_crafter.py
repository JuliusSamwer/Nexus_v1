#!/usr/bin/env python3
"""
Watch an agent play Crafter — records an episode to an mp4 you can open.

Runs locally on the Mac (no GPU, no training needed). The policy is pluggable:
  - default: random actions (works today, lets you see the world + survival loop)
  - --checkpoint <path>: load a trained DreamerV3 agent (wired once we train one)

    python3 tools/watch_crafter.py --steps 400 --out results/crafter_play.mp4
    python3 tools/watch_crafter.py --steps 400 --reward-policy   # crude survival heuristic

The point: get something on screen immediately and prove the local loop works. A
*good* agent comes from training (watch eval videos improve, then a GPU run); this is
the visceral baseline you can run any time.
"""

import argparse
import numpy as np
import crafter
import imageio.v2 as imageio


def upscale(frame, factor):
    """Nearest-neighbour upscale so 64x64 pixels stay crisp in the video."""
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def random_policy(obs, info, action_n, rng):
    return rng.integers(action_n)


def make_policy(args, action_n):
    rng = np.random.default_rng(args.seed)
    if args.checkpoint:
        raise NotImplementedError(
            "Trained-agent playback not wired yet. Train a DreamerV3 agent "
            "(third_party/dreamerv3-torch) then load its actor here. Until then, omit "
            "--checkpoint to watch a random policy."
        )
    return lambda obs, info: random_policy(obs, info, action_n, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400, help="Max steps to record.")
    ap.add_argument("--out", default="results/crafter_play.mp4")
    ap.add_argument("--render-size", type=int, default=512, help="Output video resolution (px).")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None, help="(future) trained DreamerV3 checkpoint to play.")
    args = ap.parse_args()

    # Render natively at a multiple of 64 for crisp upscaling.
    factor = max(1, args.render_size // 64)
    env = crafter.Env(size=(64, 64), seed=args.seed)
    policy = make_policy(args, env.action_space.n)

    obs = env.reset()
    info = {}
    frames = [upscale(np.asarray(env.render()), factor)]
    episodes, deaths, total_reward = 1, 0, 0.0
    unlocked = set()

    for t in range(args.steps):
        action = int(policy(obs, info))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        for ach, n in info.get("achievements", {}).items():
            if n > 0:
                unlocked.add(ach)
        frames.append(upscale(np.asarray(env.render()), factor))
        if done:
            deaths += 1
            obs = env.reset()
            episodes += 1

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Wrote {args.out}  ({len(frames)} frames, {args.fps} fps)")
    print(f"Episodes: {episodes} | deaths: {deaths} | total reward: {total_reward:.1f}")
    print(f"Achievements unlocked this run ({len(unlocked)}): "
          f"{', '.join(sorted(unlocked)) or 'none (random policy — expected)'}")


if __name__ == "__main__":
    main()
