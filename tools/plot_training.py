#!/usr/bin/env python3
"""
Plot DreamerV3 training curves from a run's metrics.jsonl.

dreamerv3-torch's Logger appends one JSON object per log step to
<logdir>/metrics.jsonl (step + all scalars). This reads it and plots the
informative scalars vs. step so you can see the rough trend.

    python3 tools/plot_training.py --logdir logdir/crafter_smoke --out results/training_curves.png
"""

import argparse
import json
import os
from collections import defaultdict

# Curated panels, in display order (only those present are shown).
CURATED = ["model_loss", "image_loss", "kl", "eval_return", "train_return",
           "actor_loss", "value_loss", "achievements_cumulative"]


def add_cumulative_achievements(series):
    """Synthesize a 'distinct achievements ever unlocked' curve from log_achievement_* counts."""
    ach_keys = [k for k in series if k.startswith("log_achievement_")]
    if not ach_keys:
        return
    # Gather (step, achievement) firsts where count > 0.
    first_step = {}
    for k in ach_keys:
        for st, v in zip(*series[k]):
            if v and v > 0:
                name = k.replace("log_achievement_", "")
                first_step[name] = min(first_step.get(name, st), st)
    if not first_step:
        return
    steps = sorted(set(first_step.values()))
    xs, ys, seen = [], [], 0
    unlocked_by_step = {}
    for s in steps:
        unlocked_by_step[s] = sum(1 for fs in first_step.values() if fs <= s)
    for s in steps:
        xs.append(s)
        ys.append(unlocked_by_step[s])
    series["achievements_cumulative"] = (xs, ys)


def load_metrics(path):
    series = defaultdict(lambda: ([], []))  # key -> (steps, values)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = row.get("step")
            if step is None:
                continue
            for k, v in row.items():
                if k == "step" or not isinstance(v, (int, float)):
                    continue
                series[k][0].append(step)
                series[k][1].append(v)
    return series


def pick_keys(series):
    return [k for k in CURATED if k in series] or list(series.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", default="results/training_curves.png")
    args = ap.parse_args()

    path = os.path.join(args.logdir, "metrics.jsonl")
    series = load_metrics(path)
    add_cumulative_achievements(series)
    print(f"Loaded {len(series)} scalar series from {path}")
    if "achievements_cumulative" in series:
        xs, ys = series["achievements_cumulative"]
        print(f"Distinct achievements unlocked during training: {ys[-1]} (by step {xs[-1]})")

    keys = pick_keys(series)[:9]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // cols][i % cols]
        steps, vals = series[k]
        ax.plot(steps, vals, marker="o", ms=2, lw=1)
        ax.set_title(k, fontsize=10)
        ax.set_xlabel("env step")
        ax.grid(alpha=0.3)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("DreamerV3 on Crafter (local MPS smoke run) — rough trends", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
