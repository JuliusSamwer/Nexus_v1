"""Instrument 4 — Multi-run Crafter evaluation overlay (the "is my change promising?" dashboard).

Compare 2+ training runs at a checkpoint — your model vs the vanilla DreamerV3
baseline — on the metrics that matter for long-horizon control, with published
SOTA numbers (EMERALD, ...) overlaid as reference lines:

  1. Crafter score        — official geometric-mean-of-achievements score (0-100).
  2. Per-achievement       — success rate for each of the 22 achievements.
  3. The cliff             — success vs causal-chain depth (the project's core figure),
                             now comparative: does your change lift the deep tier?
  4. Score trajectory      — Crafter score vs env step, so you can read the delta EARLY
                             (same env-step budget = fair, sample-efficiency axis).

Reads what dreamerv3-torch already writes — no GPU, no checkpoint load:
  <logdir>/eval_eps/*.npz   per-episode achievement unlocks (ground-truth Crafter eval)
  <logdir>/metrics.jsonl    scalar series incl. eval_return + eval step boundaries

The controlled baseline should be vanilla dreamerv3-torch trained with the SAME config
as your model (same env, same eval) — that isolates your architecture change. EMERALD/
DIAMOND are cross-codebase, so pass them as --ref reference lines, not --run arms.

Example:
    python3 -m harness.eval_overlay \
        --run baseline=logdir/crafter_smoke \
        --run mymodel=logdir/crafter_mymodel \
        --ref EMERALD=third_party/EMERALD/results/EMERALD.json \
        --out results/eval_overlay
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

from harness.achievement_depth import CAUSAL_DEPTH

# Canonical 22 Crafter achievements (the causal-depth map is the authority).
ACH_NAMES = sorted(CAUSAL_DEPTH, key=lambda a: (CAUSAL_DEPTH[a], a))


# --------------------------------------------------------------------------- #
# Crafter score
# --------------------------------------------------------------------------- #
def crafter_score(success_fraction):
    """Official Crafter score: geometric mean of per-achievement success %, S in [0,100].

    S = exp( mean_i ln(1 + s_i) ) - 1,  with s_i the success rate in PERCENT (0..100).
    Verified to reproduce EMERALD's reported 57.04 from its per-achievement fractions.
    """
    pct = [max(0.0, success_fraction.get(a, 0.0)) * 100.0 for a in ACH_NAMES]
    return math.exp(sum(math.log(1.0 + p) for p in pct) / len(pct)) - 1.0


# --------------------------------------------------------------------------- #
# Loading a run (eval episodes + metrics)
# --------------------------------------------------------------------------- #
def load_episodes(logdir):
    """Each eval episode -> {ach_name: unlocked_bool}. Sorted chronologically by filename
    (the ISO-timestamp prefix sorts in eval order)."""
    import numpy as np

    files = sorted(glob.glob(os.path.join(logdir, "eval_eps", "*.npz")))
    episodes = []
    for fp in files:
        with np.load(fp, allow_pickle=True) as d:
            unlocked = {}
            for a in ACH_NAMES:
                key = "log_achievement_" + a
                unlocked[a] = bool((d[key] > 0).any()) if key in d.files else False
            episodes.append(unlocked)
    return episodes


def load_eval_steps(logdir):
    """Env steps at which an eval happened, in order (rows with a non-None eval_return)."""
    steps = []
    path = os.path.join(logdir, "metrics.jsonl")
    if not os.path.exists(path):
        return steps
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("eval_return") is not None and row.get("step") is not None:
            steps.append(row["step"])
    return sorted(steps)


def load_metric_series(logdir, key):
    """(steps, values) for one scalar key from metrics.jsonl (skips None/non-numeric)."""
    xs, ys = [], []
    path = os.path.join(logdir, "metrics.jsonl")
    if not os.path.exists(path):
        return xs, ys
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        step, v = row.get("step"), row.get(key)
        if step is not None and isinstance(v, (int, float)):
            xs.append(step)
            ys.append(v)
    return xs, ys


def success_from_episodes(episodes):
    """{ach: fraction of episodes in which it was unlocked} over the given episodes."""
    if not episodes:
        return {a: 0.0 for a in ACH_NAMES}
    return {a: sum(ep[a] for ep in episodes) / len(episodes) for a in ACH_NAMES}


def group_into_rounds(episodes, eval_steps, episodes_per_round=None):
    """Split episodes into eval rounds and tag each with its env step.

    Prefers an exact (#episodes == #eval_steps * k) match; otherwise falls back to a
    fixed round size and uses the round index as the x value. Returns
    [(step_or_index, [episodes...]), ...] and a bool 'stepped' = steps are real env steps.
    """
    n = len(episodes)
    if eval_steps and episodes_per_round is None and n % len(eval_steps) == 0:
        epr = n // len(eval_steps)
        rounds = [episodes[i * epr:(i + 1) * epr] for i in range(len(eval_steps))]
        return list(zip(eval_steps, rounds)), True
    epr = episodes_per_round or 1
    rounds = [episodes[i:i + epr] for i in range(0, n, epr)]
    if eval_steps and len(eval_steps) == len(rounds):
        return list(zip(eval_steps, rounds)), True
    return list(zip(range(len(rounds)), rounds)), False


def analyze_run(label, logdir, last_k=None, episodes_per_round=None):
    episodes = load_episodes(logdir)
    eval_steps = load_eval_steps(logdir)
    rounds, stepped = group_into_rounds(episodes, eval_steps, episodes_per_round)

    # Headline snapshot = the checkpoint's performance. Default: the final eval round.
    if last_k is not None and last_k > 0:
        snapshot = episodes[-last_k:]
        snapshot_desc = f"last {len(snapshot)} eval episodes"
    elif rounds:
        snapshot = rounds[-1][1]
        snapshot_desc = f"final eval round ({len(snapshot)} episodes)"
    else:
        snapshot = episodes
        snapshot_desc = f"all {len(episodes)} episodes"

    success = success_from_episodes(snapshot)
    score = crafter_score(success)

    # Trajectory: per-round score + cumulative-to-date score, vs env step.
    traj_x, per_round, cumulative, pool = [], [], [], []
    for x, eps in rounds:
        traj_x.append(x)
        per_round.append(crafter_score(success_from_episodes(eps)))
        pool.extend(eps)
        cumulative.append(crafter_score(success_from_episodes(pool)))

    return {
        "label": label,
        "logdir": logdir,
        "n_episodes": len(episodes),
        "snapshot_desc": snapshot_desc,
        "score": score,
        "success": success,
        "trajectory": {
            "x": traj_x, "stepped": stepped,
            "per_round_score": per_round, "cumulative_score": cumulative,
        },
        "eval_return": load_metric_series(logdir, "eval_return"),
        "model_loss": load_metric_series(logdir, "model_loss"),
    }


def load_reference(label, path):
    """A published baseline (EMERALD-format JSON: per-ach fractions + achievements_score)."""
    d = json.load(open(path))
    success = {
        a: d.get("achievements_" + a, 0.0) for a in ACH_NAMES
    }
    score = d.get("achievements_score")
    if score is None:
        score = crafter_score(success)
    return {"label": label, "path": path, "success": success, "score": score}


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot(runs, refs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run_colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    ref_colors = plt.cm.Set2(np.linspace(0, 1, max(len(refs), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- A: per-achievement success (bars per run, markers per ref) ---
    x = np.arange(len(ACH_NAMES))
    width = 0.8 / max(len(runs), 1)
    for i, r in enumerate(runs):
        ax_vals = [r["success"][a] * 100 for a in ACH_NAMES]
        axA.bar(x + i * width, ax_vals, width, label=r["label"], color=run_colors[i])
    for j, rf in enumerate(refs):
        axA.scatter(x + 0.4 - width / 2, [rf["success"][a] * 100 for a in ACH_NAMES],
                    marker="D", s=22, color=ref_colors[j], edgecolor="black",
                    linewidth=0.4, zorder=5, label=f"{rf['label']} (ref)")
    axA.set_xticks(x + 0.4 - width / 2)
    axA.set_xticklabels([f"{a}  ·d{CAUSAL_DEPTH[a]}" for a in ACH_NAMES],
                        rotation=90, fontsize=7)
    axA.set_ylabel("success rate (%)")
    axA.set_title("Per-achievement success (sorted by causal depth)")
    axA.legend(fontsize=8)
    axA.grid(axis="y", alpha=0.3)

    # --- B: Crafter score (bar per run, dashed line per ref) ---
    labels = [r["label"] for r in runs]
    axB.bar(range(len(runs)), [r["score"] for r in runs],
            color=run_colors[:len(runs)])
    for i, r in enumerate(runs):
        axB.text(i, r["score"], f"{r['score']:.1f}", ha="center", va="bottom", fontsize=10)
    for j, rf in enumerate(refs):
        axB.axhline(rf["score"], ls="--", color=ref_colors[j], lw=1.6,
                    label=f"{rf['label']}: {rf['score']:.1f}")
    axB.set_xticks(range(len(runs)))
    axB.set_xticklabels(labels)
    axB.set_ylabel("Crafter score (0-100)")
    axB.set_title("Crafter score @ checkpoint")
    if refs:
        axB.legend(fontsize=8)
    axB.grid(axis="y", alpha=0.3)

    # --- C: the cliff — mean success vs causal depth ---
    depths = sorted(set(CAUSAL_DEPTH.values()))

    def by_depth(success):
        agg = defaultdict(list)
        for a in ACH_NAMES:
            agg[CAUSAL_DEPTH[a]].append(success[a] * 100)
        return [np.mean(agg[d]) for d in depths]

    for i, r in enumerate(runs):
        axC.plot(depths, by_depth(r["success"]), marker="o", color=run_colors[i],
                 label=r["label"])
    for j, rf in enumerate(refs):
        axC.plot(depths, by_depth(rf["success"]), marker="D", ls="--",
                 color=ref_colors[j], label=f"{rf['label']} (ref)")
    axC.set_xlabel("causal-chain depth (tech-tree rank)")
    axC.set_ylabel("mean success rate (%)")
    axC.set_title("The cliff: success vs causal depth")
    axC.legend(fontsize=8)
    axC.grid(alpha=0.3)

    # --- D: Crafter-score trajectory vs env step ---
    any_stepped = False
    for i, r in enumerate(runs):
        t = r["trajectory"]
        if not t["x"]:
            continue
        any_stepped = any_stepped or t["stepped"]
        axD.plot(t["x"], t["cumulative_score"], marker="o", color=run_colors[i],
                 label=f"{r['label']} (cumulative)")
        axD.plot(t["x"], t["per_round_score"], marker=".", ls=":", alpha=0.5,
                 color=run_colors[i], label=f"{r['label']} (per-round)")
    for j, rf in enumerate(refs):
        axD.axhline(rf["score"], ls="--", color=ref_colors[j], lw=1.4,
                    label=f"{rf['label']}: {rf['score']:.1f}")
    axD.set_xlabel("env step" if any_stepped else "eval round")
    axD.set_ylabel("Crafter score (0-100)")
    axD.set_title("Score trajectory (read the delta early)")
    axD.legend(fontsize=8)
    axD.grid(alpha=0.3)

    fig.suptitle("Crafter evaluation overlay — " + " vs ".join(labels), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


# --------------------------------------------------------------------------- #
def _kv(pairs):
    out = []
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"expected label=path, got: {p}")
        label, path = p.split("=", 1)
        out.append((label, path))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", metavar="label=logdir",
                    help="a training run to compare (repeatable)")
    ap.add_argument("--ref", action="append", metavar="label=results.json",
                    help="published reference baseline, EMERALD-format JSON (repeatable)")
    ap.add_argument("--last-k", type=int, default=None,
                    help="headline = last K eval episodes (default: the final eval round)")
    ap.add_argument("--episodes-per-round", type=int, default=None,
                    help="override round size (default: inferred from eval count)")
    ap.add_argument("--out", default="results/eval_overlay",
                    help="output basename (.png and .json are appended)")
    args = ap.parse_args()

    if not args.run:
        raise SystemExit("need at least one --run label=logdir")

    runs = [analyze_run(lbl, ld, args.last_k, args.episodes_per_round)
            for lbl, ld in _kv(args.run)]
    refs = [load_reference(lbl, p) for lbl, p in _kv(args.ref)]

    # Console summary + head-to-head vs the first run (treated as the baseline arm).
    base = runs[0]
    print(f"\n{'achievement':<22} {'depth':>5}  " +
          "  ".join(f"{r['label']:>10}" for r in runs) +
          ("   " + "  ".join(f"{r['label']:>10}" for r in refs) if refs else ""))
    for a in ACH_NAMES:
        row = (f"{a:<22} {CAUSAL_DEPTH[a]:>5}  " +
               "  ".join(f"{r['success'][a]*100:>9.1f}%" for r in runs))
        if refs:
            row += "   " + "  ".join(f"{rf['success'][a]*100:>9.1f}%" for rf in refs)
        print(row)
    print(f"\n{'SCORE':<22} {'':>5}  " +
          "  ".join(f"{r['score']:>9.2f} " for r in runs) +
          ("   " + "  ".join(f"{rf['score']:>9.2f} " for rf in refs) if refs else ""))
    for r in runs:
        print(f"  · {r['label']}: score {r['score']:.2f}  "
              f"(snapshot = {r['snapshot_desc']}, {r['n_episodes']} eval eps total)")
    for r in runs[1:]:
        delta = r["score"] - base["score"]
        print(f"  Δ {r['label']} vs {base['label']}: {delta:+.2f} score")

    summary = {
        "runs": [{k: r[k] for k in ("label", "logdir", "score", "success",
                                    "snapshot_desc", "n_episodes")} for r in runs],
        "refs": [{"label": rf["label"], "score": rf["score"], "success": rf["success"]}
                 for rf in refs],
        "causal_depth": CAUSAL_DEPTH,
    }
    out_json = args.out + ".json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"\nWrote {out_json}")

    try:
        plot(runs, refs, args.out + ".png")
    except ImportError:
        print("matplotlib not installed; skipped plot.")


if __name__ == "__main__":
    main()
