"""Instrument 1 — Achievement-vs-causal-depth logger.

Reproduce EMERALD's long-horizon cliff: plot per-achievement success against the
causal-chain depth each achievement requires. This is the project's *baseline* figure.

Works fully offline from an EMERALD-format results JSON — no GPU/checkpoint needed.

    python3 -m harness.achievement_depth --results third_party/EMERALD/results/EMERALD.json
"""

import argparse
import json
import os

# Causal-chain depth tiers for Crafter achievements.
# Depth = a coarse rank of how deep into the tech tree / how long a causal chain the
# achievement requires. Refine empirically from successful episodes (see Open Questions
# in the master doc) — these are the plan's tiers as a starting point.
CAUSAL_DEPTH = {
    "collect_wood": 1,
    "place_table": 2,
    "place_plant": 2,
    "collect_sapling": 1,
    "collect_drink": 1,
    "make_wood_pickaxe": 3,
    "make_wood_sword": 3,
    "collect_stone": 3,
    "place_stone": 4,
    "make_stone_pickaxe": 4,
    "make_stone_sword": 4,
    "eat_cow": 2,
    "defeat_zombie": 2,
    "defeat_skeleton": 3,
    "wake_up": 2,
    "place_furnace": 5,
    "collect_coal": 4,
    "collect_iron": 5,
    "make_iron_pickaxe": 6,
    "make_iron_sword": 6,
    "collect_diamond": 7,
    "eat_plant": 7,  # very long, delayed reward
}


def load_success_rates(results_path):
    """Return {achievement: success_rate} from an EMERALD-format results JSON."""
    with open(results_path) as f:
        data = json.load(f)
    return {
        k[len("achievements_"):]: v
        for k, v in data.items()
        if k.startswith("achievements_")
    }


def build_curve(success_rates):
    """Pair each achievement with its causal depth, sorted by depth."""
    rows = []
    for ach, rate in success_rates.items():
        depth = CAUSAL_DEPTH.get(ach)
        if depth is None:
            continue
        rows.append({"achievement": ach, "depth": depth, "success": rate})
    rows.sort(key=lambda r: (r["depth"], -r["success"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="third_party/EMERALD/results/EMERALD.json")
    ap.add_argument("--out", default="results/achievement_depth.json")
    ap.add_argument("--plot", default="results/achievement_depth.png")
    args = ap.parse_args()

    rows = build_curve(load_success_rates(args.results))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {args.out} ({len(rows)} achievements)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. (pip install matplotlib)")
        return
    plt.figure(figsize=(8, 5))
    plt.scatter([r["depth"] for r in rows], [r["success"] for r in rows])
    for r in rows:
        plt.annotate(r["achievement"], (r["depth"], r["success"]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    plt.xlabel("Causal-chain depth (tech-tree rank)")
    plt.ylabel("Success rate")
    plt.title("EMERALD: achievement success vs. causal depth (the cliff)")
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
