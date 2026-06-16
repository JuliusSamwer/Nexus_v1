"""Week-one go/no-go runner: generate paired rollouts -> train C + ablations -> score
every predictor against every divergence label on HELD-OUT seeds -> print the table.

Phases (decoupled so the critic never needs the WM's framework):
  --phase generate : run the adapter (torch venv for the 5M ckpt; jax venv for craftax),
                     dump features.npz.
  --phase analyze  : load features.npz (always in the jax venv), train/eval, print table.
  --phase all      : both, in the current venv (used for the jax smoke).

    python -m harness.critic.run_gate --phase generate --substrate torch \
        --checkpoint <5M.ckpt> --out runs/critic/c5m.npz --n_starts 200 --H 15
    python -m harness.critic.run_gate --phase analyze --data runs/critic/c5m.npz
"""

import argparse
import json
import os

import numpy as np

LABELS = {"L1": "token-mismatch", "L2": "kl(true||prior)", "L3": "decoded-obs",
          "L4": "reward-div", "L4b": "value-div"}
PREDICTORS_LEARNED = ["conditional", "marginal", "horizon"]
PREDICTORS_INTRINSIC = ["disagreement", "entropy", "knn_density"]


def analyze(data_path, out_dir, hidden=(128, 128), steps=3000):
    from . import critic as C
    from . import baselines as B
    from . import metrics as M
    d = dict(np.load(data_path))
    seeds = d["seed"]
    os.makedirs(out_dir, exist_ok=True)
    # seed-disjoint TEST split (hold out last 20% of unique seeds)
    uniq = np.unique(seeds)
    n_test = max(1, int(len(uniq) * 0.2))
    test_seeds = set(uniq[-n_test:].tolist())
    te = np.array([s in test_seeds for s in seeds])
    tr = ~te
    feats = {w: C.build_features(d, w) for w in PREDICTORS_LEARNED}
    table = {}
    for lab in LABELS:
        col = d[lab]
        if not np.isfinite(col).any():
            continue
        valid = np.isfinite(col)
        eps = float(np.nanmedian(col[tr & valid]))
        row = {}
        # learned critic + ablations (trained on TRAIN seeds, scored on TEST)
        for w in PREDICTORS_LEARNED:
            ybin = (col > eps).astype(np.float32)
            m = tr & valid
            trained = C.train_critic(feats[w][m], ybin[m], seeds[m],
                                     hidden=hidden, steps=steps)
            mt = te & valid
            sc = trained["score"](feats[w][mt])
            pr = trained["prob"](feats[w][mt])
            row[w] = M.evaluate_predictor(sc, col[mt], eps, probs=pr)
        # intrinsic baselines (no training; score directly on TEST)
        for w in PREDICTORS_INTRINSIC:
            if w not in d:
                continue
            mt = te & valid
            sc = np.asarray(d[w])[mt]
            ybin = (col[mt] > eps).astype(int)
            probs = B.isotonic_calibrate(sc, ybin)
            row[w] = M.evaluate_predictor(sc, col[mt], eps, probs=probs)
        table[lab] = row
    _print_table(table)
    with open(os.path.join(out_dir, "gate_results.json"), "w") as f:
        json.dump(table, f, indent=2)
    _verdict(table)
    return table


def _print_table(table):
    preds = PREDICTORS_LEARNED + PREDICTORS_INTRINSIC
    print("\n================  CONDITIONAL-CRITIC GATE — AUROC (Spearman)  ================")
    head = f"{'label':<16}" + "".join(f"{p:>16}" for p in preds)
    print(head)
    print("-" * len(head))
    for lab, row in table.items():
        line = f"{LABELS[lab]:<16}"
        for p in preds:
            if p in row and np.isfinite(row[p]["auroc"]):
                line += f"{row[p]['auroc']:>8.3f}({row[p]['spearman']:+.2f})"
            else:
                line += f"{'-':>16}"
        print(line)
    print("=" * len(head))


def _verdict(table):
    print("\nGO/NO-GO (per label):")
    for lab, row in table.items():
        if "conditional" not in row:
            continue
        c = row["conditional"]["auroc"]
        marg = row.get("marginal", {}).get("auroc", float("nan"))
        hor = row.get("horizon", {}).get("auroc", float("nan"))
        dis = row.get("disagreement", {}).get("auroc", float("nan"))
        beats_floor = (c > marg) and (c > hor)
        vs_ens = "≥ ensemble" if (np.isnan(dis) or c >= dis - 0.01) else "< ensemble"
        flag = "PASS-floor" if beats_floor else "FAIL-floor(idea broken)"
        print(f"  {LABELS[lab]:<16} C={c:.3f}  marg={marg:.3f} hor={hor:.3f} "
              f"dis={dis:.3f}  [{flag}; {vs_ens}]")
    print("\nReminder: detection-parity with disagreement is a PASS (C is differentiable; "
          "the ensemble isn't). The kill is failing the floor (marginal/horizon).")


def generate_cmd(args):
    from . import wm_adapter, paired_rollout
    cfg = None
    if args.substrate == "jax":
        from emerald_jax import config as cfgmod
        cfg = cfgmod.tiny() if args.tiny else cfgmod.craftax()
    adapter = wm_adapter.make_adapter(args.substrate, checkpoint=args.checkpoint, cfg=cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    paired_rollout.generate(adapter, args.out, n_rollouts=args.n_starts, H=args.H,
                            mixture_p=args.mixture_p, K=args.K, warmup=args.warmup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["generate", "analyze", "all"], default="all")
    ap.add_argument("--substrate", choices=["torch", "jax"], default="jax")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", default="runs/critic/features.npz")
    ap.add_argument("--out", default="runs/critic/features.npz")
    ap.add_argument("--out_dir", default="runs/critic")
    ap.add_argument("--n_starts", type=int, default=200)
    ap.add_argument("--H", type=int, default=15)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--mixture_p", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tiny", action="store_true", help="jax: use config.tiny()")
    args = ap.parse_args()

    if args.phase in ("generate", "all"):
        generate_cmd(args)
    if args.phase in ("analyze", "all"):
        analyze(args.out if args.phase == "all" else args.data, args.out_dir, steps=args.steps)


if __name__ == "__main__":
    main()
