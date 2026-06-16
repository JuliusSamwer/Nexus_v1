"""Non-learned predictors of rollout divergence — the bars the conditional critic must
beat or match. Each maps a per-step feature dict (from paired_rollout) to a 1-D score
where higher = more predicted divergence.

  disagreement  : K-sample imagination disagreement (variance across K stochastic prior
                  samples). The cheap, single-WM proxy for ensemble disagreement — a TRUE
                  K-model deep ensemble is the stronger bar but out of week-one scope.
  entropy       : prior/token entropy of the imagined step (the WM's own intrinsic uncertainty).
  knn_density   : latent k-NN distance to a reference bank (out-of-distribution-ness).
  horizon       : the imagination step index (the trivial baseline; C must beat this).

These are intrinsic / non-differentiable — they detect but give no actionable gradient.
That asymmetry (vs the differentiable C) is the paper's wedge.
"""

import numpy as np

INTRINSIC_KEYS = ["disagreement", "entropy", "knn_density", "horizon"]


def predictor_scores(data, key):
    """Pull a 1-D intrinsic score column out of the saved dataset."""
    if key not in data:
        raise KeyError(f"intrinsic '{key}' not in dataset (have {list(data.keys())})")
    return np.asarray(data[key], float)


def isotonic_calibrate(scores, labels):
    """Map raw scores -> probabilities via isotonic regression (PAV), so intrinsics get a
    fair calibration curve / ECE. Pure numpy. Returns calibrated probs aligned to `scores`."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, float)
    order = np.argsort(scores, kind="mergesort")
    y = labels[order]
    # Pool Adjacent Violators
    w = np.ones_like(y)
    g = y.copy()
    i = 0
    bounds = list(range(len(y) + 1))
    # simple O(n^2) PAV (datasets here are small)
    level = g.tolist()
    weight = w.tolist()
    blocks = [[v, wt] for v, wt in zip(level, weight)]
    merged = []
    for v, wt in blocks:
        merged.append([v, wt])
        while len(merged) > 1 and merged[-2][0] >= merged[-1][0]:
            v2, w2 = merged.pop()
            v1, w1 = merged.pop()
            nv = (v1 * w1 + v2 * w2) / (w1 + w2)
            merged.append([nv, w1 + w2])
    fitted = []
    for v, wt in merged:
        fitted.extend([v] * int(wt))
    fitted = np.array(fitted)
    out = np.empty_like(fitted)
    out[np.arange(len(y))] = fitted
    probs = np.empty_like(scores)
    probs[order] = np.clip(out, 1e-6, 1 - 1e-6)
    return probs
