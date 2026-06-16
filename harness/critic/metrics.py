"""Detection metrics, pure numpy (no sklearn): AUROC, Spearman rho, calibration/ECE.

A "predictor" is a 1-D score; the binary task is (true divergence d > epsilon). Higher
score should mean higher predicted divergence. All metrics are computed on held-out
seeds by the caller; this module is split-agnostic.
"""

import numpy as np


def auroc(scores, labels):
    """Area under ROC for binary `labels` (0/1) ranked by `scores`. Rank-based
    (Mann-Whitney U), ties handled by average rank. Returns 0.5 if degenerate."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(x, y):
    """Spearman rank correlation between two 1-D arrays."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    a_sorted = a[order]
    i = 0
    while i < len(a_sorted):
        j = i
        while j + 1 < len(a_sorted) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def calibration(probs, labels, n_bins=10):
    """Reliability curve + Expected Calibration Error for probabilistic predictions in
    [0,1]. Returns (bin_centers, bin_acc, bin_conf, ece). Empty bins -> nan."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs = [], [], []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (probs >= lo) & (probs < hi if b < n_bins - 1 else probs <= hi)
        centers.append((lo + hi) / 2)
        if m.sum() == 0:
            accs.append(np.nan)
            confs.append(np.nan)
            continue
        acc = labels[m].mean()
        conf = probs[m].mean()
        accs.append(acc)
        confs.append(conf)
        ece += (m.sum() / len(probs)) * abs(acc - conf)
    return np.array(centers), np.array(accs), np.array(confs), float(ece)


def evaluate_predictor(scores, true_div, epsilon, probs=None):
    """Full metric bundle for one predictor against one label column.
    scores: higher = more divergence. true_div: continuous ground-truth divergence.
    epsilon: threshold for the binary task. probs: optional calibrated [0,1] for ECE."""
    labels = (np.asarray(true_div) > epsilon).astype(int)
    out = {"auroc": auroc(scores, labels),
           "spearman": spearman(scores, true_div),
           "pos_rate": float(labels.mean())}
    if probs is not None:
        _, _, _, ece = calibration(probs, labels)
        out["ece"] = ece
    return out
