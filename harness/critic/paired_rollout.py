"""Generate the paired-rollout dataset: loop seeds through a WMAdapter, build the latent
k-NN reference bank across all rollouts, attach k-NN density, save one .npz.

Each row is one imagined step; `seed` tags the originating rollout so train/val/test are
seed-DISJOINT (no leakage from sharing a trajectory across the split).
"""

import numpy as np


def generate(adapter, out_path, n_rollouts=200, H=15, mixture_p=0.5, K=4, warmup=8,
             knn_k=8, seed0=10_000, verbose=True):
    cols = ["feat_prev", "feat_cur", "action", "horizon",
            "L1", "L2", "L3", "L4", "L4b", "entropy", "disagreement", "ref_latent"]
    acc = {c: [] for c in cols}
    seeds = []
    rng_master = np.random.default_rng(seed0)
    got = 0
    s = seed0
    while got < n_rollouts:
        rng = np.random.default_rng(s)
        rec = adapter.generate_rollout(H, mixture_p, K, warmup, s, rng)
        s += 1
        if rec is None or len(rec["horizon"]) == 0:
            continue
        for c in cols:
            acc[c].append(rec[c])
        seeds.append(np.full(len(rec["horizon"]), got, np.int64))
        got += 1
        if verbose and got % 25 == 0:
            print(f"[paired] {got}/{n_rollouts} rollouts", flush=True)
    data = {c: np.concatenate(acc[c], 0) for c in cols}
    data["seed"] = np.concatenate(seeds, 0)
    # k-NN density of each imagined feat_cur vs the bank of real posterior latents
    bank = data["ref_latent"]
    data["knn_density"] = _knn_density(data["feat_cur"], bank, knn_k)
    data["horizon_norm"] = (data["horizon"] / max(1.0, data["horizon"].max())).astype(np.float32)
    np.savez_compressed(out_path, **data)
    if verbose:
        print(f"[paired] saved {out_path}  ({len(data['seed'])} steps, "
              f"{got} rollouts)", flush=True)
    return out_path


def _knn_density(query, bank, k):
    """Mean distance to the k nearest bank latents (higher = more OOD). Chunked L2."""
    q = query.astype(np.float32)
    b = bank.astype(np.float32)
    k = min(k, len(b))
    out = np.empty(len(q), np.float32)
    bn = (b ** 2).sum(1)
    for i in range(0, len(q), 1024):
        qc = q[i:i + 1024]
        d2 = (qc ** 2).sum(1, keepdims=True) - 2 * qc @ b.T + bn[None]
        d2 = np.clip(d2, 0, None)
        part = np.partition(d2, k - 1, axis=1)[:, :k]
        out[i:i + len(qc)] = np.sqrt(part).mean(1)
    return out
