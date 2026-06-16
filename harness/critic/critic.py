"""The conditional critic C(s_t, a_t, ŝ_{t+1}) and its ablations, in JAX/optax.

Operates on the per-step feature arrays dumped by paired_rollout (substrate-independent):
  feat_prev  (N, D)  latent rep of context s_{t-1}
  feat_cur   (N, D)  latent rep of imagined ŝ_t   <-- the "ŝ_{t+1}" the critic judges
  action     (N, A)  one-hot a_{t-1}
  horizon    (N, 1)  normalized step index

Feature sets (ablations by masking):
  conditional : [feat_prev, action, feat_cur, horizon]   (the full C)
  marginal    : [feat_cur]                                (no context / no action)
  horizon     : [horizon]                                 (trivial baseline)

A small MLP -> single logit, trained as binary (d > epsilon) with BCE. The logit is the
divergence SCORE (monotone -> valid for AUROC & Spearman vs continuous d); sigmoid(logit)
is the calibrated probability. C stays differentiable wrt feat_cur — `grad_wrt_shat`
returns ∇C, the actionable signal an ensemble can't give (the paper's leg-2 hook).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

FEATURE_SETS = {
    "conditional": ("feat_prev", "action", "feat_cur", "horizon"),
    "marginal": ("feat_cur",),
    "horizon": ("horizon",),
}


def build_features(data, which):
    cols = [np.asarray(data[k], np.float32) for k in FEATURE_SETS[which]]
    cols = [c.reshape(c.shape[0], -1) for c in cols]
    return np.concatenate(cols, axis=1)


def _init(key, din, hidden):
    ks = jax.random.split(key, len(hidden) + 1)
    params, d = [], din
    for i, h in enumerate(hidden + [1]):
        scale = (2.0 / d) ** 0.5
        params.append((jax.random.normal(ks[i], (d, h)) * scale, jnp.zeros(h)))
        d = h
    return params


def _forward(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x[..., 0]                                  # logit


def train_critic(X, y_bin, seeds, hidden=(128, 128), steps=3000, lr=1e-3,
                 val_frac_seeds=0.2, seed=0):
    """Train the MLP on a SEED-disjoint split. X (N,D), y_bin (N,) in {0,1}, seeds (N,).
    Returns (params, score_fn, val_idx, test_is_external). Early-stops on val AUROC."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seeds)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac_seeds))
    val_seeds = set(uniq[:n_val].tolist())
    tr = np.array([s not in val_seeds for s in seeds])
    va = ~tr
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xtr, ytr = jnp.asarray(Xn[tr]), jnp.asarray(y_bin[tr].astype(np.float32))
    Xva, yva = jnp.asarray(Xn[va]), y_bin[va].astype(np.float32)

    params = _init(jax.random.PRNGKey(seed), X.shape[1], list(hidden))
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(p, xb, yb):
        logit = _forward(p, xb)
        return optax.sigmoid_binary_cross_entropy(logit, yb).mean()

    @jax.jit
    def step(p, os, xb, yb):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        upd, os = opt.update(g, os)
        return optax.apply_updates(p, upd), os, l

    from .metrics import auroc
    best, best_params, patience, bad = -1, params, 12, 0
    bs = min(4096, Xtr.shape[0])
    key = np.random.default_rng(seed + 1)
    for it in range(steps):
        idx = key.integers(0, Xtr.shape[0], size=bs)
        params, opt_state, _ = step(params, opt_state, Xtr[idx], ytr[idx])
        if (it + 1) % 100 == 0:
            sc = np.asarray(_forward(params, Xva))
            a = auroc(sc, yva.astype(int))
            if a > best:
                best, best_params, bad = a, params, 0
            else:
                bad += 1
                if bad >= patience:
                    break

    def score_fn(Xraw):
        return np.asarray(_forward(best_params, jnp.asarray((Xraw - mu) / sd)))

    def prob_fn(Xraw):
        return np.asarray(jax.nn.sigmoid(_forward(best_params, jnp.asarray((Xraw - mu) / sd))))

    return {"params": best_params, "mu": mu, "sd": sd, "score": score_fn,
            "prob": prob_fn, "val_seeds": val_seeds}


def grad_wrt_shat(trained, x_row, shat_slice):
    """∇C wrt the ŝ portion of a single feature row — the differentiable correction
    signal (leg 2). shat_slice = (start,end) column indices of feat_cur in X.
    Returns the gradient vector (end-start,). Demonstrates C exposes ∇C; an ensemble
    disagreement scalar does not."""
    mu, sd, p = trained["mu"], trained["sd"], trained["params"]
    s, e = shat_slice
    xr = jnp.asarray((x_row - mu) / sd)

    def c(xv):
        return _forward(p, xv[None])[0]
    g = jax.grad(c)(xr)
    return np.asarray(g[s:e] / sd[s:e])
