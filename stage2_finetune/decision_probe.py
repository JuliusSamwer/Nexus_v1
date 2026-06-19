#!/usr/bin/env python3
"""Latent probe: does a world model's posterior latent linearly decode control-relevant
targets? Exploitation-FREE test of whether a finetuned WM 'holds more control-relevant
information' than a baseline (no policy training, so no exploitation confound).

Protocol: collect N trajectories with a FIXED policy (--collect_ckpt, same seed -> identical
trajectories across runs), then encode the SAME real obs through the WM under test
(--checkpoint) -> posterior deter latent, and ridge-probe it for ground-truth targets from
the real trajectory:
  - return_to_go : discounted sum of future real rewards (the value-relevant target)
  - reward       : immediate real reward
Reports held-out R2. Run on the improved WM and the baseline WM with the SAME --collect_ckpt;
HIGHER R2 on improved = it holds more control-relevant info (the clean proof). LOWER = Goodhart.

  python stage2_finetune/decision_probe.py --checkpoint <WM.pkl> --collect_ckpt <baseline.pkl> --label improved
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from emerald_jax import env as cenv
from emerald_jax import model, train


def load_ckpt(path):
    blob = pickle.load(open(path, "rb"))
    return blob["cfg"], {"params": blob["params"]["params"]}


def ridge_r2(X, y, lam=10.0, split=0.7, seed=0):
    """Held-out R2 of a ridge probe X->y. Standardises features, shuffles, train/test split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    ntr = int(X.shape[0] * split)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    X = (X - mu) / sd
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    ym = ytr.mean()
    D = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(D), Xtr.T @ (ytr - ym))
    pred = Xte @ w + ym
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-9
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="WM whose latent we probe")
    ap.add_argument("--collect_ckpt", default=None, help="fixed policy for trajectories (default=checkpoint)")
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    coll_path = args.collect_ckpt or args.checkpoint

    A = cenv.NUM_ACTIONS
    env, eparams = cenv.make_env(auto_reset=True)

    # collection policy (FIXED across probe runs)
    ccfg, cparams = load_ckpt(coll_path)
    cagent = model.EmeraldAgent(ccfg, A)
    rollout = train.make_rollout(cagent, env, eparams,
                                 [f"Achievements/{n}" for n in cenv.ach_names(
                                     cenv.step(env, eparams, jax.random.split(jax.random.PRNGKey(0), args.n_envs),
                                               cenv.reset(env, eparams, jax.random.split(jax.random.PRNGKey(0), args.n_envs))[1],
                                               jnp.zeros((args.n_envs,), jnp.int32))[4])], A)
    SVc = ccfg.stoch_size * ccfg.discrete
    key = jax.random.PRNGKey(args.seed)
    key, rk = jax.random.split(key)
    obs, estate = cenv.reset(env, eparams, jax.random.split(rk, args.n_envs))
    rings = (jnp.zeros((args.n_envs, ccfg.att_context_left, 4, 4, SVc)),
             jnp.zeros((args.n_envs, ccfg.att_context_left, A)))
    (_, _, _, _), outs = rollout(cparams, obs, estate, rings, key, args.steps, True)
    imgs, a_int, reward, done, _ = outs                      # (T,N,...)

    # ground-truth control targets from the real trajectory
    rew = np.asarray(reward); dn = np.asarray(done).astype(np.float32)   # (T,N)
    T, N = rew.shape
    gamma = float(ccfg.gamma)
    rtg = np.zeros_like(rew); run = np.zeros(N)
    for t in range(T - 1, -1, -1):
        run = rew[t] + gamma * run * (1 - dn[t])
        rtg[t] = run

    # encode the SAME obs through the WM under test -> posterior deter latent
    pcfg, pparams = load_ckpt(args.checkpoint)
    pagent = model.EmeraldAgent(pcfg, A)
    images = jnp.transpose(imgs, (1, 0, 2, 3, 4))            # (N,T,3,64,64)
    ai = jnp.transpose(a_int, (1, 0))
    act_into = jax.nn.one_hot(jnp.concatenate([jnp.zeros((N, 1), jnp.int32), ai[:, :-1]], 1), A)
    key, k1 = jax.random.split(key)
    rngs = {k: k1 for k in ("sample", "mask", "order")}
    enc = pagent.apply(pparams, images, method=lambda m, im: m.encoder(im), rngs=rngs)
    post, _ = pagent.apply(pparams, enc["stoch"], act_into,
                           method=lambda m, s, a: m.tssm.observe(s, a), rngs=rngs)
    deter = np.asarray(post["deter"]).reshape(N * T, -1)     # (N*T, dim_model)
    rtg_flat = rtg.T.reshape(-1)
    rew_flat = rew.T.reshape(-1)

    r2_rtg = ridge_r2(deter, rtg_flat, seed=args.seed)
    r2_rew = ridge_r2(deter, rew_flat, seed=args.seed)
    lbl = args.label or os.path.basename(args.checkpoint)
    print(f"\n========= LATENT PROBE — {lbl} =========")
    print(f"  samples: {N*T}   deter dim: {deter.shape[1]}   collect policy: {os.path.basename(coll_path)}")
    print(f"  R2(return_to_go) : {r2_rtg:.4f}   <- control/value-relevant info")
    print(f"  R2(reward)       : {r2_rew:.4f}")
    print("=" * 42)


if __name__ == "__main__":
    main()
