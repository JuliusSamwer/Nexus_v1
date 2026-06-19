#!/usr/bin/env python3
"""Decision-relevant horizon curves for an EMERALD-JAX world model (pretrained or finetuned).

Collects on-policy real rollouts, seeds the WM at t0, imagines H steps with the REAL actions
(teacher-forced), and at each horizon k compares imagined vs real posterior:
  pol_agree[k] : % argmax policy(s_hat_k) == argmax policy(s_true_k)   [DECISION-relevant]
  val_div[k]   : |value(s_hat_k) - value(s_true_k)|                    [DECISION-relevant]
  tok_acc[k]   : % imagined stoch tokens == real posterior tokens       [general; misleading]
Reports the curves, the 'usable decision horizon' (where pol_agree crosses 50%), and the
greedy Crafter score (the guardrail). This is the before/after metric for Stage-2 finetuning.

  python stage2_finetune/decision_eval.py --checkpoint <ckpt.pkl> --n_rollouts 64 --label full-10M
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
from emerald_jax.dists import SymLogDiscreteDist


def load_ckpt(path):
    blob = pickle.load(open(path, "rb"))
    return blob["cfg"], {"params": blob["params"]["params"]}


def _imagine_tf(agent, params, seed_stoch, seed_deter, actions, key):
    """Teacher-forced fixed-buffer imagine. seed_* (N,1,...), actions (N,H,A) -> (N,H+1,...)."""
    def fn(m, s0, d0, acts):
        c = m.cfg
        B, H = s0.shape[0], acts.shape[1]
        s, d = s0, d0
        xbuf = jnp.zeros((B, H, c.dim_model))
        stochs, deters = [s], [d]
        for h in range(H):
            xbuf = xbuf.at[:, h:h + 1].set(m.tssm.mix(m.tssm.encode_stoch(s), acts[:, h:h + 1]))
            d = m.tssm.transformer(xbuf, causal=True)[:, h:h + 1]
            _, s_oh = m.tssm.mask_network.sample(m.tssm.deter_to_dec(d), c.num_decoding_steps)
            s = s_oh.reshape(B, 1, 4, 4, -1)
            stochs.append(s); deters.append(d)
        return {"stoch": jnp.concatenate(stochs, 1), "deter": jnp.concatenate(deters, 1)}
    return agent.apply(params, seed_stoch, seed_deter, actions, method=fn,
                       rngs={k: key for k in ("sample", "mask", "order")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_rollouts", type=int, default=64)
    ap.add_argument("--H", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, params = load_ckpt(args.checkpoint)
    A = cenv.NUM_ACTIONS
    agent = model.EmeraldAgent(cfg, A)
    env, eparams = cenv.make_env(auto_reset=True)
    SV = cfg.stoch_size * cfg.discrete
    key = jax.random.PRNGKey(args.seed)

    # ---- collect on-policy real rollouts (one per env) ----
    T = args.warmup + args.H + 1
    N = args.n_rollouts
    key, rk = jax.random.split(key)
    obs, estate = cenv.reset(env, eparams, jax.random.split(rk, N))
    rings = (jnp.zeros((N, cfg.att_context_left, 4, 4, SV)),
             jnp.zeros((N, cfg.att_context_left, A)))
    # build a temporary state so we can reuse train.make_rollout's policy collection
    st = {"agent": agent, "params": params, "A": A}
    rollout = train.make_rollout(agent, env, eparams,
                                 [f"Achievements/{n}" for n in cenv.ach_names(
                                     cenv.step(env, eparams, jax.random.split(rk, N), estate,
                                               jnp.zeros((N,), jnp.int32))[4])], A)
    (_, _, _, _), outs = rollout(params, obs, estate, rings, key, T, False)  # greedy on-policy
    imgs, a_int, reward, done, _ = outs                       # imgs (T,N,3,64,64), a_int (T,N)

    images = jnp.transpose(imgs, (1, 0, 2, 3, 4))             # (N,T,3,64,64)
    a_int = jnp.transpose(a_int, (1, 0))                      # (N,T)
    # observe wants action[i] = action INTO state i; collected a_int[t] is action FROM state t,
    # so shift right with a 0 at the front.
    act_into = jnp.concatenate([jnp.zeros((N, 1), jnp.int32), a_int[:, :-1]], 1)
    act_into_oh = jax.nn.one_hot(act_into, A)
    a_from_oh = jax.nn.one_hot(a_int, A)

    key, k1 = jax.random.split(key)
    rngs = {k: k1 for k in ("sample", "mask", "order")}
    enc = agent.apply(params, images, method=lambda m, im: m.encoder(im), rngs=rngs)
    post, _ = agent.apply(params, enc["stoch"], act_into_oh,
                          method=lambda m, s, a: m.tssm.observe(s, a), rngs=rngs)

    t0 = args.warmup
    seed_stoch = post["stoch"][:, t0:t0 + 1]
    seed_deter = post["deter"][:, t0:t0 + 1]
    fut = a_from_oh[:, t0:t0 + args.H]                        # actions FROM the seed onward
    key, k2 = jax.random.split(key)
    imag = _imagine_tf(agent, params, seed_stoch, seed_deter, fut, k2)

    # ---- per-horizon decision-relevant metrics ----
    def pol_argmax(feat):
        lg = agent.apply(params, feat, method=lambda m, f: m.policy_head(f), rngs=rngs)
        return jnp.argmax(lg, -1)

    def val(feat):
        lg = agent.apply(params, feat, method=lambda m, f: m.value_head(f), rngs=rngs)
        return SymLogDiscreteDist(lg).mode()[..., 0]

    pol_agree, val_div, tok_acc = [], [], []
    real_logits = enc["logits"]                              # (N,T,4,4,S,V) posterior tokens
    for k in range(1, args.H + 1):
        rf = (post["stoch"][:, t0 + k:t0 + k + 1], post["deter"][:, t0 + k:t0 + k + 1])
        hf = (imag["stoch"][:, k:k + 1], imag["deter"][:, k:k + 1])
        pol_agree.append(100.0 * float((pol_argmax(hf) == pol_argmax(rf)).mean()))
        val_div.append(float(jnp.abs(val(hf) - val(rf)).mean()))
        real_tok = jnp.argmax(real_logits[:, t0 + k], -1)               # (N,4,4,S)
        hat_tok = jnp.argmax(imag["stoch"][:, k].reshape(N, 4, 4, cfg.stoch_size, cfg.discrete), -1)
        tok_acc.append(100.0 * float((hat_tok == real_tok).mean()))

    # usable decision horizon: last k with pol_agree >= 50 before first crossing
    usable = 0
    for k, p in enumerate(pol_agree, 1):
        if p >= 50.0:
            usable = k
        else:
            break

    # ---- guardrail: greedy Crafter score ----
    st_full = train.init_state(cfg, args.seed)
    st_full["params"] = params
    ev = train.evaluate(st_full, rollout, args.eval_steps)

    lbl = args.label or os.path.basename(args.checkpoint)
    print(f"\n================ DECISION-RELEVANT HORIZON — {lbl} ================")
    print(f"  Crafter score (greedy): {ev['score']:.2f}   [guardrail]")
    print(f"  usable decision horizon (pol_agree>=50%): {usable}/{args.H} steps")
    print(f"  {'k':>3} {'pol_agree%':>11} {'val_div':>9} {'tok_acc%':>9}")
    for k in range(args.H):
        print(f"  {k+1:>3} {pol_agree[k]:>11.1f} {val_div[k]:>9.3f} {tok_acc[k]:>9.1f}")
    print("=" * 60)
    return {"label": lbl, "score": ev["score"], "usable_horizon": usable,
            "pol_agree": pol_agree, "val_div": val_div, "tok_acc": tok_acc}


if __name__ == "__main__":
    main()
