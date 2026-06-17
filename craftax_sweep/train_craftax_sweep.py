#!/usr/bin/env python3
"""Capacity-sweep training driver for the EMERALD-JAX Craftax-Classic arms.

Trains ONE arm (full or tiny world model) to a target env-step count, optimised to
keep a single GPU loaded, while logging everything we need for the LATER divergence /
learned-trust eval:

  * milestone checkpoints  ckpts/<arm>/step_<N>.pkl   (kept, every --milestone-every)
  * rolling latest         ckpts/<arm>/latest.pkl     (overwritten, for --resume)
  * training metrics        ckpts/<arm>/metrics.jsonl  (every --log-every env-steps)
      total/model/image/kl_prior/kl_post/kl_mask/reward/cont/actor/value loss,
      imag_reward_mean, returns_mean, policy_ent, perc_low/high, env_s, grad_s
  * eval metrics            ckpts/<arm>/eval.jsonl     (every --eval-every env-steps)
      Crafter score, num_episodes, mean_step_reward, per-achievement rates

The imagined-rollout fidelity metrics (policy-divergence over the horizon, reward MAE,
decoded-pixel MSE, token accuracy vs depth) are intentionally NOT computed here — they
are fully reconstructable offline from the milestone checkpoints via harness/critic/,
which already reads this exact pickle format. Keep the milestones and nothing is lost.

Run from the repo root (or anywhere — repo root is added to sys.path):
    python scripts/train_craftax_sweep.py --arm full --total 10_000_000 \
        --ckpt-root /workspace/ckpts --resume
    python scripts/train_craftax_sweep.py --arm tiny --total 10_000_000 \
        --ckpt-root /workspace/ckpts --resume

Keep --gpc and any --num-envs/--batch-size/--capacity overrides IDENTICAL across the
full and tiny arms — the sweep is a controlled comparison; only WM capacity should differ.
"""

import argparse
import json
import os
import sys
import time

# repo root on sys.path so `import emerald_jax` works no matter the cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402

from emerald_jax import config as cfgmod  # noqa: E402
from emerald_jax import replay  # noqa: E402
from emerald_jax import train  # noqa: E402

ARMS = {"full": cfgmod.craftax_fast, "tiny": cfgmod.craftax_fast_tiny_wm}

_LOG_KEYS = ("total_loss", "model_loss", "image_loss", "kl_prior", "kl_post",
             "kl_mask", "reward_loss", "cont_loss", "actor_loss", "value_loss",
             "imag_reward_mean", "returns_mean", "policy_ent", "perc_low", "perc_high")


def _append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def save_milestone(path, st):
    """Params-only checkpoint for the later divergence/eval pass — no optimizer state
    (that lives in latest.pkl for --resume). Same {"params", "cfg", ...} shape the
    harness/critic EmeraldJaxAdapter reads; rebuild the agent from blob["cfg"]."""
    import pickle
    blob = {"params": st["params"], "cfg": st["cfg"],
            "env_step": st["env_step"], "grad_step": st["grad_step"]}
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(blob), f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--total", type=int, default=10_000_000, help="target env-steps")
    p.add_argument("--ckpt-root", default="ckpts", help="per-arm dir is <root>/<arm>")
    p.add_argument("--collect-steps", type=int, default=64)
    p.add_argument("--gpc", type=int, default=1,
                   help="grad steps per collect (KEEP IDENTICAL across arms)")
    p.add_argument("--milestone-every", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--log-every", type=int, default=20_000)
    p.add_argument("--eval-steps", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    # GPU-saturation overrides (apply to BOTH arms equally if used)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--capacity", type=int, default=None,
                   help="replay rows; keep capacity*num_envs ~constant when tuning")
    args = p.parse_args()

    cfg = ARMS[args.arm]()
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.capacity is not None:
        cfg.capacity = args.capacity

    ckpt_dir = os.path.join(args.ckpt_root, args.arm)
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, "latest.pkl")
    metrics_path = os.path.join(ckpt_dir, "metrics.jsonl")
    eval_path = os.path.join(ckpt_dir, "eval.jsonl")

    print(f"[devices] {jax.devices()}", flush=True)
    print(f"[arm={args.arm}] dim_model={cfg.dim_model} dim_cnn={cfg.dim_cnn} "
          f"blocks_trans={cfg.num_blocks_trans} heads={cfg.num_heads_trans} "
          f"blocks_mask={cfg.num_blocks_mask} reduced={cfg.reduced_channels} | "
          f"num_envs={cfg.num_envs} batch={cfg.batch_size} cap={cfg.capacity} gpc={args.gpc}",
          flush=True)

    st = train.init_state(cfg, args.seed)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(st["params"]))
    print(f"[arm={args.arm}] total params: {n_params/1e6:.2f}M", flush=True)

    rollout = train.make_rollout(st["agent"], st["env"], st["eparams"],
                                 st["ach_keys"], st["A"])
    train_step = train.make_train_step(st["agent"], st["tx"], cfg)

    if args.resume and os.path.exists(latest):
        st = train.load_ckpt(latest, st)
        print(f"[resume] from env_step {st['env_step']}", flush=True)

    def collect(sample):
        (st["obs"], st["estate"], st["rings"], st["key"]), outs = rollout(
            st["params"], st["obs"], st["estate"], st["rings"], st["key"],
            args.collect_steps, sample)
        img, a_int, reward, done, _ = outs
        st["buf"] = replay.add_rollout(st["buf"], img, a_int, reward, done)
        st["env_step"] += args.collect_steps * cfg.num_envs

    # Fill the replay buffer to a healthy level before any grad steps. The buffer is
    # NOT checkpointed (it's ~GBs and regenerable), so this ALSO runs on --resume to
    # rebuild it from the restored policy. Gating on buffer FILL (not env_step) makes
    # the fresh-start and resume paths identical and removes the degenerate thin-buffer
    # transient that an env_step-gated prefill would skip on resume.
    warmup_rows = int(min(cfg.capacity, max(cfg.prefill // cfg.num_envs, 8 * cfg.L)))
    if int(st["buf"].size) < warmup_rows:
        print(f"[warmup] filling replay buffer to {warmup_rows} rows "
              f"(have {int(st['buf'].size)})...", flush=True)
        while int(st["buf"].size) < warmup_rows:
            collect(True)

    n_grad = max(1, int(args.gpc * args.collect_steps))
    # align milestone/eval/log to the current step so --resume doesn't refire them
    next_milestone = ((st["env_step"] // args.milestone_every) + 1) * args.milestone_every
    last_eval = last_log = st["env_step"]
    t0 = time.time()
    es0, gs0 = st["env_step"], st["grad_step"]
    print(f"[train] arm={args.arm} target {args.total} env-steps "
          f"({cfg.num_envs} envs, {n_grad} grad/iter)", flush=True)

    while st["env_step"] < args.total:
        collect(True)
        metrics = None
        for _ in range(n_grad):
            st["key"], bk, tk = jax.random.split(st["key"], 3)
            batch = replay.sample(st["buf"], bk, cfg.batch_size, cfg.L, st["A"])
            st["params"], st["opt_state"], st["perc"], metrics = train_step(
                st["params"], st["opt_state"], st["perc"], batch, tk)
            st["grad_step"] += 1

        if st["env_step"] - last_log >= args.log_every:
            dt = time.time() - t0
            eps = (st["env_step"] - es0) / dt
            gps = (st["grad_step"] - gs0) / dt
            m = {k: float(metrics[k]) for k in _LOG_KEYS}
            row = {"env_step": int(st["env_step"]), "grad_step": int(st["grad_step"]),
                   "env_s": round(eps, 1), "grad_s": round(gps, 2),
                   "wall_s": round(dt, 1), **m}
            _append_jsonl(metrics_path, row)
            print(f"[{st['env_step']:>9}] env/s {eps:7.0f} grad/s {gps:5.1f} | "
                  f"loss {m['total_loss']:8.2f} img {m['image_loss']:7.2f} "
                  f"klpr {m['kl_prior']:4.2f} rew {m['reward_loss']:4.2f} "
                  f"act {m['actor_loss']:+.3f} val {m['value_loss']:5.2f}", flush=True)
            last_log = st["env_step"]

        if st["env_step"] - last_eval >= args.eval_every:
            ev = train.evaluate(st, rollout, args.eval_steps)
            _append_jsonl(eval_path, {"env_step": int(st["env_step"]),
                                      "grad_step": int(st["grad_step"]), **ev})
            print(f"[eval @ {st['env_step']}] score {ev['score']:.2f} "
                  f"({ev['num_episodes']} eps) reward/step {ev['mean_step_reward']:.4f}",
                  flush=True)
            train.save_ckpt(latest, st)          # rolling, for resume
            last_eval = st["env_step"]

        if st["env_step"] >= next_milestone:
            mpath = os.path.join(ckpt_dir, f"step_{st['env_step']}.pkl")
            save_milestone(mpath, st)
            print(f"[milestone] saved {mpath}", flush=True)
            next_milestone += args.milestone_every

    train.save_ckpt(latest, st)
    final_milestone = os.path.join(ckpt_dir, f"step_{st['env_step']}.pkl")
    save_milestone(final_milestone, st)
    ev = train.evaluate(st, rollout, args.eval_steps)
    _append_jsonl(eval_path, {"env_step": int(st["env_step"]),
                              "grad_step": int(st["grad_step"]), "final": True, **ev})
    print(f"[done] arm={args.arm} {st['env_step']} env-steps | "
          f"final score {ev['score']:.2f} | ckpt {final_milestone}", flush=True)


if __name__ == "__main__":
    main()
