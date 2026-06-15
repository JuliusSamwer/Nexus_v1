"""JAX EMERALD training loop on Craftax-Classic.

The whole hot path is jit-compiled and runs on-device:
  * collection: lax.scan the CURRENT actor over the vmapped Craftax envs, carrying a
    per-env latent history (last att_context_left stochs+actions), reset on episode end;
  * train_step: the proven value_and_grad over EmeraldAgent.compute_losses with
    optax.multi_transform (wm/actor/critic, value_target frozen) + value_target EMA;
  * eval: a greedy rollout that reads Craftax's Achievements/* and computes the
    Crafter score.

No Python env loop -> on a GPU this is where the speedup over CPU-bound Crafter lives.
Throughput (env-steps/s, grad-steps/s) is printed each log interval.

Usage (e.g. from the Colab notebook):
    from emerald_jax import config, train
    train.run(config.craftax(), total_env_steps=1_000_000, ckpt_dir="/content/drive/...")

Online-acting context handling is the documented EMERALD approximation (fixed-ctx
zero-padded history); the world-model loss path (observe) is exact.
"""

import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax

from . import env as cenv
from . import model
from . import replay

_GROUP = {"encoder": "wm", "decoder": "wm", "tssm": "wm", "reward_head": "wm",
          "continue_head": "wm", "value_head": "critic", "policy_head": "actor",
          "value_target": "frozen"}


# --------------------------------------------------------------------------- #
# Optimizer (per-group lr + clip, value_target frozen) — matches the smoke
# --------------------------------------------------------------------------- #
def make_optimizer(cfg, params):
    labels = {"params": {k: jax.tree_util.tree_map(lambda _: _GROUP[k], v)
                         for k, v in params["params"].items()}}
    mk = lambda lr, eps, clip: optax.chain(
        optax.clip_by_global_norm(clip), optax.adam(lr, eps=eps))
    tx = optax.multi_transform(
        {"wm": mk(cfg.model_lr, cfg.model_eps, cfg.model_grad_max_norm),
         "actor": mk(cfg.actor_lr, cfg.actor_eps, cfg.actor_grad_max_norm),
         "critic": mk(cfg.value_lr, cfg.value_eps, cfg.value_grad_max_norm),
         "frozen": optax.set_to_zero()}, labels)
    return tx, tx.init(params)


# --------------------------------------------------------------------------- #
# Init
# --------------------------------------------------------------------------- #
def init_state(cfg, seed=0):
    env, eparams = cenv.make_env(auto_reset=True)
    A = cenv.NUM_ACTIONS
    agent = model.EmeraldAgent(cfg, A)
    key = jax.random.PRNGKey(seed)
    key, ik = jax.random.split(key)

    SV = cfg.stoch_size * cfg.discrete
    B, L = cfg.batch_size, cfg.L
    dummy = {"image": jnp.zeros((B, L, 3, cfg.image_size, cfg.image_size)),
             "action": jnp.zeros((B, L, A)), "reward": jnp.zeros((B, L)),
             "cont": jnp.ones((B, L))}
    rngs = dict(zip(("params", "sample", "mask", "order"), jax.random.split(ik, 4)))
    params = agent.init(rngs, dummy, 0.0, 0.0, method=agent.compute_losses)
    # sync slow critic to critic at init
    params = {"params": {**params["params"],
                         "value_target": params["params"]["value_head"]}}

    tx, opt_state = make_optimizer(cfg, params)

    # reset envs + empty history rings
    key, rk = jax.random.split(key)
    obs, estate = cenv.reset(env, eparams, jax.random.split(rk, cfg.num_envs))
    ctx = cfg.att_context_left
    rings = (jnp.zeros((cfg.num_envs, ctx, 4, 4, SV)),
             jnp.zeros((cfg.num_envs, ctx, A)))
    buf = replay.init_buffer(cfg.capacity, cfg.num_envs, cfg.image_size)

    # achievement key list (static, sorted) from one probe step
    _, _, _, _, info = cenv.step(env, eparams, jax.random.split(rk, cfg.num_envs),
                                 estate, jnp.zeros((cfg.num_envs,), jnp.int32))
    ach_keys = [f"Achievements/{n}" for n in cenv.ach_names(info)]

    return dict(cfg=cfg, env=env, eparams=eparams, agent=agent, A=A, params=params,
                tx=tx, opt_state=opt_state, buf=buf, obs=obs, estate=estate,
                rings=rings, perc=(jnp.float32(0.0), jnp.float32(0.0)),
                key=key, env_step=0, grad_step=0, ach_keys=ach_keys)


# --------------------------------------------------------------------------- #
# Rollout (collection / eval share this)
# --------------------------------------------------------------------------- #
def make_rollout(agent, env, eparams, ach_keys, num_actions):
    def stack_ach(info):
        return jnp.stack([info[k] for k in ach_keys], axis=-1)        # (N, 22)

    @partial(jax.jit, static_argnames=("num_steps", "sample"))
    def rollout(params, obs, estate, rings, key, num_steps, sample):
        sr0, ar0 = rings

        def step(carry, _):
            obs, estate, sr, ar, key = carry
            key, ka, ks = jax.random.split(key, 3)
            a_oh, stoch_t = agent.apply(params, obs[:, None], sr, ar, sample,
                                        method=agent.act, rngs={"sample": ka})
            a_int = jnp.argmax(a_oh, -1).astype(jnp.int32)
            keys = jax.random.split(ks, obs.shape[0])
            nobs, estate, reward, done, info = cenv.step(
                env, eparams, keys, estate, a_int)
            sr2 = jnp.concatenate([sr[:, 1:], stoch_t[:, None]], axis=1)
            ar2 = jnp.concatenate([ar[:, 1:], a_oh[:, None]], axis=1)
            sr2 = jnp.where(done[:, None, None, None, None], 0.0, sr2)
            ar2 = jnp.where(done[:, None, None], 0.0, ar2)
            out = (obs, a_int, reward, done, stack_ach(info))
            return (nobs, estate, sr2, ar2, key), out

        carry, outs = jax.lax.scan(
            step, (obs, estate, sr0, ar0, key), None, length=num_steps)
        nobs, estate, sr, ar, key = carry
        return (nobs, estate, (sr, ar), key), outs
    return rollout


# --------------------------------------------------------------------------- #
# Train step (the proven pattern)
# --------------------------------------------------------------------------- #
def make_train_step(agent, tx, cfg):
    def loss_fn(params, batch, perc, rk):
        ks = jax.random.split(rk, 3)
        return agent.apply(params, batch, perc[0], perc[1],
                           rngs={"sample": ks[0], "mask": ks[1], "order": ks[2]},
                           method=agent.compute_losses)

    @jax.jit
    def train_step(params, opt_state, perc, batch, rk):
        (total, (metrics, perc_new)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, batch, perc, rk)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        d = cfg.critic_ema_decay
        vt = jax.tree_util.tree_map(
            lambda t, h: (1 - d) * t + d * h,
            params["params"]["value_target"], params["params"]["value_head"])
        params = {"params": {**params["params"], "value_target": vt}}
        return params, opt_state, perc_new, metrics
    return train_step


# --------------------------------------------------------------------------- #
# Eval — Crafter score from achievement success rates
# --------------------------------------------------------------------------- #
def crafter_score(rates):
    # geometric mean of (1 + success%) minus 1, per Crafter (Hafner 2021)
    pct = rates * 100.0
    return float(jnp.exp(jnp.mean(jnp.log(1.0 + pct))) - 1.0)


def evaluate(st, rollout, steps=1000, seed=12345):
    cfg = st["cfg"]
    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    obs, estate = cenv.reset(st["env"], st["eparams"],
                             jax.random.split(rk, cfg.num_envs))
    SV = cfg.stoch_size * cfg.discrete
    rings = (jnp.zeros((cfg.num_envs, cfg.att_context_left, 4, 4, SV)),
             jnp.zeros((cfg.num_envs, cfg.att_context_left, st["A"])))
    (_, _, _, _), outs = rollout(st["params"], obs, estate, rings, key, steps, False)
    _, _, reward, done, ach = outs                  # ach (T,N,22), done (T,N)
    done = done.reshape(-1)
    ach = ach.reshape(-1, ach.shape[-1])
    completed = jnp.where(done[:, None], ach, 0.0).sum(0)
    n_ep = jnp.maximum(done.sum(), 1)
    rates = completed / n_ep
    names = [k.split("/", 1)[1] for k in st["ach_keys"]]
    return {"score": crafter_score(rates), "num_episodes": int(done.sum()),
            "achievement_rates": {n: float(r) for n, r in zip(names, rates)},
            "mean_step_reward": float(reward.mean())}


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def save_ckpt(path, st):
    blob = {"params": st["params"], "opt_state": st["opt_state"],
            "perc": st["perc"], "env_step": st["env_step"],
            "grad_step": st["grad_step"], "cfg": st["cfg"]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(blob), f)


def load_ckpt(path, st):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    st["params"] = blob["params"]
    st["opt_state"] = blob["opt_state"]
    st["perc"] = blob["perc"]
    st["env_step"] = blob["env_step"]
    st["grad_step"] = blob["grad_step"]
    return st


def run(cfg, total_env_steps=1_000_000, collect_steps=16, grad_per_collect=1,
        eval_every_env=50_000, log_every_env=10_000, eval_steps=1000,
        ckpt_dir=None, seed=0, resume=False):
    st = init_state(cfg, seed)
    rollout = make_rollout(st["agent"], st["env"], st["eparams"], st["ach_keys"], st["A"])
    train_step = make_train_step(st["agent"], st["tx"], cfg)
    ckpt_path = os.path.join(ckpt_dir, "emerald_jax.pkl") if ckpt_dir else None
    if resume and ckpt_path and os.path.exists(ckpt_path):
        st = load_ckpt(ckpt_path, st)
        print(f"[resume] from env_step {st['env_step']}", flush=True)

    def collect(sample):
        (st["obs"], st["estate"], st["rings"], st["key"]), outs = rollout(
            st["params"], st["obs"], st["estate"], st["rings"], st["key"],
            collect_steps, sample)
        img, a_int, reward, done, _ = outs
        st["buf"] = replay.add_rollout(st["buf"], img, a_int, reward, done)
        st["env_step"] += collect_steps * cfg.num_envs

    print(f"[prefill] to {cfg.prefill} env-steps...", flush=True)
    while st["env_step"] < cfg.prefill:
        collect(True)

    n_grad = max(1, int(grad_per_collect * collect_steps))
    last_eval = last_log = st["env_step"]
    t0 = time.time()
    es0, gs0 = st["env_step"], st["grad_step"]
    print(f"[train] target {total_env_steps} env-steps "
          f"({cfg.num_envs} envs, {n_grad} grad/iter)", flush=True)
    while st["env_step"] < total_env_steps:
        collect(True)
        for _ in range(n_grad):
            st["key"], bk, tk = jax.random.split(st["key"], 3)
            batch = replay.sample(st["buf"], bk, cfg.batch_size, cfg.L, st["A"])
            st["params"], st["opt_state"], st["perc"], metrics = train_step(
                st["params"], st["opt_state"], st["perc"], batch, tk)
            st["grad_step"] += 1

        if st["env_step"] - last_log >= log_every_env:
            dt = time.time() - t0
            eps = (st["env_step"] - es0) / dt
            gps = (st["grad_step"] - gs0) / dt
            m = {k: float(metrics[k]) for k in ("total_loss", "image_loss",
                 "kl_prior", "reward_loss", "actor_loss", "value_loss")}
            print(f"[{st['env_step']:>8}] env/s {eps:7.0f} grad/s {gps:5.1f} | "
                  f"loss {m['total_loss']:8.2f} img {m['image_loss']:7.2f} "
                  f"klpr {m['kl_prior']:4.2f} rew {m['reward_loss']:4.2f} "
                  f"act {m['actor_loss']:+.3f} val {m['value_loss']:5.2f}", flush=True)
            last_log = st["env_step"]

        if st["env_step"] - last_eval >= eval_every_env:
            ev = evaluate(st, rollout, eval_steps)
            print(f"[eval @ {st['env_step']}] score {ev['score']:.2f} "
                  f"({ev['num_episodes']} eps) reward/step {ev['mean_step_reward']:.4f}",
                  flush=True)
            if ckpt_path:
                save_ckpt(ckpt_path, st)
            last_eval = st["env_step"]

    if ckpt_path:
        save_ckpt(ckpt_path, st)
    ev = evaluate(st, rollout, eval_steps)
    print(f"[done] {st['env_step']} env-steps | final score {ev['score']:.2f}", flush=True)
    return st, ev
