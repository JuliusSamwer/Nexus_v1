"""Craftax-Classic env wrapper for the JAX EMERALD port.

Craftax-Classic-Pixels gives obs (63,63,3) float32 in [0,1], 17 actions, and a flat
dict of cumulative Achievements/* in `info`. We pad to 64x64, move to channel-first,
and center to [-0.5, 0.5] so the encoder geometry + image range match emerald_torch.

Everything here is pure/jittable: the env object is static (closed over), states are
flax-struct pytrees, so reset/step compose under jax.vmap / jax.lax.scan.
"""

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

ENV_NAME = "Craftax-Classic-Pixels-v1"
NUM_ACTIONS = 17
IMAGE_SIZE = 64
RAW_SIZE = 63
# 22 Crafter achievements, recovered from info keys at make-time.


def make_env(auto_reset=True):
    env = make_craftax_env_from_name(ENV_NAME, auto_reset=auto_reset)
    params = env.default_params
    return env, params


def ach_names(info):
    """Sorted list of the 22 'Achievements/<name>' keys present in step info."""
    return sorted(k.split("/", 1)[1] for k in info if k.startswith("Achievements/"))


def process_obs(obs):
    """(63,63,3) [0,1] HWC  ->  (3,64,64) [-0.5,0.5] CHW. Vmappable over a leading
    batch via jax.vmap; written for a single observation."""
    img = jnp.pad(obs, ((0, IMAGE_SIZE - RAW_SIZE), (0, IMAGE_SIZE - RAW_SIZE), (0, 0)),
                  mode="edge")                                   # (64,64,3)
    img = jnp.transpose(img, (2, 0, 1))                          # (3,64,64)
    return img - 0.5


def reset(env, params, keys):
    """Vmapped reset over a batch of keys -> (image (N,3,64,64), state)."""
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    return jax.vmap(process_obs)(obs), state


def step(env, params, keys, state, action):
    """Vmapped step. Returns image (N,3,64,64), state, reward (N,), done (N,), info."""
    obs, state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None))(keys, state, action, params)
    return jax.vmap(process_obs)(obs), state, reward, done, info
