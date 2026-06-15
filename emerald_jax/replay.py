"""On-device replay buffer for the JAX EMERALD port.

A ring buffer of shape (capacity, num_envs, ...) holding one transition per
(time, env): image_t (CHW [-0.5,0.5]), action_t (int, taken FROM image_t), reward_t
and done_t (the result of action_t). Add whole collected rollouts; sample length-L
training windows. Fully jittable (flax.struct pytree, lax scatter/gather).

ALIGNMENT (faithful to emerald_torch): the world model wants, per state s_t,
"the action that led INTO s_t" (a_0=0) and "the reward for arriving at s_t". So we
sample L+1 consecutive rows and shift:
    image[i]  = row[i+1].image      # the state s_{i+1}
    action[i] = row[i].action       # a_i, taken at s_i -> leads into image[i]
    reward[i] = row[i].reward        # reward of that same transition
    cont[i]   = 1 - row[i].done       # continue flag of that transition
This reproduces observe()'s prev_stoch=[init, stoch[:-1]] / actions pairing exactly.

WITHIN-EPISODE SIMPLIFICATION: windows are sampled uniformly from the filled region.
With Craftax auto_reset a window can rarely straddle an episode boundary (or the ring
write-seam). We ACCEPT this (no rejection sampling) — it matches emerald_torch's own
"within-episode is approximate" note, and the cont=1-done signal stays correct across
the seam so the reward/continue heads are unaffected; only observe()'s single-init
assumption is mildly violated on those rare windows.
"""

import flax
import jax
import jax.numpy as jnp


# Images are stored as uint8 (0..255) to 4x the on-device capacity (DreamerV3-style);
# they round-trip to/from the model's [-0.5, 0.5] float range here.
def _to_u8(img):                          # float [-0.5,0.5] -> uint8
    return jnp.clip(jnp.round((img + 0.5) * 255.0), 0, 255).astype(jnp.uint8)


def _from_u8(u8):                         # uint8 -> float [-0.5,0.5]
    return u8.astype(jnp.float32) / 255.0 - 0.5


@flax.struct.dataclass
class Buffer:
    image: jnp.ndarray          # (cap, num_envs, 3, 64, 64) uint8
    action: jnp.ndarray         # (cap, num_envs) i32
    reward: jnp.ndarray         # (cap, num_envs) f32
    done: jnp.ndarray           # (cap, num_envs) bool
    ptr: jnp.ndarray            # () i32  next write row
    size: jnp.ndarray           # () i32  filled rows (<= cap)
    capacity: int = flax.struct.field(pytree_node=False)
    num_envs: int = flax.struct.field(pytree_node=False)


def init_buffer(capacity, num_envs, image_size=64, channels=3):
    z = jnp.zeros
    return Buffer(
        image=z((capacity, num_envs, channels, image_size, image_size), jnp.uint8),
        action=z((capacity, num_envs), jnp.int32),
        reward=z((capacity, num_envs), jnp.float32),
        done=z((capacity, num_envs), jnp.bool_),
        ptr=jnp.int32(0), size=jnp.int32(0),
        capacity=capacity, num_envs=num_envs)


def add_rollout(buf, image, action, reward, done):
    """Append a collected rollout of T steps. Arrays are (T, num_envs, ...);
    image is float [-0.5,0.5] and is stored as uint8."""
    cap = buf.capacity
    T = image.shape[0]
    rows = (buf.ptr + jnp.arange(T)) % cap
    buf = buf.replace(
        image=buf.image.at[rows].set(_to_u8(image)),
        action=buf.action.at[rows].set(action.astype(jnp.int32)),
        reward=buf.reward.at[rows].set(reward.astype(jnp.float32)),
        done=buf.done.at[rows].set(done.astype(jnp.bool_)),
        ptr=(buf.ptr + T) % cap,
        size=jnp.minimum(buf.size + T, cap))
    return buf


def sample(buf, key, batch_size, L, num_actions):
    """Return a training batch of within-episode windows (see module docstring)."""
    cap = buf.capacity
    k1, k2 = jax.random.split(key)
    full = buf.size >= cap
    base = jnp.where(full, buf.ptr, 0)                       # logical-0 -> physical row
    max_start = jnp.maximum(buf.size - (L + 1), 1)
    starts = jax.random.randint(k1, (batch_size,), 0, max_start)
    envs = jax.random.randint(k2, (batch_size,), 0, buf.num_envs)

    def gather_one(start, e):
        rows = (base + start + jnp.arange(L + 1)) % cap
        return (buf.image[rows, e], buf.action[rows, e],
                buf.reward[rows, e], buf.done[rows, e])

    img, act, rew, don = jax.vmap(gather_one)(starts, envs)  # (B, L+1, ...)
    img = _from_u8(img)
    return {
        "image": img[:, 1:],                                 # (B,L,3,64,64)
        "action": jax.nn.one_hot(act[:, :-1], num_actions),  # (B,L,A)
        "reward": rew[:, :-1],                               # (B,L)
        "cont": 1.0 - don[:, :-1].astype(jnp.float32),       # (B,L)
    }
