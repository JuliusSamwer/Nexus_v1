"""Distributions / losses in JAX, faithful to emerald_torch.dists.

Differences from the torch version, both forced by JAX's functional RNG:
  * sample(key) / rsample(key) take an explicit PRNGKey.
All operate on the LAST dim as the logits/class dim. Reductions over extra event
dims (the stoch_size categoricals or the 4x4 grid) are done by the caller.
"""

import jax
import jax.numpy as jnp


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def _mix_probs(logits, uniform_mix):
    probs = jax.nn.softmax(logits, axis=-1)
    if uniform_mix > 0:
        n = logits.shape[-1]
        probs = (1 - uniform_mix) * probs + uniform_mix / n
    return probs


class OneHotDist:
    """Categorical over the last dim, returned one-hot, with uniform mix + straight-
    through gradients (DreamerV3/EMERALD)."""

    def __init__(self, logits, uniform_mix=0.01):
        self.logits = logits
        self.num_classes = logits.shape[-1]
        self.probs = _mix_probs(logits, uniform_mix)
        self._logprobs = jnp.log(self.probs)

    def sample(self, key):
        # Sample from the (mixed) probs by sampling from log(probs).
        idx = jax.random.categorical(key, self._logprobs, axis=-1)
        return jax.nn.one_hot(idx, self.num_classes, dtype=self.probs.dtype)

    def rsample(self, key):
        # Straight-through: hard sample forward, soft grad backward.
        hard = self.sample(key)
        return hard + (self.probs - jax.lax.stop_gradient(self.probs))

    def mode(self):
        idx = jnp.argmax(self.probs, axis=-1)
        return jax.nn.one_hot(idx, self.num_classes, dtype=self.probs.dtype)

    def log_prob(self, value):
        # value: one-hot over last dim. Returns per-categorical log prob.
        return jnp.sum(value * self._logprobs, axis=-1)

    def entropy(self):
        return -jnp.sum(self.probs * self._logprobs, axis=-1)


def kl_onehot(logits_p, logits_q, uniform_mix=0.01):
    """KL( cat(p) || cat(q) ) over the last (class) dim, with uniform mix.
    Returns shape == logits_p.shape[:-1]."""
    p = _mix_probs(logits_p, uniform_mix)
    q = _mix_probs(logits_q, uniform_mix)
    return jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=-1)


class SymLogDiscreteDist:
    """Twohot symlog regression head. `logits` last dim == bins; bins evenly spaced in
    symlog space over [low, high]; predictions map to real space via symexp."""

    def __init__(self, logits, low=-20.0, high=20.0):
        self.logits = logits
        self.bins = logits.shape[-1]
        self.buckets = jnp.linspace(low, high, self.bins, dtype=logits.dtype)
        self.logprobs = jax.nn.log_softmax(logits, axis=-1)
        self.probs = jnp.exp(self.logprobs)

    def mode(self):
        return symexp(jnp.sum(self.probs * self.buckets, axis=-1, keepdims=True))

    def mean(self):
        return self.mode()

    def log_prob(self, value):
        # value: (..., 1) in real space. Build twohot target in symlog space.
        x = jnp.clip(symlog(value), self.buckets[0], self.buckets[-1])      # (...,1)
        below = jnp.sum(self.buckets <= x, axis=-1) - 1                     # (...,)
        below = jnp.clip(below, 0, self.bins - 1)
        above = jnp.clip(below + 1, 0, self.bins - 1)
        b_below = self.buckets[below]
        b_above = self.buckets[above]
        denom = b_above - b_below
        weight_above = jnp.where(jnp.abs(denom) < 1e-8, 0.0,
                                 (jnp.squeeze(x, -1) - b_below) / denom)
        weight_above = jnp.clip(weight_above, 0.0, 1.0)
        weight_below = 1.0 - weight_above
        target = (jax.nn.one_hot(below, self.bins) * weight_below[..., None]
                  + jax.nn.one_hot(above, self.bins) * weight_above[..., None])
        target = target.astype(self.logprobs.dtype)
        return jnp.sum(target * self.logprobs, axis=-1)


class MSEDist:
    """Unit-variance Gaussian proxy: mode == mean, log_prob == -squared error summed
    over the trailing `event_dims` (3 for images: C,H,W)."""

    def __init__(self, mode, event_dims=3):
        self._mode = mode
        self.event_dims = event_dims

    def mode(self):
        return self._mode

    def log_prob(self, value):
        se = (self._mode - value) ** 2
        flat = se.reshape(*se.shape[:-self.event_dims], -1)
        return -jnp.sum(flat, axis=-1)


class BernoulliDist:
    """Binary head for the continue predictor. `logits`: (..., 1)."""

    def __init__(self, logits):
        self.logits = logits
        self.probs = jax.nn.sigmoid(logits)

    def mode(self):
        return (self.probs > 0.5).astype(self.logits.dtype)

    def mean(self):
        return self.probs

    def log_prob(self, value):
        # value: (..., 1) in {0,1}. -BCE_with_logits.
        return value * jax.nn.log_sigmoid(self.logits) + (1 - value) * jax.nn.log_sigmoid(-self.logits)
