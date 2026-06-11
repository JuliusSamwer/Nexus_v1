"""Distributions / losses, standalone (faithful to EMERALD's nnet.distributions).

All operate on the LAST dim as the logits/class dim unless noted. Reductions over
extra event dims (e.g. the stoch_size categoricals, or the 4x4 grid) are done by the
caller, matching EMERALD's loss code.
"""

import torch
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class OneHotDist:
    """Categorical over the last dim, returned as one-hot, with a uniform mix and
    straight-through gradients (DreamerV3/EMERALD style)."""

    def __init__(self, logits, uniform_mix=0.01):
        self.logits = logits
        self.num_classes = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)
        if uniform_mix > 0:
            probs = (1 - uniform_mix) * probs + uniform_mix / self.num_classes
        self.probs = probs
        self._logprobs = torch.log(probs)

    def _sample_indices(self):
        # Inverse-CDF sampling (matches EMERALD's multinomial_one_hot), no grad.
        u = torch.rand(self.probs.shape[:-1], device=self.probs.device,
                       dtype=self.probs.dtype).unsqueeze(-1)
        return torch.sum(u > torch.cumsum(self.probs, dim=-1)[..., :-1], dim=-1)

    def sample(self):
        idx = self._sample_indices()
        return F.one_hot(idx, self.num_classes).type(self.probs.dtype)

    def rsample(self):
        # Straight-through: hard sample on the forward pass, soft grad on the backward.
        hard = self.sample()
        return hard + (self.probs - self.probs.detach())

    def mode(self):
        idx = self.probs.argmax(dim=-1)
        return F.one_hot(idx, self.num_classes).type(self.probs.dtype)

    def log_prob(self, value):
        # value: one-hot over last dim. Returns per-categorical log prob (sum over class dim).
        return (value * self._logprobs).sum(dim=-1)

    def entropy(self):
        return -(self.probs * self._logprobs).sum(dim=-1)


def kl_onehot(logits_p, logits_q, uniform_mix=0.01):
    """KL( cat(logits_p) || cat(logits_q) ) over the last (class) dim, with uniform mix.
    Returns shape == logits_p.shape[:-1] (one value per categorical)."""
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    if uniform_mix > 0:
        n = logits_p.shape[-1]
        p = (1 - uniform_mix) * p + uniform_mix / n
        q = (1 - uniform_mix) * q + uniform_mix / n
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)


class SymLogDiscreteDist:
    """Twohot symlog regression head (DreamerV3/EMERALD). `logits` last dim == bins.
    Bins are evenly spaced in symlog space over [low, high]; predictions live in real
    space via symexp."""

    def __init__(self, logits, low=-20.0, high=20.0):
        self.logits = logits
        self.bins = logits.shape[-1]
        self.buckets = torch.linspace(low, high, self.bins, device=logits.device,
                                      dtype=logits.dtype)
        self.logprobs = F.log_softmax(logits, dim=-1)
        self.probs = self.logprobs.exp()

    def mode(self):
        # Expected bucket value in symlog space, mapped back to real space.
        return symexp((self.probs * self.buckets).sum(dim=-1, keepdim=True))

    # alias used in a couple of EMERALD call sites
    def mean(self):
        return self.mode()

    def log_prob(self, value):
        # value: (..., 1) in real space. Build the twohot target in symlog space.
        x = symlog(value).clamp(self.buckets[0], self.buckets[-1])  # (..., 1)
        # locate the bracketing buckets
        below = (self.buckets <= x).sum(dim=-1) - 1            # (...,)
        below = below.clamp(0, self.bins - 1)
        above = (below + 1).clamp(0, self.bins - 1)
        b_below = self.buckets[below]
        b_above = self.buckets[above]
        denom = (b_above - b_below)
        weight_above = torch.where(denom.abs() < 1e-8,
                                   torch.zeros_like(denom),
                                   (x.squeeze(-1) - b_below) / denom).clamp(0, 1)
        weight_below = 1.0 - weight_above
        target = (F.one_hot(below, self.bins) * weight_below.unsqueeze(-1)
                  + F.one_hot(above, self.bins) * weight_above.unsqueeze(-1)).type(self.logprobs.dtype)
        return (target * self.logprobs).sum(dim=-1)


class MSEDist:
    """Gaussian-with-unit-variance proxy: mode == mean, log_prob == -squared error
    summed over the trailing `event_dims` (EMERALD uses 3 for images: C,H,W)."""

    def __init__(self, mode, event_dims=3):
        self._mode = mode
        self.event_dims = event_dims

    def mode(self):
        return self._mode

    def log_prob(self, value):
        se = (self._mode - value) ** 2
        return -se.flatten(start_dim=-self.event_dims).sum(dim=-1)


class BernoulliDist:
    """Binary head for the continue predictor. `logits`: (..., 1)."""

    def __init__(self, logits):
        self.logits = logits
        self.probs = torch.sigmoid(logits)

    def mode(self):
        return (self.probs > 0.5).type(self.logits.dtype)

    def mean(self):
        return self.probs

    def log_prob(self, value):
        # value: (..., 1) in {0,1}. Returns (..., 1).
        return -F.binary_cross_entropy_with_logits(self.logits, value, reduction="none")
