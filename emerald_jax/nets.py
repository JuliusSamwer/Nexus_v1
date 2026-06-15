"""EMERALD networks in Flax (linen), ported from emerald_torch.nets.

Memory-layout divergence from torch (math identical): everything is channels-LAST
(NHWC). The per-step latent is:
  stoch  : (B, T, 4, 4, S*V) one-hot grid   (torch stored it (B,T,S*V,4,4))
  logits : (B, T, 4, 4, S, V)               (same as torch)
  deter  : (B, T, dim_model)
Spatial conv feature maps are (..., 4, 4, C). LayerNorm is over the channel (last)
axis, so torch's ChLayerNorm is just nn.LayerNorm here. Stride-2 down/up convs use
'SAME' padding (64<->32<->16<->8<->4); exact torch pad isn't needed since we train
from scratch, only consistent shapes.

Modules that sample (encoder posterior, MaskGIT) pull randomness via
self.make_rng('sample'); pass rngs={'sample': key} at apply time.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from . import dists


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
class ResNetV2Block(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):                                   # x: (..., H, W, C)
        h = nn.Conv(self.channels, (3, 3), padding="SAME")(
            nn.silu(nn.LayerNorm(epsilon=1e-3)(x)))
        h = nn.Conv(self.channels, (3, 3), padding="SAME")(
            nn.silu(nn.LayerNorm(epsilon=1e-3)(h)))
        return x + h


class MLP(nn.Module):
    dims: tuple
    norm: bool = True

    @nn.compact
    def __call__(self, x):
        for h in self.dims:
            x = nn.Dense(h)(x)
            if self.norm:
                x = nn.LayerNorm(epsilon=1e-3)(x)
            x = nn.silu(x)
        return x


# --------------------------------------------------------------------------- #
# Transformer (post-norm causal self-attention, learned positional embedding)
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    dim: int
    heads: int

    @nn.compact
    def __call__(self, x, causal=False):
        B, T, D = x.shape
        dh = self.dim // self.heads
        qkv = nn.Dense(3 * self.dim)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        sh = lambda t: t.reshape(B, T, self.heads, dh).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(0, 1, 3, 2)) / (dh ** 0.5)   # (B,h,T,T)
        if causal:
            mask = jnp.tril(jnp.ones((T, T), bool))
            att = jnp.where(mask, att, -jnp.inf)
        att = jax.nn.softmax(att, axis=-1)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return nn.Dense(self.dim)(out)


class TransformerBlock(nn.Module):
    dim: int
    heads: int
    ff_ratio: int = 2

    @nn.compact
    def __call__(self, x, causal=False):
        # post-norm (EMERALD module_pre_norm=False)
        x = nn.LayerNorm(epsilon=1e-3)(x + SelfAttention(self.dim, self.heads)(x, causal))
        ff = nn.Dense(self.ff_ratio * self.dim)(x)
        ff = nn.Dense(self.dim)(nn.relu(ff))
        x = nn.LayerNorm(epsilon=1e-3)(x + ff)
        return x


class Transformer(nn.Module):
    dim: int
    num_blocks: int
    heads: int
    ff_ratio: int = 2
    pos_emb: bool = True
    max_pos: int = 2048

    @nn.compact
    def __call__(self, x, causal=False):
        if self.pos_emb:
            pos = self.param("pos", nn.initializers.zeros, (1, self.max_pos, self.dim))
            x = x + pos[:, :x.shape[1]]
        for _ in range(self.num_blocks):
            x = TransformerBlock(self.dim, self.heads, self.ff_ratio)(x, causal)
        return x


# --------------------------------------------------------------------------- #
# Encoder / image decoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    cfg: object

    @nn.compact
    def __call__(self, image):
        # image: (B,T,3,64,64) CHW -> stoch (B,T,4,4,SV), logits (B,T,4,4,S,V)
        cfg = self.cfg
        B, T = image.shape[:2]
        x = jnp.transpose(image.reshape(-1, *image.shape[2:]), (0, 2, 3, 1))  # (N,64,64,3)
        for d in (cfg.dim_cnn, 2 * cfg.dim_cnn, 4 * cfg.dim_cnn, 8 * cfg.dim_cnn):
            x = nn.Conv(d, (4, 4), strides=(2, 2), padding="SAME")(x)
            for _ in range(cfg.num_res_layers):
                x = ResNetV2Block(d)(x)
        logits = nn.Conv(cfg.stoch_size * cfg.discrete, (3, 3), padding="SAME")(x)  # (N,4,4,SV)
        logits = logits.reshape(-1, 4, 4, cfg.stoch_size, cfg.discrete)
        stoch = dists.OneHotDist(logits, cfg.uniform_mix).rsample(self.make_rng("sample"))
        stoch = stoch.reshape(-1, 4, 4, cfg.stoch_size * cfg.discrete)
        return {
            "stoch": stoch.reshape(B, T, 4, 4, cfg.stoch_size * cfg.discrete),
            "logits": logits.reshape(B, T, 4, 4, cfg.stoch_size, cfg.discrete),
        }


class ImageDecoder(nn.Module):
    cfg: object

    @nn.compact
    def __call__(self, feats):
        cfg = self.cfg
        stoch, deter = feats                                 # (B,T,4,4,SV), (B,T,512)
        B, T = stoch.shape[:2]
        feat_size = cfg.stoch_size * cfg.discrete
        x = stoch.reshape(-1, 4, 4, feat_size)
        d = MLP((4 * 4 * cfg.reduced_channels,))(deter.reshape(-1, deter.shape[-1]))
        d = d.reshape(-1, 4, 4, cfg.reduced_channels)
        d = nn.Conv(cfg.dim_cnn, (1, 1))(d)
        x = nn.Conv(8 * cfg.dim_cnn, (1, 1))(jnp.concatenate([x, d], axis=-1))
        dims = (8 * cfg.dim_cnn, 4 * cfg.dim_cnn, 2 * cfg.dim_cnn, cfg.dim_cnn)
        for i, dd in enumerate(dims):
            for _ in range(cfg.num_res_layers):
                x = ResNetV2Block(dd)(x)
            out = cfg.image_channels if i == len(dims) - 1 else dims[i + 1]
            x = nn.ConvTranspose(out, (4, 4), strides=(2, 2), padding="SAME")(x)
        x = jnp.transpose(x, (0, 3, 1, 2))                   # (N,3,64,64)
        return dists.MSEDist(x.reshape(B, T, *x.shape[1:]), event_dims=3)


# --------------------------------------------------------------------------- #
# MaskGIT prior network
# --------------------------------------------------------------------------- #
class MaskNetwork(nn.Module):
    cfg: object
    dim_model: int

    def setup(self):
        cfg = self.cfg
        self.S, self.V = cfg.stoch_size, cfg.discrete
        self.mask_id = self.V
        self.N = 4 * 4 * self.V
        self.steps = cfg.num_decoding_steps
        self.dynamics_predictor = nn.Dense(self.S * self.V)
        if self.steps > 0:
            self.embed = nn.Embed(self.V + 1, self.V)
            self.input_layer = nn.Dense(self.dim_model)
            self.pos = self.param("pos", nn.initializers.zeros, (1, 16, self.dim_model))
            self.transformer = Transformer(
                self.dim_model, cfg.num_blocks_mask, cfg.num_heads_trans,
                cfg.ff_ratio_trans, pos_emb=False)
            self.output_layer = nn.Dense(self.S * self.V)

    def _inner(self, stoch_masked, deter):
        # stoch_masked ids (B,T,4,4,S); deter (B,T,4,4,Dm) -> logits (B,T,4,4,S,V)
        B, T = stoch_masked.shape[:2]
        e = self.embed(stoch_masked).reshape(*stoch_masked.shape[:-1], self.S * self.V)
        x = self.input_layer(jnp.concatenate([e, deter], axis=-1))      # (B,T,4,4,Dm)
        x = x.reshape(B * T, 16, -1) + self.pos
        x = self.transformer(x, causal=False).reshape(B, T, 4, 4, -1)
        return self.output_layer(x).reshape(B, T, 4, 4, self.S, self.V)

    def train_logits(self, stoch, deter):
        """deter: (B,T,4,4,Dm). Returns (logits, logits_masked, mask)."""
        cfg = self.cfg
        B, T = deter.shape[:2]
        logits = self.dynamics_predictor(deter).reshape(B, T, 4, 4, self.S, self.V)
        if self.steps == 0:
            return logits, None, None
        stoch_argmax = stoch.reshape(B, T, 4, 4, self.S, self.V).argmax(-1)  # (B,T,4,4,S)
        kr, ko = self.make_rng("mask"), self.make_rng("order")
        ratio = jax.random.uniform(kr, (B, T))
        ratio = jnp.clip(jnp.floor(jnp.cos(jnp.pi / 2 * ratio) * self.N), 1.0)
        order = jnp.argsort(jax.random.uniform(ko, (B, T, self.N)), axis=-1)
        rank = jnp.argsort(order, axis=-1)
        mask = (rank < ratio[..., None]).reshape(B, T, 4, 4, self.S)         # True==masked
        stoch_masked = jnp.where(mask, self.mask_id, stoch_argmax)
        logits_masked = self._inner(stoch_masked, deter)
        return logits, logits_masked, mask

    def sample(self, deter, num_steps):
        """Iterative MaskGIT decoding. deter (B,T,4,4,Dm) -> (logits, stoch one-hot
        (B,T,4,4,S,V)). Randomness via self.make_rng('sample')."""
        cfg = self.cfg
        B, T = deter.shape[:2]
        if num_steps == 0 or self.steps == 0:
            logits = self.dynamics_predictor(deter).reshape(B, T, 4, 4, self.S, self.V)
            stoch = dists.OneHotDist(logits, cfg.uniform_mix).sample(self.make_rng("sample"))
            return logits, stoch
        ratios = jnp.arange(num_steps) / num_steps
        ratios = jnp.clip(jnp.floor(jnp.cos(jnp.pi / 2 * ratios) * self.N), 1.0)
        mask = logits = stoch = None
        for step in range(num_steps):
            if step > 0:
                sel = jnp.sum(jax.nn.softmax(logits, -1) * stoch, -1)        # (B,T,4,4,S)
                g = jax.random.uniform(self.make_rng("sample"), sel.shape)
                conf = jnp.log(sel + 1e-8) - jnp.log(-jnp.log(g + 1e-8) + 1e-8)
                if mask is not None:
                    conf = jnp.where(jnp.logical_not(mask), jnp.inf, conf)
                flat = conf.reshape(B, T, self.N)
                rank = jnp.argsort(jnp.argsort(flat, -1), -1)
                mask = (rank < ratios[step]).reshape(B, T, 4, 4, self.S)
                stoch_masked = jnp.where(mask, self.mask_id, stoch.argmax(-1))
            else:
                stoch_masked = jnp.full((B, T, 4, 4, self.S), self.mask_id, jnp.int32)
            pred = self._inner(stoch_masked, deter)
            if step > 0:
                logits = jnp.where(mask[..., None], pred, logits)
                new = dists.OneHotDist(logits, cfg.uniform_mix).sample(self.make_rng("sample"))
                stoch = jnp.where(mask[..., None], new, stoch)
            else:
                logits = pred
                stoch = dists.OneHotDist(logits, cfg.uniform_mix).sample(self.make_rng("sample"))
        return logits, stoch


# --------------------------------------------------------------------------- #
# TSSM — transformer state-space dynamics
# --------------------------------------------------------------------------- #
class TSSM(nn.Module):
    cfg: object
    num_actions: int

    def setup(self):
        cfg = self.cfg
        self.feat_size = cfg.stoch_size * cfg.discrete
        self.dim_cnn_inner = 8 * cfg.dim_cnn                 # 256
        self.enc_conv = nn.Conv(cfg.reduced_channels, (1, 1))
        self.enc_norm = nn.LayerNorm(epsilon=1e-3)
        self.enc_lin = nn.Dense(cfg.dim_model)
        self.action_mixer = MLP((cfg.dim_model,))
        self.action_mixer2 = nn.Dense(cfg.dim_model)
        self.transformer = Transformer(cfg.dim_model, cfg.num_blocks_trans,
                                       cfg.num_heads_trans, cfg.ff_ratio_trans, pos_emb=True)
        self.dec_mlp = MLP((4 * 4 * cfg.reduced_channels,))
        self.dec_conv = nn.Conv(self.dim_cnn_inner, (1, 1))
        self.mask_network = MaskNetwork(cfg, dim_model=self.dim_cnn_inner)
        if cfg.learn_initial:
            self.weight_init = self.param("weight_init", nn.initializers.zeros,
                                          (cfg.dim_model,))

    def encode_stoch(self, stoch):
        # stoch (B,T,4,4,SV) -> (B,T,512)
        B, T = stoch.shape[:2]
        x = nn.silu(self.enc_norm(self.enc_conv(stoch.reshape(-1, 4, 4, self.feat_size))))
        return self.enc_lin(x.reshape(B * T, -1)).reshape(B, T, -1)

    def mix(self, stoch_emb, action):
        return self.action_mixer2(self.action_mixer(
            jnp.concatenate([stoch_emb, action], axis=-1)))

    def deter_to_dec(self, deter):
        # (B,T,512) -> (B,T,4,4,256)
        B, T = deter.shape[:2]
        x = self.dec_mlp(deter.reshape(-1, deter.shape[-1]))
        x = self.dec_conv(x.reshape(-1, 4, 4, self.cfg.reduced_channels))
        return x.reshape(B, T, 4, 4, self.dim_cnn_inner)

    def initial_deter(self, B, T):
        if self.cfg.learn_initial:
            return jnp.broadcast_to(jnp.tanh(self.weight_init), (B, T, self.cfg.dim_model))
        return jnp.zeros((B, T, self.cfg.dim_model))

    def initial_stoch(self, B, T):
        deter = self.initial_deter(B, T)
        _, stoch = self.mask_network.sample(self.deter_to_dec(deter), 0)
        return jax.lax.stop_gradient(
            stoch.reshape(B, T, 4, 4, self.feat_size))       # (B,T,4,4,SV)

    def observe(self, stoch, actions):
        """stoch (B,L,4,4,SV) posterior; actions (B,L,A). Returns post, prior."""
        B, L = stoch.shape[:2]
        init = self.initial_stoch(B, 1)
        prev_stoch = jnp.concatenate([init, stoch[:, :-1]], axis=1)
        x = self.mix(self.encode_stoch(prev_stoch), actions)
        deter = self.transformer(x, causal=True)             # (B,L,512)
        deter_dec = self.deter_to_dec(deter)
        logits, logits_masked, mask = self.mask_network.train_logits(stoch, deter_dec)
        post = {"stoch": stoch, "deter": deter}
        prior = {"logits": logits, "logits_masked": logits_masked, "mask": mask}
        return post, prior


# --------------------------------------------------------------------------- #
# Prediction heads
# --------------------------------------------------------------------------- #
class Head(nn.Module):
    cfg: object
    out_dim: int
    zero_init_out: bool = False

    @nn.compact
    def __call__(self, feats):
        cfg = self.cfg
        stoch, deter = feats                                 # (...,4,4,SV), (...,Dm)
        feat_size = cfg.stoch_size * cfg.discrete
        lead = stoch.shape[:-3]
        x = nn.Conv(cfg.reduced_channels, (1, 1))(stoch.reshape(-1, 4, 4, feat_size))
        x = nn.silu(nn.LayerNorm(epsilon=1e-3)(x))
        x = nn.Dense(cfg.dim_model)(x.reshape(-1, 4 * 4 * cfg.reduced_channels))
        x = x.reshape(*lead, cfg.dim_model)
        x = jnp.concatenate([x, deter], axis=-1)
        x = MLP((cfg.dim_model,) * cfg.num_layers)(x)
        out_kw = {}
        if self.zero_init_out:
            out_kw = dict(kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)
        return nn.Dense(self.out_dim, **out_kw)(x)
