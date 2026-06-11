"""EMERALD networks, clean-room in plain PyTorch.

Latent geometry (EMERALD defaults): image 64x64x3 -> 4x4 spatial grid; per cell
stoch_size=32 categoricals each with discrete=32 classes. So the per-step latent is a
(4,4,32,32) one-hot, stored channel-first as stoch (1024,4,4). deter (dim_model=512)
is the transformer summary of history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dists


# --------------------------------------------------------------------------- #
# Norm / blocks
# --------------------------------------------------------------------------- #
class ChLayerNorm(nn.Module):
    """LayerNorm over the channel dim for (N, C, H, W) conv features."""

    def __init__(self, channels, eps=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResNetV2Block(nn.Module):
    """norm -> act -> conv -> norm -> act -> conv -> + residual."""

    def __init__(self, channels, act=nn.SiLU):
        super().__init__()
        self.n1 = ChLayerNorm(channels)
        self.n2 = ChLayerNorm(channels)
        self.c1 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.act = act()

    def forward(self, x):
        h = self.c1(self.act(self.n1(x)))
        h = self.c2(self.act(self.n2(h)))
        return x + h


def mlp(dim_in, dims, act=nn.SiLU, norm=True):
    layers, d = [], dim_in
    for i, h in enumerate(dims):
        layers.append(nn.Linear(d, h))
        if norm:
            layers.append(nn.LayerNorm(h, eps=1e-3))
        layers.append(act())
        d = h
    return nn.Sequential(*layers), d


# --------------------------------------------------------------------------- #
# Transformer (standard pre-/post-norm causal self-attention)
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    def __init__(self, dim, heads, drop=0.1):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, causal=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        if causal:
            m = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            att = att.masked_fill(m, float("-inf"))
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_ratio=2, drop=0.1):
        super().__init__()
        self.attn = SelfAttention(dim, heads, drop)
        self.n1 = nn.LayerNorm(dim, eps=1e-3)
        self.n2 = nn.LayerNorm(dim, eps=1e-3)
        self.ff = nn.Sequential(nn.Linear(dim, ff_ratio * dim), nn.ReLU(),
                                nn.Linear(ff_ratio * dim, dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x, causal=False):
        # post-norm (EMERALD module_pre_norm=False)
        x = self.n1(x + self.drop(self.attn(x, causal=causal)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_blocks, heads, ff_ratio=2, drop=0.1, max_pos=2048,
                 pos_emb=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, ff_ratio, drop) for _ in range(num_blocks)])
        self.pos = nn.Parameter(torch.zeros(1, max_pos, dim)) if pos_emb else None

    def forward(self, x, causal=False):
        if self.pos is not None:
            x = x + self.pos[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x, causal=causal)
        return x


# --------------------------------------------------------------------------- #
# Encoder / image decoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dims = [cfg.dim_cnn, 2 * cfg.dim_cnn, 4 * cfg.dim_cnn, 8 * cfg.dim_cnn]
        layers, c = [], cfg.image_channels
        for d in dims:
            layers.append(nn.Conv2d(c, d, 4, 2, padding=1))
            for _ in range(cfg.num_res_layers):
                layers.append(ResNetV2Block(d))
            c = d
        self.cnn = nn.Sequential(*layers)
        self.repr = nn.Conv2d(8 * cfg.dim_cnn, cfg.stoch_size * cfg.discrete, 3, 1, padding=1)

    def forward(self, image):
        # image: (B, T, 3, 64, 64) -> stoch (B,T,SV,4,4), logits (B,T,4,4,S,V)
        cfg = self.cfg
        B, T = image.shape[:2]
        x = self.cnn(image.reshape(-1, *image.shape[2:]))            # (N, 256, 4, 4)
        logits = self.repr(x).permute(0, 2, 3, 1)                    # (N, 4, 4, SV)
        logits = logits.reshape(-1, 4, 4, cfg.stoch_size, cfg.discrete)
        stoch = dists.OneHotDist(logits, cfg.uniform_mix).rsample()  # (N,4,4,S,V)
        stoch = stoch.flatten(-2, -1).permute(0, 3, 1, 2)            # (N, SV, 4, 4)
        return {
            "stoch": stoch.reshape(B, T, cfg.stoch_size * cfg.discrete, 4, 4),
            "logits": logits.reshape(B, T, 4, 4, cfg.stoch_size, cfg.discrete),
        }


class ImageDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feat_size = cfg.stoch_size * cfg.discrete
        self.deter_dec, _ = mlp(cfg.dim_model, [4 * 4 * cfg.reduced_channels])
        self.unflat = (cfg.reduced_channels, 4, 4)
        self.deter_conv = nn.Conv2d(cfg.reduced_channels, cfg.dim_cnn, 1)
        self.proj = nn.Conv2d(self.feat_size + cfg.dim_cnn, 8 * cfg.dim_cnn, 1)
        dims = [8 * cfg.dim_cnn, 4 * cfg.dim_cnn, 2 * cfg.dim_cnn, cfg.dim_cnn]
        layers = []
        for i, d in enumerate(dims):
            for _ in range(cfg.num_res_layers):
                layers.append(ResNetV2Block(d))
            out = cfg.image_channels if i == len(dims) - 1 else dims[i + 1]
            layers.append(nn.ConvTranspose2d(d, out, 4, 2, padding=1))
        self.cnn = nn.Sequential(*layers)

    def forward(self, feats):
        stoch, deter = feats
        B, T = stoch.shape[:2]
        x = stoch.reshape(-1, self.feat_size, 4, 4)
        d = self.deter_dec(deter.reshape(-1, deter.shape[-1]))
        d = self.deter_conv(d.reshape(-1, *self.unflat))
        x = self.proj(torch.cat([x, d], dim=1))
        x = self.cnn(x)                                              # (N, 3, 64, 64)
        return dists.MSEDist(x.reshape(B, T, *x.shape[1:]), event_dims=3)


# --------------------------------------------------------------------------- #
# MaskGIT prior network
# --------------------------------------------------------------------------- #
class MaskNetwork(nn.Module):
    def __init__(self, cfg, dim_model):
        super().__init__()
        self.cfg = cfg
        self.S, self.V = cfg.stoch_size, cfg.discrete
        self.mask_id = self.V
        self.N = 4 * 4 * self.V                                     # masked-token count
        self.steps = cfg.num_decoding_steps
        self.dynamics_predictor = nn.Linear(dim_model, self.S * self.V)
        if self.steps > 0:
            self.embed = nn.Embedding(self.V + 1, self.V)
            self.input_layer = nn.Linear(dim_model + self.S * self.V, dim_model)
            self.pos = nn.Parameter(torch.zeros(1, 16, dim_model))
            self.transformer = Transformer(dim_model, cfg.num_blocks_mask,
                                           cfg.num_heads_trans, cfg.ff_ratio_trans,
                                           cfg.drop_rate_trans, pos_emb=False)
            self.output_layer = nn.Linear(dim_model, self.S * self.V)

    def _inner(self, stoch_masked, deter):
        # stoch_masked ids (B,T,4,4,S); deter (B,T,4,4,Dm) -> logits (B,T,4,4,S,V)
        B, T = stoch_masked.shape[:2]
        e = self.embed(stoch_masked).flatten(-2, -1)               # (B,T,4,4,S*V)
        x = self.input_layer(torch.cat([e, deter], dim=-1))        # (B,T,4,4,Dm)
        x = x.reshape(B * T, 16, -1) + self.pos
        x = self.transformer(x, causal=False).reshape(B, T, 4, 4, -1)
        return self.output_layer(x).reshape(B, T, 4, 4, self.S, self.V)

    def train_logits(self, stoch, deter_dec):
        """Training path: direct prior + masked prior + mask (EMERALD forward)."""
        cfg = self.cfg
        deter = deter_dec.permute(0, 1, 3, 4, 2)                    # (B,T,4,4,Dm)
        B, T = deter.shape[:2]
        logits = self.dynamics_predictor(deter).reshape(B, T, 4, 4, self.S, self.V)
        if self.steps == 0:
            return logits, None, None
        stoch_argmax = (stoch.permute(0, 1, 3, 4, 2)
                        .reshape(B, T, 4, 4, self.S, self.V).argmax(dim=-1))  # (B,T,4,4,S)
        # cosine mask ratio -> number of masked tokens
        ratio = torch.empty(B, T, device=deter.device).uniform_(0, 1)
        ratio = torch.floor(torch.cos(torch.pi / 2 * ratio) * self.N).clamp(min=1.0)
        order = torch.argsort(torch.rand(B, T, self.N, device=deter.device), dim=-1)
        rank = torch.argsort(order, dim=-1)
        mask = (rank < ratio.unsqueeze(-1)).reshape(B, T, 4, 4, self.S)  # True == masked
        stoch_masked = torch.where(mask, self.mask_id, stoch_argmax)
        logits_masked = self._inner(stoch_masked, deter)
        return logits, logits_masked, mask

    def sample(self, deter_dec, num_steps):
        """Imagination path: iterative MaskGIT decoding -> (logits, stoch one-hot)."""
        cfg = self.cfg
        deter = deter_dec.permute(0, 1, 3, 4, 2)                    # (B,T,4,4,Dm)
        B, T = deter.shape[:2]
        if num_steps == 0 or self.steps == 0:
            logits = self.dynamics_predictor(deter).reshape(B, T, 4, 4, self.S, self.V)
            stoch = dists.OneHotDist(logits, cfg.uniform_mix).sample()
            return logits, stoch
        ratios = torch.arange(num_steps, device=deter.device) / num_steps
        ratios = torch.floor(torch.cos(torch.pi / 2 * ratios) * self.N).clamp(min=1.0)
        mask, logits, stoch = None, None, None
        for step in range(num_steps):
            if step > 0:
                sel = torch.sum(torch.softmax(logits, dim=-1) * stoch, dim=-1)  # (B,T,4,4,S)
                conf = torch.log(sel + 1e-8) - torch.log(
                    -torch.log(torch.rand_like(sel) + 1e-8) + 1e-8)
                if mask is not None:
                    conf = torch.where(mask.logical_not(), torch.inf, conf)
                flat = conf.reshape(B, T, self.N)
                rank = torch.argsort(torch.argsort(flat, dim=-1), dim=-1)
                mask = (rank < ratios[step]).reshape(B, T, 4, 4, self.S)
                stoch_masked = torch.where(mask, self.mask_id, stoch.argmax(dim=-1))
            else:
                stoch_masked = self.mask_id * torch.ones(
                    B, T, 4, 4, self.S, device=deter.device, dtype=torch.long)
            pred = self._inner(stoch_masked, deter)
            if step > 0:
                logits = torch.where(mask.unsqueeze(-1), pred, logits)
                new = dists.OneHotDist(logits, cfg.uniform_mix).sample()
                stoch = torch.where(mask.unsqueeze(-1), new, stoch)
            else:
                logits = pred
                stoch = dists.OneHotDist(logits, cfg.uniform_mix).sample()
        return logits, stoch


# --------------------------------------------------------------------------- #
# TSSM — transformer state-space dynamics
# --------------------------------------------------------------------------- #
class TSSM(nn.Module):
    def __init__(self, cfg, num_actions):
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        self.feat_size = cfg.stoch_size * cfg.discrete
        dim_cnn_inner = 8 * cfg.dim_cnn                             # 256, the deter_dec width
        # stoch (SV,4,4) -> 512
        self.enc_conv = nn.Conv2d(self.feat_size, cfg.reduced_channels, 1)
        self.enc_norm = ChLayerNorm(cfg.reduced_channels)
        self.enc_lin = nn.Linear(4 * 4 * cfg.reduced_channels, cfg.dim_model)
        self.act = nn.SiLU()
        self.action_mixer, _ = mlp(cfg.dim_model + num_actions, [cfg.dim_model])
        self.action_mixer2 = nn.Linear(cfg.dim_model, cfg.dim_model)
        self.transformer = Transformer(cfg.dim_model, cfg.num_blocks_trans,
                                       cfg.num_heads_trans, cfg.ff_ratio_trans,
                                       cfg.drop_rate_trans, pos_emb=True)
        # deter (512) -> deter_dec (dim_cnn_inner,4,4)
        self.dec_mlp, _ = mlp(cfg.dim_model, [4 * 4 * cfg.reduced_channels])
        self.dec_conv = nn.Conv2d(cfg.reduced_channels, dim_cnn_inner, 1)
        self.mask_network = MaskNetwork(cfg, dim_model=dim_cnn_inner)
        if cfg.learn_initial:
            self.weight_init = nn.Parameter(torch.zeros(cfg.dim_model))

    # ---- helpers -------------------------------------------------------- #
    def encode_stoch(self, stoch):
        # stoch (B,T,SV,4,4) -> (B,T,512)
        B, T = stoch.shape[:2]
        x = self.act(self.enc_norm(self.enc_conv(stoch.reshape(-1, self.feat_size, 4, 4))))
        return self.enc_lin(x.flatten(1)).reshape(B, T, -1)

    def mix(self, stoch_emb, action):
        return self.action_mixer2(self.action_mixer(torch.cat([stoch_emb, action], dim=-1)))

    def deter_to_dec(self, deter):
        B, T = deter.shape[:2]
        x = self.dec_mlp(deter.reshape(-1, deter.shape[-1]))
        x = self.dec_conv(x.reshape(-1, self.cfg.reduced_channels, 4, 4))
        return x.reshape(B, T, *x.shape[1:])                        # (B,T,256,4,4)

    def initial_deter(self, B, T, device, dtype=torch.float32):
        if self.cfg.learn_initial:
            return torch.tanh(self.weight_init).to(device, dtype).repeat(B, T, 1)
        return torch.zeros(B, T, self.cfg.dim_model, device=device, dtype=dtype)

    def initial_stoch(self, B, T, device, dtype=torch.float32):
        deter = self.initial_deter(B, T, device, dtype)
        _, stoch = self.mask_network.sample(self.deter_to_dec(deter), num_steps=0)
        return stoch.flatten(-2, -1).permute(0, 1, 4, 2, 3).detach()  # (B,T,SV,4,4)

    def get_feat(self, state, detach=False):
        if detach:
            return state["stoch"].detach(), state["deter"].detach()
        return state["stoch"], state["deter"]

    # ---- observe (full causal attention over the window) ---------------- #
    def observe(self, stoch, actions):
        """stoch (B,L,SV,4,4) posterior encodings; actions (B,L,A) (action into each
        state, a_0=0). Returns post, prior dicts. Within-episode windows only."""
        B, L = stoch.shape[:2]
        init = self.initial_stoch(B, 1, stoch.device, stoch.dtype)
        prev_stoch = torch.cat([init, stoch[:, :-1]], dim=1)        # (B,L,SV,4,4)
        x = self.mix(self.encode_stoch(prev_stoch), actions)        # (B,L,512)
        deter = self.transformer(x, causal=True)                    # (B,L,512)
        deter_dec = self.deter_to_dec(deter)
        logits, logits_masked, mask = self.mask_network.train_logits(stoch, deter_dec)
        post = {"stoch": stoch, "deter": deter}
        prior = {"logits": logits, "logits_masked": logits_masked, "mask": mask}
        return post, prior

    def last_deter(self, stoch_hist, action_hist):
        """Deter for the most recent step given history (for online acting). Lists of
        (SV,4,4) stochs and (A,) actions, action_hist[t]=action into state t, a_0=0."""
        ctx = self.cfg.att_context_left
        stoch_hist = stoch_hist[-ctx:]
        action_hist = action_hist[-ctx:]
        T = len(stoch_hist)
        device = stoch_hist[0].device
        stoch_seq = torch.stack(stoch_hist, dim=0).unsqueeze(0)     # (1,T,SV,4,4)
        act_seq = torch.stack(action_hist, dim=0).unsqueeze(0)      # (1,T,A)
        init = self.initial_stoch(1, 1, device, stoch_seq.dtype)
        prev_stoch = torch.cat([init, stoch_seq[:, :-1]], dim=1)
        x = self.mix(self.encode_stoch(prev_stoch), act_seq)
        deter = self.transformer(x, causal=True)
        return deter[:, -1:]                                        # (1,1,512)


# --------------------------------------------------------------------------- #
# Prediction heads (reward / value / continue / policy)
# --------------------------------------------------------------------------- #
class Head(nn.Module):
    """Shared trunk: cnn_proj(stoch) + deter -> MLP -> linear_proj."""

    def __init__(self, cfg, out_dim, zero_init_out=False):
        super().__init__()
        self.feat_size = cfg.stoch_size * cfg.discrete
        self.conv = nn.Conv2d(self.feat_size, cfg.reduced_channels, 1)
        self.norm = ChLayerNorm(cfg.reduced_channels)
        self.act = nn.SiLU()
        self.lin = nn.Linear(4 * 4 * cfg.reduced_channels, cfg.dim_model)
        self.mlp, _ = mlp(2 * cfg.dim_model, [cfg.dim_model] * cfg.num_layers)
        self.out = nn.Linear(cfg.dim_model, out_dim)
        if zero_init_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def trunk(self, feats):
        stoch, deter = feats
        B = stoch.shape[:-3]
        x = self.act(self.norm(self.conv(stoch.reshape(-1, self.feat_size, 4, 4))))
        x = self.lin(x.flatten(1)).reshape(*B, -1)
        x = torch.cat([x, deter], dim=-1)
        return self.out(self.mlp(x))


class RewardHead(Head):
    def __init__(self, cfg):
        super().__init__(cfg, cfg.bins, zero_init_out=True)

    def forward(self, feats):
        return dists.SymLogDiscreteDist(self.trunk(feats))


class ValueHead(Head):
    def __init__(self, cfg):
        super().__init__(cfg, cfg.bins, zero_init_out=True)

    def forward(self, feats):
        return dists.SymLogDiscreteDist(self.trunk(feats))


class ContinueHead(Head):
    def __init__(self, cfg):
        super().__init__(cfg, 1, zero_init_out=True)

    def forward(self, feats):
        return dists.BernoulliDist(self.trunk(feats))


class PolicyHead(Head):
    def __init__(self, cfg, num_actions):
        super().__init__(cfg, num_actions)
        self.uniform_mix = cfg.uniform_mix

    def forward(self, feats):
        return dists.OneHotDist(self.trunk(feats), self.uniform_mix)
