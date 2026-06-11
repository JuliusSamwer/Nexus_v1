"""NexusAgent — ties the EMERALD step tier (reused verbatim) to the new skill tier.

v1 scope:
  * Step tier (Stage 1): EMERALD's world model + actor-critic, used exactly as-is via
    `emerald_torch`. **Documented v1 gap:** the step actor is NOT yet skill-conditioned
    (the doc's "+1 input"); that closes the Stage-4 loop and is the one deferred piece.
  * Skill tier (Stages 2-3): segmentation (segment.py) → jumpy WM training (terminal
    MaskGIT + Σr + τ + continue + Hₙ), skill VQ, boundary-proposer imitation, and a
    compact HL actor-critic in jumpy imagination (γ^τ).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from emerald_torch.model import EmeraldAgent
from . import segment as segmod
from .jumpy import JumpyWM
from .segment import BoundaryProposer
from .skill import SkillEncoder


def terminal_loss(jumpy, c, z_term):
    """CE of the terminal latent under the jumpy MaskGIT (direct + masked). c (B,N,H),
    z_term (B,N,SV,4,4). Returns (direct (B,N), masked (B,N))."""
    cond = jumpy.cond(c)
    logits, logits_masked, mmask = jumpy.terminal.train_logits(z_term, cond)
    S, V = jumpy.terminal.S, jumpy.terminal.V
    tgt = z_term.permute(0, 1, 3, 4, 2).reshape(*logits.shape[:4], S, V)
    direct = -(tgt * torch.log_softmax(logits, dim=-1)).sum(-1).sum((-3, -2, -1))
    if logits_masked is not None:
        ce = -(tgt * torch.log_softmax(logits_masked, dim=-1)).sum(-1)        # (B,N,4,4,S)
        m = mmask.float()
        masked = (ce * m).sum((-3, -2, -1)) / (m.sum((-3, -2, -1)) + 1e-8)
    else:
        masked = torch.zeros_like(direct)
    return direct, masked


class NexusAgent(nn.Module):
    def __init__(self, cfg, num_actions):
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        self.step = EmeraldAgent(cfg.step, num_actions)            # EMERALD step tier
        self.skill_enc = SkillEncoder(cfg, num_actions)
        self.jumpy = JumpyWM(cfg, num_actions)
        self.proposer = BoundaryProposer(cfg, num_actions)
        self.register_buffer("perc_low", torch.tensor(0.0))
        self.register_buffer("perc_high", torch.tensor(0.0))

    # parameter groups
    def step_wm_parameters(self):
        return self.step.wm_parameters()

    def skill_tier_parameters(self):
        mods = [self.skill_enc, self.jumpy, self.proposer]
        return [p for m in mods for p in m.parameters()]

    # ---- latents for the HL stream (run EMERALD WM observe, no grad) ------ #
    @torch.no_grad()
    def encode_hl(self, batch):
        enc = self.step.encoder(batch["image"])
        post, _ = self.step.tssm.observe(enc["stoch"], batch["action"])
        return {"stoch": post["stoch"].detach(), "deter": post["deter"].detach(),
                "action": batch["action"], "reward": batch["reward"], "cont": batch["cont"]}

    # ---- assemble padded jump tensors from per-element segments ----------- #
    def _assemble(self, segs, hl):
        stoch, deter, reward, cont = hl["stoch"], hl["deter"], hl["reward"], hl["cont"]
        B = stoch.shape[0]
        device = stoch.device
        SV = stoch.shape[2]
        Nmax = max(max(len(s) for s in segs), 1)
        z0 = torch.zeros(B, Nmax, SV, 4, 4, device=device)
        zt = torch.zeros(B, Nmax, SV, 4, 4, device=device)
        h0 = torch.zeros(B, Nmax, self.cfg.step.dim_model, device=device)
        a_idx = torch.zeros(B, Nmax, dtype=torch.long, device=device)
        b_idx = torch.zeros(B, Nmax, dtype=torch.long, device=device)
        sigmar = torch.zeros(B, Nmax, device=device)
        tau = torch.ones(B, Nmax, device=device)
        hlcont = torch.zeros(B, Nmax, 1, device=device)
        mask = torch.zeros(B, Nmax, device=device)
        T = reward.shape[1]
        for b in range(B):
            for n, (a, e, _k) in enumerate(segs[b]):
                z0[b, n] = stoch[b, a]
                zt[b, n] = stoch[b, min(e - 1, T - 1)]
                h0[b, n] = deter[b, a]
                a_idx[b, n] = a; b_idx[b, n] = e
                sigmar[b, n] = reward[b, a:e].sum()
                tau[b, n] = e - a
                hlcont[b, n, 0] = cont[b, a:e].prod()
                mask[b, n] = 1.0
        return dict(z0=z0, zt=zt, h0=h0, a_idx=a_idx, b_idx=b_idx,
                    sigmar=sigmar, tau=tau, hlcont=hlcont, mask=mask)

    # ---- Stage 2+3: segment, then train the jumpy WM + skill + proposer --- #
    def hl_world_model_loss(self, hl):
        cfg = self.cfg
        seg = segmod.segment(cfg, self.proposer, self.skill_enc, self.jumpy, hl)
        J = self._assemble(seg["segments"], hl)
        mask = J["mask"]
        msum = mask.sum().clamp(min=1)

        def mmean(x):
            return (x * mask).sum() / msum

        # skills for the chosen segments (train-mode VQ updates the codebook)
        g = self.skill_enc.features(hl["stoch"], hl["action"])
        ps = self.skill_enc.prefix_sums(g)
        seg_feat = (ps.gather(1, J["b_idx"].unsqueeze(-1).expand(-1, -1, g.shape[-1]))
                    - ps.gather(1, J["a_idx"].unsqueeze(-1).expand(-1, -1, g.shape[-1]))) \
            / (J["b_idx"] - J["a_idx"]).clamp(min=1).unsqueeze(-1)
        z_q, k_idx, commit_loss, perplexity = self.skill_enc.code_of(seg_feat)
        k_emb = z_q                                                # straight-through code embed

        # Hₙ recurrence + context
        z_emb = self.jumpy.embed_z(J["z0"])
        h_emb = self.jumpy.h_proj(J["h0"])
        tokens = self.jumpy.jump_token(z_emb, k_emb, h_emb, drop_h=True)
        Hn = self.jumpy.roll_Hn(tokens)
        c = self.jumpy.context(Hn, z_emb, k_emb)

        # outcome losses (masked over valid jumps)
        direct, masked = terminal_loss(self.jumpy, c, J["zt"])
        heads = self.jumpy.outcome_heads(c)
        loss_term = mmean(direct) + mmean(masked)
        loss_sigmar = mmean(-heads["sigma_r"].log_prob(J["sigmar"].unsqueeze(-1)).squeeze(-1))
        loss_tau = mmean(-heads["tau"].log_prob(J["tau"].unsqueeze(-1)).squeeze(-1))
        loss_cont = mmean(-heads["continue"].log_prob(J["hlcont"]).squeeze(-1))

        # boundary proposer imitates the DP posterior marginals
        blogits = self.proposer(hl["stoch"], hl["action"])
        loss_prop = F.binary_cross_entropy_with_logits(blogits, seg["marg_target"])

        loss = (loss_term + loss_sigmar + loss_tau + loss_cont + commit_loss + loss_prop)
        metrics = {
            "hl_terminal": loss_term.item(), "hl_sigmar": loss_sigmar.item(),
            "hl_tau": loss_tau.item(), "hl_cont": loss_cont.item(),
            "vq_commit": commit_loss.item(), "vq_perplexity": perplexity.item(),
            "proposer_bce": loss_prop.item(),
            "mean_seg_len": seg["stats"]["mean_seg_len"],
            "mean_n_segs": seg["stats"]["mean_n_segs"],
        }
        # detached starts for HL actor-critic imagination
        starts = {"Hn": Hn.detach(), "z0": J["z0"].detach(), "mask": mask.detach()}
        return loss, metrics, starts

    # ---- Stage 3: HL actor-critic in jumpy imagination (γ^τ) -------------- #
    def hl_actor_critic_loss(self, starts):
        cfg = self.cfg
        Hn0 = starts["Hn"]; z0 = starts["z0"]; mask = starts["mask"]
        B, N = mask.shape
        # flatten valid starts
        flat = mask.reshape(-1) > 0
        if flat.sum() == 0:
            zero = torch.zeros((), device=mask.device, requires_grad=True)
            return zero, zero, {"hl_actor": 0.0, "hl_value": 0.0}
        H = Hn0.reshape(-1, Hn0.shape[-1])[flat][:, None, :]      # (P,1,H)
        z = z0.reshape(-1, *z0.shape[2:])[flat][:, None]          # (P,1,SV,4,4)

        with torch.no_grad():
            z_embs, H_seq, actions, taus, rewards = [], [], [], [], []
            cur_z, cur_H = z, H
            tok_seq = []
            for _ in range(cfg.hl_H):
                z_emb = self.jumpy.embed_z(cur_z)
                a = self.jumpy.actor_dist(cur_H, z_emb).sample()           # (P,1,K)
                k_emb = self.skill_enc.vq.lookup(a.argmax(-1))             # (P,1,code)
                c = self.jumpy.context(cur_H, z_emb, k_emb)
                heads = self.jumpy.outcome_heads(c)
                _, s_oh = self.jumpy.terminal.sample(self.jumpy.cond(c), cfg.jumpy_decoding_steps)
                nz = s_oh.flatten(-2, -1).permute(0, 1, 4, 2, 3)
                z_embs.append(z_emb); H_seq.append(cur_H); actions.append(a)
                taus.append(heads["tau"].mode()); rewards.append(heads["sigma_r"].mode())
                tok = self.jumpy.jump_token(z_emb, k_emb, torch.zeros_like(z_emb), drop_h=False)
                tok_seq.append(tok)
                cur_H = self.jumpy.roll_Hn(torch.cat(tok_seq, dim=1))[:, -1:]
                cur_z = nz
            z_embs = torch.cat(z_embs, dim=1); H_seq = torch.cat(H_seq, dim=1)
            actions = torch.cat(actions, dim=1)
            taus = torch.cat(taus, dim=1).clamp(min=1.0)                   # (P,Hh,1)
            rewards = torch.cat(rewards, dim=1)
            values = self.jumpy.critic_dist(H_seq, z_embs).mode()
            disc = cfg.gamma ** taus                                       # γ^τ
            # λ-returns over jumps
            interm = rewards + disc * (1 - cfg.lambda_td) * values
            vals = [values[:, -1]]
            for t in reversed(range(interm.shape[1] - 1)):
                vals.append(interm[:, t] + disc[:, t] * cfg.lambda_td * vals[-1])
            returns = torch.stack(list(reversed(vals)), dim=1)            # (P,Hh,1)
            adv = (returns - values).squeeze(-1)

        # actor
        dist = self.jumpy.actor_dist(H_seq.detach(), z_embs.detach())
        logp = dist.log_prob(actions.detach())
        ent = dist.entropy()
        actor_loss = -((logp * adv.detach() + cfg.eta_entropy * ent)).mean()
        # critic
        vdist = self.jumpy.critic_dist(H_seq.detach(), z_embs.detach())
        value_loss = -vdist.log_prob(returns.detach()).mean()
        metrics = {"hl_actor": actor_loss.item(), "hl_value": value_loss.item(),
                   "hl_returns": returns.mean().item()}
        return actor_loss, value_loss, metrics
