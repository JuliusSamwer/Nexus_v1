"""Week-1 kill experiment (§16.1) — the make-or-break test of the reward-free bet.

  Frozen pretrained EMERALD  →  collect replay  →  train ONLY the skill tier on its
  latents (λ_r=0, EM-style segment↔heads)  →  segment held-out trajectories  →
  LOOK AT THE CUTS.

This isolates the one load-bearing assumption (A2 / H-RF: do dynamics funnels coincide
with task structure?) with the fewest moving parts — no HL actor/critic, no online loop.
EMERALD is loaded and FROZEN; only `skill_tier_parameters()` (jumpy WM heads + skill VQ +
boundary proposer) receive gradients, exactly as the spec's "we segment in its space, we
don't retrain it" (A3).

Deliverables written to <out>/:
  * overlay_*.png   — discovered cuts vs achievement-unlock steps, per held-out episode
  * verdict.json    — the §14 row-1/row-2 verdicts: degeneracy sweep + boundary↔ach F1

    python3 -m nexus.kill_experiment \
        --checkpoint logdir/emerald_smoke/latest.pt --device cuda \
        --out results/kill_week1 --collect_eps 64 --skill_steps 1500
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

from emerald_torch import env as envmod
from emerald_torch.replay import ReplayBuffer
from emerald_torch.train import run_episode
from . import config as config_mod
from . import segment as segmod
from .model import NexusAgent


# --------------------------------------------------------------------------- #
# load + freeze EMERALD                                                        #
# --------------------------------------------------------------------------- #
def load_frozen(cfg, num_actions, ckpt_path, device):
    agent = NexusAgent(cfg, num_actions).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["agent"] if "agent" in ck else ck
    missing, unexpected = agent.step.load_state_dict(sd, strict=True)
    for p in agent.step.parameters():               # FREEZE the step tier
        p.requires_grad_(False)
    agent.step.eval()
    print(f"loaded frozen EMERALD from {ckpt_path} (trained to step "
          f"{ck.get('step', '?')}); step-tier frozen.")
    return agent


# --------------------------------------------------------------------------- #
# collect replay with the frozen policy, keeping per-step achievement counts   #
# --------------------------------------------------------------------------- #
def collect(agent, cfg, device, n_eps, seed):
    env = envmod.CrafterEnv(seed=seed)
    replay = ReplayBuffer(capacity=int(1e6), num_actions=env.num_actions)
    raw = []                                          # keep ach for the overlay/F1
    for i in range(n_eps):
        ep, ret = run_episode(agent.step, env, cfg.step, device, sample=True)
        rep = ep.to_replay()
        replay.add_episode(rep)
        raw.append(rep)
        n_unlock = sum(1 for k in envmod.ACH_NAMES if max(rep["ach"][k]) > 0)
        if (i + 1) % 8 == 0:
            print(f"  collected {i + 1}/{n_eps} eps | last len={len(rep['reward'])} "
                  f"ret={ret:.1f} unlocked={n_unlock}/22")
    return replay, raw


# --------------------------------------------------------------------------- #
# train ONLY the skill tier on frozen latents (Week-1 scope: WM heads, no HL AC)#
# --------------------------------------------------------------------------- #
def train_skill_tier(agent, replay, cfg, device, steps, batch_hl, log_every):
    opt = torch.optim.Adam(agent.skill_tier_parameters(), lr=cfg.hl_lr,
                           eps=cfg.hl_eps, weight_decay=cfg.weight_decay)
    if not replay.can_sample(cfg.T):
        raise SystemExit(f"No episode reaches T={cfg.T}; lower --window or collect more.")
    for it in range(steps):
        img = replay.sample(batch_hl, cfg.T, device)
        hl = agent.encode_hl(img)
        wm_loss, m, _ = agent.hl_world_model_loss(hl)     # segment↔heads EM, λ_r in cfg
        opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.skill_tier_parameters(), cfg.hl_grad_clip)
        opt.step()
        if it % log_every == 0 or it == steps - 1:
            print(f"  [skill {it:>4}/{steps}] term={m['hl_terminal']:.2f} "
                  f"tau={m['hl_tau']:.2f} vq_ppl={m['vq_perplexity']:.1f} "
                  f"seglen={m['mean_seg_len']:.1f} n_segs={m['mean_n_segs']:.1f} "
                  f"prop_bce={m['proposer_bce']:.3f}")
    return agent


# --------------------------------------------------------------------------- #
# analysis helpers                                                            #
# --------------------------------------------------------------------------- #
def _window_batch(rep, off, w, device):
    """A length-w slice of one episode → an encode_hl-ready batch (B=1)."""
    sl = slice(off, off + w)
    img = torch.from_numpy(rep["image"][sl]).float().div(255).sub(0.5)[None].to(device)
    action = torch.from_numpy(rep["action"][sl]).float()[None].to(device)
    reward = torch.from_numpy(rep["reward"][sl]).float()[None].to(device)
    cont = torch.from_numpy(rep["cont"][sl]).float()[None].to(device)
    return {"image": img, "action": action, "reward": reward, "cont": cont}


def unlock_steps(rep, T):
    """Steps at which any achievement's cumulative count increments (the H2 targets)."""
    steps = []
    for k in envmod.ACH_NAMES:
        c = np.asarray(rep["ach"][k][:T])
        for t in np.where(np.diff(c) > 0)[0] + 1:
            steps.append((int(t), k))
    return sorted(steps)


def segment_episode(agent, cfg, rep, device, max_steps=2048):
    """Segment a whole episode by STREAMING the length-T DP window and restarting it at
    the last discovered cut (so windows align to real boundaries — no arbitrary window-edge
    cuts, and in-distribution since the heads were trained on length-T windows). Returns
    (segments, discovered_cuts, T_full, n_forced) — n_forced = max-length segments where the
    DP found no interior cut in a full window (a real 'no boundary in T steps', not an artifact)."""
    T_full = min(len(rep["reward"]), max_steps)
    segs, off, n_forced = [], 0, 0
    while off < T_full:
        w = min(cfg.T, T_full - off)
        if w < 4:                                      # tail too short → extend last seg
            if segs:
                a, b, k = segs[-1]; segs[-1] = (a, T_full, k)
            else:
                segs.append((off, T_full, 0))
            break
        hl = agent.encode_hl(_window_batch(rep, off, w, device))
        seg = segmod.segment(cfg, agent.proposer, agent.skill_enc, agent.jumpy, hl)
        wsegs = seg["segments"][0] or [(0, w, 0)]
        interior = [b for _, b, _ in wsegs if 0 < b < w]
        if w == T_full - off:                          # final window: accept all of it
            segs += [(a + off, b + off, k) for a, b, k in wsegs]; break
        if interior:                                   # restart at the last discovered cut
            last = interior[-1]
            segs += [(a + off, b + off, k) for a, b, k in wsegs if b <= last]
            off += last
        else:                                          # no cut in a full window → max-len seg
            segs.append((off, off + w, wsegs[0][2])); n_forced += 1; off += w
    cuts = sorted({b for _, b, _ in segs if 0 < b < T_full})
    return segs, cuts, T_full, n_forced


def boundary_ach_f1(cuts, unlocks, tol):
    """A cut matches an unlock if within ±tol steps. Returns precision/recall/F1."""
    if not unlocks:
        return None
    up = [s for s, _ in unlocks]
    if not cuts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_cuts": 0, "n_unlocks": len(up), "tol": tol}
    cut_hit = sum(any(abs(c - u) <= tol for u in up) for c in cuts)
    unl_hit = sum(any(abs(c - u) <= tol for c in cuts) for u in up)
    prec = cut_hit / len(cuts)
    rec = unl_hit / len(up)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_cuts": len(cuts), "n_unlocks": len(up), "tol": tol}


def overlay_plot(rep, segs, cuts, unlocks, T, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 3.2))
    reward = np.asarray(rep["reward"][:T])
    ax.plot(np.arange(T), np.cumsum(reward), color="0.6", lw=1.0, label="cum. reward")
    for a, b, k in segs:                               # shaded segments, alternating
        ax.axvspan(a, min(b, T), color="C0" if k % 2 else "C1", alpha=0.08)
    for c in cuts:
        ax.axvline(c, color="C0", lw=1.2, alpha=0.8)
    for s, name in unlocks:
        ax.axvline(s, color="crimson", lw=1.4, ls="--", alpha=0.9)
        ax.text(s, ax.get_ylim()[1], name, rotation=90, va="top", ha="right",
                fontsize=6, color="crimson")
    ax.set_xlabel("env step"); ax.set_ylabel("cum reward")
    ax.set_title(f"Discovered cuts (blue, n={len(cuts)}) vs achievement unlocks "
                 f"(red dashed, n={len(unlocks)})  —  λ_r=0")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def degeneracy_sweep(agent, cfg, reps, device, ell_bars):
    """§14 row-1: re-segment under each ℓ̄ on FROZEN heads; report seg-length stats.
    Degenerate iff almost-all length-1 or one-giant-segment across the sweep."""
    out = {}
    for ell in ell_bars:
        c = copy.copy(cfg); c.ell_bar = ell
        lens = []
        for rep in reps:
            segs, _, _, _ = segment_episode(agent, c, rep, device)
            lens += [b - a for a, b, _ in segs]
        lens = np.asarray(lens) if lens else np.array([0])
        out[str(ell)] = {
            "mean_seg_len": float(lens.mean()),
            "frac_len1": float((lens <= 1).mean()),
            "frac_ge_Lmax": float((lens >= cfg.L_max).mean()),
            "n_segs": int(len(lens)),
        }
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="frozen EMERALD latest.pt")
    ap.add_argument("--out", default="results/kill_week1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", default="base", choices=list(config_mod.PRESETS))
    ap.add_argument("--collect_eps", type=int, default=64)
    ap.add_argument("--skill_steps", type=int, default=1500)
    ap.add_argument("--window", type=int, default=128, help="HL window T (≤ episode len)")
    ap.add_argument("--batch_hl", type=int, default=8)
    ap.add_argument("--n_overlay", type=int, default=6, help="held-out eps to plot")
    ap.add_argument("--tol", type=int, default=5, help="±steps for cut↔unlock match")
    ap.add_argument("--lambda_r", type=float, default=0.0, help="0 = reward-free headline")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    os.makedirs(args.out, exist_ok=True)

    cfg = config_mod.PRESETS[args.configs]()
    cfg.T = args.window
    cfg.step.att_context_left = min(cfg.step.att_context_left, cfg.step.L)
    # λ_r enters ONLY as a thing each segment predicts (gradient never touches cut position)
    cfg.lambda_r = args.lambda_r
    print(f"Week-1 kill experiment | device={device} λ_r={args.lambda_r} "
          f"T={cfg.T} ell_bar={cfg.ell_bar}")

    agent = load_frozen(cfg, envmod.NUM_ACTIONS, args.checkpoint, device)

    print(f"collecting {args.collect_eps} episodes with the frozen policy...")
    replay, raw = collect(agent, cfg, device, args.collect_eps, args.seed)

    # train / held-out split by episode (held-out = richest in unlocks, for a real test)
    order = sorted(range(len(raw)),
                   key=lambda i: -sum(max(raw[i]["ach"][k]) > 0 for k in envmod.ACH_NAMES))
    held = order[:args.n_overlay]
    print(f"training skill tier for {args.skill_steps} steps on frozen latents...")
    train_skill_tier(agent, replay, cfg, device, args.skill_steps,
                     args.batch_hl, log_every=max(1, args.skill_steps // 20))

    # ---- analysis ---- #
    print("segmenting held-out trajectories + overlaying cuts...")
    agent.eval()
    f1s, overlays = [], []
    for rank, idx in enumerate(held):
        rep = raw[idx]
        segs, cuts, T, n_forced = segment_episode(agent, cfg, rep, device)
        unl = unlock_steps(rep, T)
        f1 = boundary_ach_f1(cuts, unl, args.tol)
        png = os.path.join(args.out, f"overlay_{rank}_ep{idx}.png")
        overlay_plot(rep, segs, cuts, unl, T, png)
        overlays.append(os.path.basename(png))
        if f1:
            f1s.append(f1["f1"])
            print(f"  ep{idx}: cuts={f1['n_cuts']} unlocks={f1['n_unlocks']} "
                  f"P={f1['precision']:.2f} R={f1['recall']:.2f} F1={f1['f1']:.2f}")

    deg = degeneracy_sweep(agent, cfg, [raw[i] for i in held], device, [25, 50, 100])

    verdict = {
        "checkpoint": args.checkpoint, "lambda_r": args.lambda_r,
        "boundary_achievement_f1": {
            "mean": float(np.mean(f1s)) if f1s else None,
            "per_episode": f1s, "tol": args.tol,
        },
        "degeneracy_sweep_ell_bar": deg,         # §14 row-1
        "overlays": overlays,
        "notes": "F1 is a DIAGNOSTIC (H2), not the success criterion. Degeneracy: "
                 "look for NOT-almost-all frac_len1≈1 or frac_ge_Lmax≈1 across ℓ̄.",
    }
    with open(os.path.join(args.out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nDONE → {args.out}/verdict.json + {len(overlays)} overlays")
    print(f"  boundary↔achievement F1 (mean) = {verdict['boundary_achievement_f1']['mean']}")
    print(f"  degeneracy sweep = {json.dumps(deg, indent=2)}")


if __name__ == "__main__":
    main()
