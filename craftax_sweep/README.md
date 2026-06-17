# Craftax-Classic capacity sweep (EMERALD-JAX)

Scale-down capacity sweep on the fast JAX/Flax EMERALD port: train a **full** and a
**tiny** world model on Craftax-Classic under an *identical* data/collection regime, so
only WM capacity differs. Tests whether a weaker WM (closer to the real-world regime
where the model is weaker than the world) makes the conditional-divergence / learned-trust
signal stronger. A flat/null result is itself informative.

## Arms (defined in `emerald_jax/config.py`)
| arm  | preset                      | params | what changes |
|------|-----------------------------|--------|--------------|
| full | `craftax_fast()`            | ~32.6M | baseline capacity |
| tiny | `craftax_fast_tiny_wm()`    | ~5.4M  | dim_model 512→128, dim_cnn 32→16, heads 8→4, blocks_trans 4→2, blocks_mask 2→1, reduced 128→64 |

Both keep the **same** num_envs=64, batch_size=32, L=64, H=15, capacity, loss scales,
lrs, gamma, and the latent target (stoch_size=32, discrete=32). Keep `--gpc` (and any
`--num-envs/--batch-size/--capacity` saturation overrides) **identical across arms**.

## Run on a pod (4090 / 5090)
```bash
bash craftax_sweep/run_pod.sh                  # full then tiny, 10M each, gpc=1, nohup
TOTAL=5000000 bash craftax_sweep/run_pod.sh    # 5M instead
ARMS="tiny" bash craftax_sweep/run_pod.sh      # one arm only
```
Survives SSH disconnect; re-run to `--resume` from `latest.pkl`.

## Or one arm directly
```bash
python craftax_sweep/train_craftax_sweep.py --arm full --total 10_000_000 \
    --ckpt-root /workspace/ckpts --resume
```

## Outputs (per arm, under `<ckpt-root>/<arm>/`)
- `step_<N>.pkl` — params-only milestone checkpoints (every `--milestone-every`, default 1M).
  Read directly by `harness/critic/` (`{"params", "cfg", ...}`); rebuild the agent from `blob["cfg"]`.
- `latest.pkl` — full state incl. optimizer, for `--resume`.
- `metrics.jsonl` — per-`--log-every` training losses + throughput.
- `eval.jsonl` — per-`--eval-every` Crafter score + per-achievement rates.

The imagined-rollout divergence eval (policy-KL over horizon, reward MAE, token-acc vs
depth) is **not** run during training — reconstruct it offline from the milestones.
