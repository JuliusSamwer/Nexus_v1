"""Instrument 2 — Open-loop imagination-fidelity curve.

THE paper's opening figure. From real observed states, roll the world model forward k
steps under the *logged* actions and measure how fast imagination diverges from ground
truth as a function of k. The architecture's job (Phase 2) is to bend this curve.

Requires a loaded EMERALD checkpoint on a CUDA GPU — the local M3 Max has none, so this
module documents the exact calls and leaves a single `TODO(checkpoint)` seam.

Hook points (third_party/EMERALD):
  - Seed real states:   TSSM.observe(...)                      nnet/modules/emerald/tssm.py:193
  - Roll forward k:     TSSM.imagine(p_net, prev_state,        nnet/modules/emerald/tssm.py:219
                          img_steps=k, actions=<logged>,
                          return_stoch_steps=True)
  - Decode to frames:   the decoder used in the world-model forward (emerald.py).

Plan:
  1. Sample a batch of real trajectories from the replay buffer (length > k_max + context).
  2. observe() the first `context` steps to get a real posterior `prev_state`.
  3. imagine() forward k steps under the trajectory's logged actions (open loop — NO
     posterior correction). Keep the imagined latents at each k.
  4. Compare imagined latent at step k to the real posterior latent at the same env step:
       - latent_token_error(k): fraction of HxW*stoch tokens whose argmax disagrees,
         and/or categorical KL(post || imagined_prior).
       - decoded_frame_error(k): MSE (and optionally LPIPS) between decoded imagined
         frame and the real frame.
  5. Average over the batch → error vs k. Write results/open_loop_fidelity.json.
"""

import argparse
import json
import os


def latent_token_error(imagined_stoch, real_stoch):
    """Fraction of latent tokens whose argmax disagrees. Shapes (B, H, W, stoch, discrete)."""
    # import torch  # available where the model runs
    # imag = imagined_stoch.argmax(-1)
    # real = real_stoch.argmax(-1)
    # return (imag != real).float().mean().item()
    raise NotImplementedError("Run on the GPU box with real tensors from observe()/imagine().")


def open_loop_curve(model, replay_buffer, k_max=120, context=5, num_batches=8):
    """Return [{k, latent_error, frame_error}] averaged over sampled trajectories.

    TODO(checkpoint): instantiate EMERALD and load a trained checkpoint, e.g.

        import nnet
        model = nnet.models.EMERALD()
        model.load(<path-to-callbacks/run_name>)      # see emerald.py:323
        model.set_replay_buffer(replay_buffer)

    then for each sampled trajectory:
        posts, _ = model.tssm.observe(states[:, :context], prev_actions, is_firsts)
        prev_state = {key: v[:, -1:] for key, v in posts.items()}
        img = model.tssm.imagine(model.actor, prev_state, img_steps=k_max,
                                 actions=logged_actions, return_stoch_steps=True)
        # compare img['stoch'][:, k] to the real posterior at env-step (context + k)
    """
    raise NotImplementedError(
        "Needs a loaded EMERALD checkpoint + replay buffer on CUDA. See the docstring for "
        "the exact observe()/imagine() calls (tssm.py:193 / :219)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=False,
                    help="Path under callbacks/<run_name> to a trained EMERALD checkpoint.")
    ap.add_argument("--k-max", type=int, default=120)
    ap.add_argument("--context", type=int, default=5)
    ap.add_argument("--out", default="results/open_loop_fidelity.json")
    args = ap.parse_args()

    if not args.checkpoint:
        print(__doc__)
        print("\nNo --checkpoint given. This instrument runs on the GPU box; "
              "see the docstring for the exact observe()/imagine() calls.")
        return

    # TODO(checkpoint): load model + replay buffer, then:
    curve = open_loop_curve(model=None, replay_buffer=None,
                            k_max=args.k_max, context=args.context)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
