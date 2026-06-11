"""Crafter env wrapper + dreamerv3-torch-compatible logging (so harness/eval_overlay.py
reads EMERALD's eval output directly)."""

import json
import os
import uuid

import numpy as np

import crafter

ACH_NAMES = list(crafter.constants.achievements)   # 22 names
NUM_ACTIONS = 17


class CrafterEnv:
    """Returns CHW uint8 images. info carries cumulative achievement counts + discount."""

    def __init__(self, seed=0, reward=True):
        self._env = crafter.Env(size=(64, 64), reward=reward, seed=seed)

    @property
    def num_actions(self):
        return self._env.action_space.n

    def reset(self):
        img = self._env.reset()                                    # (64,64,3) uint8
        info = {"achievements": {k: 0 for k in ACH_NAMES}, "discount": 1.0}
        return img.transpose(2, 0, 1), info

    def step(self, action):
        img, reward, done, info = self._env.step(int(action))
        return img.transpose(2, 0, 1), float(reward), bool(done), info


def onehot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def save_eval_episode(directory, ep):
    """Write one eval episode as a dreamer-format .npz (HWC images + log_achievement_*)."""
    os.makedirs(directory, exist_ok=True)
    T = len(ep["reward"])
    image_hwc = np.stack(ep["image"]).transpose(0, 2, 3, 1)        # (T,64,64,3) uint8
    is_first = np.zeros(T, bool); is_first[0] = True
    is_last = np.zeros(T, bool); is_last[-1] = True
    is_terminal = np.array(ep["is_terminal"], bool)
    reward = np.array(ep["reward"], np.float32)
    out = {
        "image": image_hwc,
        "is_first": is_first, "is_last": is_last, "is_terminal": is_terminal,
        "reward": reward, "discount": (1.0 - is_terminal).astype(np.float32),
        "log_reward": reward,
        "action": np.stack(ep["action"]).astype(np.float32),
    }
    for k in ACH_NAMES:
        out[f"log_achievement_{k}"] = np.array(ep["ach"][k], np.int32)
    fname = f"{uuid.uuid4().hex}-{T}.npz"
    np.savez_compressed(os.path.join(directory, fname), **out)


class MetricsWriter:
    """Appends one JSON row per call to <logdir>/metrics.jsonl (dreamer-compatible)."""

    def __init__(self, logdir):
        os.makedirs(logdir, exist_ok=True)
        self.path = os.path.join(logdir, "metrics.jsonl")

    def write(self, step, scalars):
        row = {"step": int(step)}
        row.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in scalars.items()})
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
