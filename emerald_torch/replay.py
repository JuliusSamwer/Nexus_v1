"""Minimal episode replay buffer. Stores whole episodes; samples within-episode windows
of length L (so a window never crosses an episode boundary — keeps TSSM.observe's single
init-prepend correct without a segment mask)."""

import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, num_actions):
        self.capacity = capacity
        self.num_actions = num_actions
        self.episodes = []          # each: dict of np arrays, length T_i
        self.num_steps = 0

    def add_episode(self, ep):
        self.episodes.append(ep)
        self.num_steps += len(ep["reward"])
        while self.num_steps > self.capacity and len(self.episodes) > 1:
            old = self.episodes.pop(0)
            self.num_steps -= len(old["reward"])

    def can_sample(self, L):
        return any(len(ep["reward"]) >= L for ep in self.episodes)

    def sample(self, batch_size, L, device):
        eligible = [ep for ep in self.episodes if len(ep["reward"]) >= L]
        out = {k: [] for k in ("image", "action", "reward", "cont")}
        for _ in range(batch_size):
            ep = random.choice(eligible)
            T = len(ep["reward"])
            i = random.randint(0, T - L)
            out["image"].append(ep["image"][i:i + L])
            out["action"].append(ep["action"][i:i + L])
            out["reward"].append(ep["reward"][i:i + L])
            out["cont"].append(ep["cont"][i:i + L])
        image = torch.from_numpy(np.stack(out["image"])).to(device)           # (B,L,3,64,64) uint8
        image = image.float() / 255.0 - 0.5
        action = torch.from_numpy(np.stack(out["action"])).float().to(device)  # (B,L,A)
        reward = torch.from_numpy(np.stack(out["reward"])).float().to(device)  # (B,L)
        cont = torch.from_numpy(np.stack(out["cont"])).float().to(device)      # (B,L)
        return {"image": image, "action": action, "reward": reward, "cont": cont}
