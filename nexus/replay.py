"""Stream replay for the segment-native WM.

Random-policy Crafter episodes average ~170 steps, so within-episode length-T=256 windows
(emerald_torch.replay) would never sample. This buffer samples length-T windows from the
CONCATENATION of stored episodes, forcing cont=0 at each episode seam so the continue head
still learns resets. (Boundaries are scheduled on the window, independent of episode
seams — a documented N1 simplification; a seam inside a segment is a rare, cont-flagged
event.)
"""

import random

import numpy as np
import torch


class StreamReplay:
    def __init__(self, capacity, num_actions):
        self.capacity = capacity
        self.num_actions = num_actions
        self.episodes = []
        self.num_steps = 0

    def add_episode(self, ep):
        self.episodes.append(ep)
        self.num_steps += len(ep["reward"])
        while self.num_steps > self.capacity and len(self.episodes) > 1:
            old = self.episodes.pop(0)
            self.num_steps -= len(old["reward"])

    def can_sample(self, T):
        return self.num_steps >= T

    def _read(self, start, T, lens, cum):
        img, act, rew, con = [], [], [], []
        remaining, ei = T, int(np.searchsorted(cum, start, "right") - 1)
        off = start - cum[ei]
        while remaining > 0:
            ep, L = self.episodes[ei], lens[ei]
            take = min(remaining, L - off)
            img.append(ep["image"][off:off + take])
            act.append(ep["action"][off:off + take])
            rew.append(ep["reward"][off:off + take])
            c = ep["cont"][off:off + take].copy()
            if off + take == L and remaining - take > 0:
                c[-1] = 0.0                                   # episode seam -> reset
            con.append(c)
            remaining -= take
            ei = (ei + 1) % len(self.episodes)
            off = 0
        return (np.concatenate(img), np.concatenate(act),
                np.concatenate(rew), np.concatenate(con))

    def sample(self, batch_size, T, device):
        lens = [len(ep["reward"]) for ep in self.episodes]
        cum = np.cumsum([0] + lens)
        total = int(cum[-1])
        imgs, acts, rews, cons = [], [], [], []
        for _ in range(batch_size):
            s = random.randint(0, total - T)
            i, a, r, c = self._read(s, T, lens, cum)
            imgs.append(i); acts.append(a); rews.append(r); cons.append(c)
        image = torch.from_numpy(np.stack(imgs)).to(device).float() / 255.0 - 0.5
        action = torch.from_numpy(np.stack(acts)).float().to(device)
        reward = torch.from_numpy(np.stack(rews)).float().to(device)
        cont = torch.from_numpy(np.stack(cons)).float().to(device)
        return {"image": image, "action": action, "reward": reward, "cont": cont}
