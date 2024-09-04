import torch

import numpy as np


class Buffer():

    def __init__(self, size, obs_dim, action_sequence_length, act_dim, device=None):
        self.bufs = (
            np.zeros((size, obs_dim), dtype=np.float32),
            np.zeros((size, action_sequence_length, act_dim), dtype=np.float32),
        )

        self.device = device
        self.size = size
        self.pos = 0
        self.full = False

    def add(self, *args):
        for i, buf in enumerate(self.bufs):
            buf[self.pos] = args[i]

        self.pos += 1
        if self.pos == self.size:
            self.pos = 0
            self.full = True

    def sample(self, batch_size):
        n = self.pos if not self.full else self.size
        i = np.random.randint(n, size=batch_size)
        return tuple(map(lambda a: torch.as_tensor(a[i], device=self.device), self.bufs))