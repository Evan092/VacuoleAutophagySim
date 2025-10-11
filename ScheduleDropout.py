import torch
import torch.nn as nn
import math

class ScheduledDropout(nn.Module):
    def __init__(self, p_start=0.9, p_end=0.0, T_max=80, mode='cosine', epoch=0):
        super().__init__()
        self.p_start = p_start
        self.p_end   = p_end
        self.T_max   = T_max
        self.mode    = mode
        self.epoch   = epoch
        self.p = 0

    @property
    def value(self):
        return self.p

    def forward(self, x):
        # Compute current dropout probability
        if self.mode == 'linear':
            self.p = self.p_start - (self.p_start - self.p_end) * (self.epoch / self.T_max)
        else:  # cosine
            self.p = self.p_end + 0.5 * (self.p_start - self.p_end) * (1 + math.cos(math.pi * self.epoch / self.T_max))
        return nn.functional.dropout(x, p=self.p, training=self.training)

    def setEpoch(self, epoch):
        self.epoch = epoch

    def step(self):
        """Call once per epoch."""
        self.epoch = min(self.epoch + 1, self.T_max)