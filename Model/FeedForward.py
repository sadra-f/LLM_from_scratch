import torch
from torch import nn



class MFeedForward(nn.Module):
    def __init__(self, inp_dim, outp_dim):
        super().__init__()
        self._internal_dim = 2 * inp_dim
        self._net = nn.Sequential(
            nn.Linear(inp_dim, self._internal_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self._net(x)