import torch
from torch import nn



class MFeedForward(nn.Module):
    def __init__(self, inp_dim, output_dim):
        super().__init__()
        self._internal_dim = 4 * inp_dim
        self._net = nn.Sequential(
            nn.Linear(inp_dim, self._internal_dim),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(self._internal_dim, output_dim), 
            # Won't End with an activation function so that the new features are not lost + ReLU removes all negative features which hold singal for the model
        )
    def forward(self, x):
        return self._net(x)