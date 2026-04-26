import torch
from torch import nn



class MFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._internal_dim = 4 * dim
        self._net = nn.Sequential(
            nn.Linear(dim, self._internal_dim),
            nn.ReLU(),
            nn.Linear(self._internal_dim, dim), 
            # Won't End with an activation function so that the new features are not lost + ReLU removes all negative features which hold singal for the model
        )
    def forward(self, x):
        return self._net(x)