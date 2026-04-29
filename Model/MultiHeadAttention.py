import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        if d_model % num_heads != 0:
            #Decided not to handle this which would lead to a messy code!
            raise ValueError(f"model dimension({d_model}) not divisble by requsted number of attention heads ({num_heads})!")
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self._heads_out_dim = d_model / num_heads



    def forward(self, x, mask):
        pass
        #TODO: Implement with single projetion and later splitting!


