import torch
from torch import nn
from .SelfAttention import MSelfAttention
from .MultiHeadAttention import ImplicitMultiHeadAttention
from .FeedForward import MFeedForward

class MTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        self.fnn = MFeedForward(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.atl = ImplicitMultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask):
        attn_out = self.atl(x, mask) # Custom Attention Block/Module
        x = x + self.norm2(attn_out) # add the input to the output ( Residual connection ) and then normalize / changed to normalize previous layer result then add residual data
        fnn_out = self.fnn(x) # extract new features
        x = x + self.norm(fnn_out) # add the input to the output ( Residual connection ) and then normalize / changed to normalize previous layer result then add residual data
        return x