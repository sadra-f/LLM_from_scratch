import torch
from torch import nn
import math

class MSelfAttention(nn.Module):
    def __init__(self, inp_dim, outp_dim):
        super().__init__()
        self._internal_dim = outp_dim
        self.Q_W = nn.Linear(inp_dim, self._internal_dim)
        self.K_W = nn.Linear(inp_dim, self._internal_dim)
        self.V_W = nn.Linear(inp_dim, self._internal_dim)
        self._dim_log = math.sqrt(self._internal_dim)
        self.dim_adj = nn.Linear(self._internal_dim, outp_dim)

    def forward(self, x):
        Q = self.Q_W(x) # from (batch, seq_len, inp_dim) to (batch, seq_len, internal_dim)
        K = self.K_W(x)
        V = self.V_W(x)
        qkt = torch.matmul(Q, torch.transpose(K, -2, -1)) # given (batch, seq, dim) transpose to (batch, dim, seq) for K
        qkt = qkt / self._dim_log
        weights = torch.softmax(qkt, dim=-1) # gotta provide the dim for softmax not get consfued!
        transformed = torch.matmul(weights, V) # its a matmul not element-wise
        return self.dim_adj(transformed)