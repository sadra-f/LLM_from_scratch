import torch
from torch import nn
from .SelfAttention import MSelfAttention
from .FeedForward import MFeedForward

class EplicitMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        if d_model % num_heads != 0:
            #Decided not to handle this which would lead to a messy code!
            raise ValueError(f"model dimension({d_model}) not divisble by requsted number of attention heads ({num_heads})!")
        super().__init__()
        self._d_model = d_model
        self._num_heads = num_heads
        self._heads_out_dim = d_model / num_heads
        self._heads_in_dim = d_model
        self._proj_layer = nn.Linear(d_model, d_model)
        self._heads = nn.ModuleList([ MSelfAttention(self._heads_in_dim, self._heads_out_dim) for _ in range(self._num_heads) ])
        #Concat head outputs
        self._out_Linear = nn.Linear(d_model, d_model)




    def forward(self, x, mask):
        # _batch, _seq, _model_d = x.shape
        x = self._proj_layer(x)
        # att_inps = torch.transpose(x.view(_batch, _seq, self._num_heads, -1), 2, 1) # from (b, t, d) to (b, t, h, d) to (b, h, t, d) ! it was conceptually wrong this way heads won't have access to the full info!
        tmp_res = []
        weights = []
        for head in self._heads:
            att_res, weights = head(x, mask)
            tmp_res.append(att_res)
            weights.append(weights)
        x = torch.concat(tmp_res, dim=-1)
        x = self._out_Linear(x)

        return x, weights
    


class ImplicitMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        if d_model % num_heads != 0:
            #Decided not to handle this which would lead to a messy code!
            raise ValueError(f"model dimension({d_model}) not divisble by requsted number of attention heads ({num_heads})!")
        super().__init__()
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_att = d_model / num_heads
        self._att_scale = torch.sqrt(torch.tensor(self._d_att))
        self._linear_proj = nn.Linear(d_model, d_model)
        self.q_w = nn.Linear(d_model, d_model)
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)
        self._att_dropout = nn.Dropout(0.1)
        #concat!
        self._linear_out = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        _batch, _seq, _dim = x.shape
        x = self._linear_proj(x) # (_batch, _seq, _dim)
        Q = self.q_w(x).reshape(_batch, _seq, self._num_heads, -1).transpose(2,1) # from (_batch, _seq, _dim) to (_batch, _seq, num_heads, d_att) to (_batch, num_heads, _seq, d_att)
        K = self.k_w(x).reshape(_batch, _seq, self._num_heads, -1).transpose(2,1) # from (_batch, _seq, _dim) to (_batch, _seq, num_heads, d_att) to (_batch, num_heads, _seq, d_att)
        V = self.v_w(x).reshape(_batch, _seq, self._num_heads, -1).transpose(2,1) # from (_batch, _seq, _dim) to (_batch, _seq, num_heads, d_att) to (_batch, num_heads, _seq, d_att)
        QKt = torch.matmul(Q, K.transpose(-2, -1)) / self._att_scale # (_batch, num_heads, _seq, _seq)
        QKt.masked_fill_(mask == 0, -1e10)
        att_weights = torch.softmax(QKt, dim=-1)
        if self.training:
            att_weights = self._att_dropout(att_weights)
        pre_x = torch.matmul(att_weights, V) # (_batch, num_heads, _seq, d_att)
        x = pre_x.transpose(2,1).reshape(_batch, _seq, -1) # from (_batch, num_heads, _seq, d_att) to (_batch, _seq, num_heads, d_att) to (_batch, _seq, _dim)
        x = self._linear_out(x)
        
        return x, att_weights

