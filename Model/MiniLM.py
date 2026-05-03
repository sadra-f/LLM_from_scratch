import torch
from torch import nn
from .Transformer import MTransformerBlock



class MiniLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, do_logits_output=True, transformer_heads=4, return_att=False):
        super().__init__()
        
        self._vocab_size = vocab_size
        self._out_dim = vocab_size
        self._model_dim = d_model
        # self._token_dim = int(3 * d_model / 4)
        # self._pos_dim = d_model - self._token_dim
        self._num_layers = num_layers
        self._MAX_POS_VALUE = 256
        self.do_logits_output = do_logits_output
        self._return_att = return_att

        self._embedding = nn.Embedding(self._vocab_size, self._model_dim)
        self._pos_embedding = nn.Embedding(self._MAX_POS_VALUE, self._model_dim)

        self._transformer_blocks = nn.ModuleList([ MTransformerBlock(self._model_dim, transformer_heads) for _ in range(self._num_layers) ])

        self._out_linear = nn.Linear(self._model_dim, self._vocab_size)
        self._out_softmax = nn.Softmax(-1)
        

    def forward(self, x):
        _batch_size, _seq_len = x.shape
        _casual_mask = torch.tril(torch.ones((_seq_len, _seq_len), device=x.device)).unsqueeze(0).expand(_batch_size, -1, -1)
        _positions = torch.arange(_seq_len, device=x.device).expand(_batch_size, _seq_len)
        x = self._embedding(x) +  self._pos_embedding(_positions)
        att_weights = []
        for block in  self._transformer_blocks:
            x, _att = block(x, _casual_mask)
            att_weights.append(_att)
        
        x = self._out_linear(x)
        if not self.do_logits_output:
            x = self._out_softmax(x)
        if self._return_att:
            return x, att_weights
        return x