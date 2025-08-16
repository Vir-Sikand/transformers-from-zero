import torch, torch.nn as nn
from week3_encoder.multihead_attention import MultiHeadAttention
from week3_encoder.feedforward import PositionwiseFFN

class DecoderBlock(nn.Module):
    """
    Pre-LN decoder block:
      x -> x + SelfMHA(LN(x), causal_mask)
      x -> x + CrossMHA(LN(x), enc_out)
      x -> x + FFN(LN(x))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # TODO: ln1, ln2, ln3, self_attn, cross_attn, ffn, dropout
        raise NotImplementedError

    def forward(self, x, enc_out=None, self_mask=None, cross_mask=None):
        raise NotImplementedError
