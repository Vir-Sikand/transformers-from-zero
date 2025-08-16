import torch, torch.nn as nn
from .multihead_attention import MultiHeadAttention
from .feedforward import PositionwiseFFN

class EncoderBlock(nn.Module):
    """Pre-LN: x + MHA(LN(x)) -> x + FFN(LN(x))."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # TODO: ln1, ln2, mha, ffn, dropout
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError
