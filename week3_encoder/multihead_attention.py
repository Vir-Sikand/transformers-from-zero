import torch, torch.nn as nn
from week2_attention.attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    """
    Project Q,K,V -> split heads -> scaled-dot attention -> concat -> out proj.
    Acceptance:
      - d_model % num_heads == 0
      - Supports self-attn (x as Q,K,V) and cross-attn (Q!=K,V)
      - Optional mask
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        # TODO: q_proj, k_proj, v_proj, out_proj; store head_dim
        raise NotImplementedError

    def forward(self, x_q, x_kv=None, mask=None):
        raise NotImplementedError
