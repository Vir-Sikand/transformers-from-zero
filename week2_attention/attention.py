import torch

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
      Q,K,V: (B,H,T,D)
      mask: broadcastable to (B,1,T,T) or (B,H,T,T); 1=keep, 0=mask
    Returns:
      out: (B,H,T,D), attn: (B,H,T,T)
    Steps:
      scores = Q @ K^T / sqrt(D)
      if mask: set masked to -inf before softmax
      attn = softmax(scores, dim=-1)
      out = attn @ V
    """
    raise NotImplementedError
