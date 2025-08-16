import torch

def causal_mask(T: int) -> torch.Tensor:
    """Return lower-triangular mask (T,T) with 1 allowed, 0 masked."""
    raise NotImplementedError

def padding_mask(lengths, max_len=None) -> torch.Tensor:
    """
    From 1D lengths -> (B,1,1,T) mask with 1 for tokens, 0 for pads.
    """
    raise NotImplementedError
