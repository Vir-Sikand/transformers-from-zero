import torch

def sinusoidal_positional_encoding(T: int, d_model: int) -> torch.Tensor:
    """
    Return (T, d_model) sinusoidal table per Vaswani §3.5.
    Acceptance: shape (T,d_model), dtype float32/float64.
    """
    raise NotImplementedError
