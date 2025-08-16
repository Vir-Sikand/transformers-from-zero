import torch, torch.nn as nn

class PositionwiseFFN(nn.Module):
    """d_model -> d_ff -> d_model with GELU and optional dropout."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # TODO: two Linear layers + GELU + Dropout
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
