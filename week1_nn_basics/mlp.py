import torch
import torch.nn as nn
from .layers import Linear, ReLU

class MLP(nn.Module):
    """
    Multi-Layer Perceptron using your Linear/ReLU.
    Flow: Linear -> ReLU -> [Linear -> ReLU]* -> Linear(out_dim)
    Acceptance: (B,in_dim)->(B,out_dim), trains in run_overfit.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_hidden: int = 1, dropout: float = 0.0):
        super().__init__()
        # TODO: build nn.Sequential/layers list from your modules
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
