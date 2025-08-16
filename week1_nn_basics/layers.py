import torch
import torch.nn as nn

class Linear(nn.Module):
    """
    Custom Linear layer using nn.Parameter.
    Args: in_features, out_features
    Forward: x @ W.T + b
    Acceptance:
      - Registers W(out,in) and b(out) as nn.Parameter
      - Forward returns shape (B,out)
      - Works with autograd (no .detach/.no_grad inside)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # TODO: create self.W and self.b; init W with xavier/kaiming; b zeros
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ReLU(nn.Module):
    """Clamp negatives to 0 via torch.clamp_min."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
