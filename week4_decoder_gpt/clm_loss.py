import torch
import torch.nn.functional as F

def clm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Next-token prediction (causal LM) loss.
    Args:
      logits: (B,T,V), labels: (B,T)
    Shift labels by one (predict t from < t).
    Return scalar loss.
    """
    raise NotImplementedError
