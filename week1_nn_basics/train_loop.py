import torch
from week0_tensors.tensors_autograd import manual_cross_entropy

def train_one_epoch(model, dataloader, optimizer, device='cpu', clip_grad=None, log_every=100):
    """
    Manual training loop using your manual_cross_entropy.
    Returns average loss (float).
    """
    raise NotImplementedError

@torch.no_grad()
def eval_loss(model, dataloader, device='cpu'):
    """Return average loss on a validation dataloader."""
    raise NotImplementedError
