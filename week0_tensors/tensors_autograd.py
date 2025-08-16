"""
Week 0 — TensorLab + manual softmax/cross-entropy.

Use ONLY PyTorch tensor operations. Keep code autograd-friendly where relevant.
"""
import torch

class TensorLab:
    """
    Tiny tensor helpers. Implement each function per its docstring.
    """

    @staticmethod
    def create_range(n: int, device=None, dtype=None):
        """
        Return 1D tensor [0,1,...,n-1].
        Acceptance: shape (n,), correct dtype/device when provided.
        """
        raise NotImplementedError

    @staticmethod
    def reshape_to(x: torch.Tensor, shape: tuple):
        """
        Return x reshaped to `shape` with .reshape().
        Acceptance: no copy assumptions; shape must equal `shape`.
        """
        raise NotImplementedError

    @staticmethod
    def permute_to(x: torch.Tensor, dims: tuple):
        """
        Return x permuted to `dims` with .permute().
        """
        raise NotImplementedError

    @staticmethod
    def broadcast_add(vec: torch.Tensor, mat: torch.Tensor):
        """
        Add a 1D tensor `vec` to each row of 2D `mat` via broadcasting.
        vec: (C,), mat: (R,C) -> out: (R,C)
        """
        raise NotImplementedError

    @staticmethod
    def masked_fill_above(x: torch.Tensor, threshold: float, value: float):
        """
        Return copy of x where elements > threshold are set to `value`.
        Acceptance: original x not modified; dtype preserved.
        """
        raise NotImplementedError

    @staticmethod
    def row_normalize(x: torch.Tensor, eps: float = 1e-9):
        """
        Row-wise L1 normalize: x / (sum along dim=1 + eps).
        Acceptance: each row sums to ~1 (within 1e-5).
        """
        raise NotImplementedError

    @staticmethod
    def batched_matmul(A: torch.Tensor, B: torch.Tensor):
        """
        Batched matmul: A (B,N,M) @ B (B,M,K) -> (B,N,K).
        Use torch.matmul or the @ operator.
        """
        raise NotImplementedError

    @staticmethod
    def to_device(x: torch.Tensor, device: str):
        """
        Move tensor to 'cpu' or 'cuda' (if available).
        Acceptance: returns a tensor on the requested device when available.
        """
        raise NotImplementedError


def manual_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax along `dim`:
      shift = logits - logits.max(dim, keepdim=True).values
      exps = shift.exp()
      probs = exps / exps.sum(dim, keepdim=True)
    Acceptance: sums to ~1 along `dim` (1e-5 tolerance).
    """
    raise NotImplementedError


def manual_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy via log-softmax:
      log_probs = logits - logsumexp(logits, dim=-1, keepdim=True)
      loss = -mean( log_probs[range(B), targets] )
    Args:
      logits: (B, C), targets: (B,) int64
    Return: scalar tensor (loss)
    """
    raise NotImplementedError
