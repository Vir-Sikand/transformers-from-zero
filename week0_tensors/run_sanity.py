"""
Run after implementing Week 0 stubs to see sanity outputs.
"""
import torch
from .tensors_autograd import TensorLab, manual_softmax

def main():
    x = torch.randn(3, 5)
    print("create_range(5):", TensorLab.create_range(5))
    rn = TensorLab.row_normalize(torch.abs(x))
    print("row sums:", rn.sum(dim=1))
    p = manual_softmax(x, dim=-1)
    print("softmax row sums:", p.sum(dim=-1))

if __name__ == "__main__":
    main()
