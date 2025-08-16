import torch
from week1_nn_basics.layers import Linear, ReLU
from week1_nn_basics.mlp import MLP

def test_linear_mlp_shapes():
    lin = Linear(8,4); x = torch.randn(2,8)
    y = lin(x); assert y.shape == (2,4)
    mlp = MLP(8,16,3, num_hidden=2)
    y2 = mlp(torch.randn(2,8))
    assert y2.shape == (2,3)
