import torch
from week0_tensors.tensors_autograd import TensorLab, manual_softmax, manual_cross_entropy

def test_tensorlab_create_and_row_norm():
    t = TensorLab.create_range(5)
    assert t.shape == (5,)
    x = torch.randn(3,7).abs()
    rn = TensorLab.row_normalize(x)
    assert torch.allclose(rn.sum(dim=1), torch.ones(3), atol=1e-5)

def test_manual_softmax_and_ce():
    logits = torch.randn(2,8)
    probs = manual_softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    targets = torch.tensor([1,2])
    loss = manual_cross_entropy(logits, targets)
    assert loss.dim() == 0
