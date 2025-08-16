import torch
from week2_attention.attention import scaled_dot_product_attention
from week2_attention.masking import causal_mask

def test_attention_and_mask_shapes():
    B,H,T,D = 2,3,5,4
    Q = torch.randn(B,H,T,D); K = torch.randn(B,H,T,D); V = torch.randn(B,H,T,D)
    M = causal_mask(T).unsqueeze(0).unsqueeze(0)
    out, attn = scaled_dot_product_attention(Q,K,V,mask=M)
    assert out.shape == (B,H,T,D) and attn.shape == (B,H,T,T)
    assert torch.allclose(attn[0,0].triu(1), torch.zeros_like(attn[0,0].triu(1)), atol=1e-5)
