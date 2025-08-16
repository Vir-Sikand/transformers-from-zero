# Week 2 — Attention & Masks

## Goal
Understand and implement scaled dot-product attention, padding masks, and causal masks.

## Learn
- Attention: https://arxiv.org/abs/1706.03762 (Sec 3.2)
- Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/

## Flow
1) Implement `scaled_dot_product_attention(Q,K,V,mask)`
2) Implement `causal_mask(T)` and `padding_mask(lengths)`
3) Write a tiny script (optional) to visualize attention maps for toy inputs.

Tests enforce shapes and masking behavior.
