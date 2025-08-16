# Week 3 — Transformer Encoder

## Goal
Build Encoder blocks: Multi-Head Attention + Positionwise FFN with Pre-LN and residuals.

## Learn
- Vaswani §3.2 (MHA, FFN, LayerNorm & residuals)

## Flow
1) Implement `MultiHeadAttention` (project/split/attend/concat/out)
2) Implement `PositionwiseFFN` (d_model→d_ff→d_model with GELU)
3) Implement `EncoderBlock` (Pre-LN residual pattern)

Tests check that these components exist and wire up correctly.
