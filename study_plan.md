# Study Plan (Weeks 0–4)

Daily cadence (~60–120 min): read 20–30m → implement 40–90m → run tests → notes.

## Week 0 — Tensors
- Learn: PyTorch Tensors, broadcasting, reductions, device moves, autograd intro
- Implement: TensorLab utilities + `manual_softmax`, `manual_cross_entropy`
- Sanity: run `make week0_sanity`; tests `-k week0`

## Week 1 — Simple Neural Network
- Learn: nn.Module, Parameters, forward/backward, training loop
- Implement: `Linear`, `ReLU`, `MLP`, `train_one_epoch`, `eval_loss`
- Sanity: run `make week1_overfit`; tests `-k week1`

## Week 2 — Attention & Masks
- Learn: Scaled dot-product attention; causal/padding masks
- Implement: `scaled_dot_product_attention`, `causal_mask`, `padding_mask`

## Week 3 — Transformer Encoder
- Learn: Multi-head attention, Positionwise FFN, LayerNorm, Residuals (Pre-LN)
- Implement: `MultiHeadAttention`, `PositionwiseFFN`, `EncoderBlock`

## Week 4 — Transformer Decoder & GPT-Style LM
- Learn: Sinusoidal vs learned positional encodings; CLM training
- Implement: `DecoderBlock`, `GPTDecoder`, `sinusoidal_positional_encoding`, `clm_loss`, `train_gpt` (tiny)
