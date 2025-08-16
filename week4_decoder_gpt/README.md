# Week 4 — Transformer Decoder & GPT-Style LM

## Goal
Add Decoder (masked self-attn + cross-attn), build a simple GPT-like decoder-only model,
and train a tiny language model with a causal loss.

## Learn
- Positional Encodings: sinusoidal vs learned
- CLM training (next-token prediction)

## Flow
1) Implement `DecoderBlock`
2) Implement `sinusoidal_positional_encoding`
3) Implement `GPTDecoder` (embeddings + N decoder blocks + LM head)
4) Implement `clm_loss` (shifted targets) and wire up `train_gpt.py` (minimal)

Links:
- Vaswani §3.5 (positional encodings)
- GPT-2 (overview): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- HF course: https://huggingface.co/learn/nlp-course/chapter6
