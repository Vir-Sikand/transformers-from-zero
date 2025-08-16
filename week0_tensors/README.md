# Week 0 — PyTorch Tensors (Detailed Guide)

## Goal
Be fluent with PyTorch tensors so nothing about shapes, broadcasting, or devices slows you down later.

## Learn (official)
- Tensors: https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- Broadcasting: https://pytorch.org/docs/stable/notes/broadcasting.html
- Autograd basics: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

## Flow
1) Implement **TensorLab** utilities in `tensors_autograd.py`
2) Implement **`manual_softmax`** and **`manual_cross_entropy`**
3) Run sanity: `make week0_sanity`
4) Run tests: `pytest -k week0`

## Exercises
- Tensor creation, reshape/view/permute, broadcasting, masked ops, row-normalize, batched matmul, device move.
- Manual softmax (stable) and manual cross-entropy (scalar).

All functions have docstrings that specify the acceptance criteria.
