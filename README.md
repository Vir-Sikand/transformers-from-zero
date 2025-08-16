# Transformers From Zero — Implement-It-Yourself (No RL)

A **straightforward, guided** repo to learn:
1) **Week 0:** PyTorch tensor fundamentals
2) **Week 1:** Build a simple neural network (forward/backward, layers, training loop)
3) **Week 2:** Attention & masking
4) **Week 3:** Transformer Encoder (MHA + FFN + Pre-LN)
5) **Week 4:** Transformer Decoder & GPT-style causal LM (positional encodings, CLM loss, tiny training loop)

Everything is **stubs + TODOs** with **detailed docstrings, acceptance criteria, runnable scripts, tests, and links**.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                 # see what fails first
make week0_sanity      # after implementing Week 0 basics
make week1_overfit     # after Week 1 training loop
```
