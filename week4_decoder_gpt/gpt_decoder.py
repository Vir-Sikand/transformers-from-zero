import torch, torch.nn as nn
from .decoder_block import DecoderBlock
from week2_attention.masking import causal_mask

class GPTDecoder(nn.Module):
    """
    Decoder-only model:
      token_emb + pos_emb -> N * DecoderBlock (self-attn only) -> LM head
    Forward:
      input_ids (B,T) -> logits (B,T,V)
    """
    def __init__(self):
        super().__init__()
        # TODO: define embeddings, blocks, final linear head
        raise NotImplementedError

    def forward(self, input_ids):
        raise NotImplementedError
