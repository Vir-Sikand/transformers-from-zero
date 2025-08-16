from week3_encoder.multihead_attention import MultiHeadAttention
from week3_encoder.feedforward import PositionwiseFFN
from week3_encoder.encoder_block import EncoderBlock

def test_encoder_components_exist():
    assert hasattr(MultiHeadAttention, '__doc__')
    assert hasattr(PositionwiseFFN, '__doc__')
    assert hasattr(EncoderBlock, '__doc__')
