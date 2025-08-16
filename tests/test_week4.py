from week4_decoder_gpt.decoder_block import DecoderBlock
from week4_decoder_gpt.positional_encoding import sinusoidal_positional_encoding
from week4_decoder_gpt.gpt_decoder import GPTDecoder
from week4_decoder_gpt.clm_loss import clm_loss

def test_decoder_gpt_components_exist():
    assert hasattr(DecoderBlock, '__doc__')
    assert callable(sinusoidal_positional_encoding)
    assert hasattr(GPTDecoder, '__doc__')
    assert callable(clm_loss)
