# tests/test_transformer_model.py
import torch
import pytest
import numpy as np
from models.transformer_synthesis import TextEncoder, PositionalEncoding


def test_positional_encoding_output_shape():
    pe = PositionalEncoding(d_model=256)
    x = torch.randn(2, 10, 256)  # (batch, seq, d_model)
    out = pe(x)
    assert out.shape == (2, 10, 256)


def test_positional_encoding_adds_signal():
    pe = PositionalEncoding(d_model=256, dropout=0.0)
    x = torch.zeros(1, 5, 256)
    out = pe(x)
    # output should not be all zeros — positional signal was added
    assert out.abs().sum() > 0


def test_text_encoder_output_shape():
    enc = TextEncoder(vocab_size=77, d_model=256, nhead=8, num_layers=4, ff_dim=512)
    text = torch.randint(0, 77, (2, 15))       # (batch=2, text_len=15)
    text_mask = torch.ones(2, 15, dtype=torch.float)
    out = enc(text, text_mask)
    assert out.shape == (2, 15, 256)


def test_text_encoder_respects_padding_mask():
    """Padded positions should not crash and shape is preserved."""
    enc = TextEncoder(vocab_size=77, d_model=256, nhead=8, num_layers=4, ff_dim=512)
    enc.eval()
    text = torch.randint(0, 77, (1, 10))
    mask_half = torch.zeros(1, 10)
    mask_half[0, :5] = 1.0
    with torch.no_grad():
        out_half = enc(text, mask_half)
    assert out_half.shape == (1, 10, 256)
