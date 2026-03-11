# models/transformer_synthesis.py
import math
import numpy as np
import torch
import torch.nn as nn
from models.models import sample_from_out_dist


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    Encodes a padded character sequence into context vectors.
    Replaces the one-hot + Gaussian window mechanism.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text:      (batch, text_len) integer character indices
            text_mask: (batch, text_len) float, 1=valid 0=padding
        Returns:
            (batch, text_len, d_model)
        """
        x = self.embedding(text)
        x = self.pos_enc(x)
        # PyTorch key_padding_mask: True = ignore (padding)
        padding_mask = (text_mask == 0)
        return self.encoder(x, src_key_padding_mask=padding_mask)
