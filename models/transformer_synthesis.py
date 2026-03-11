# models/transformer_synthesis.py
import math
import torch
import torch.nn as nn


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


class StyleVAE(nn.Module):
    """
    Encodes a stroke prefix sequence into a latent style vector z.

    Architecture:
      - Bidirectional LSTM, 2 layers, hidden=256
      - Final layer hidden state: forward + backward concatenated → 512-dim
      - fc_mu:     Linear(512, 64) → μ
      - fc_logvar: Linear(512, 64) → log σ²

    forward(strokes, training=True) → (z, mu, logvar)
      - strokes: (batch, seq_len, 3)  [eos, dx, dy]
      - training=True:  z = μ + σ * ε,  ε ~ N(0, I)  (reparameterization)
      - training=False: z = μ  (deterministic)
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 256, num_layers: int = 2, latent_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Final layer forward + backward → 512-dim
        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, strokes: torch.Tensor, training: bool = True):
        """
        Args:
            strokes:  (batch, seq_len, 3)
            training: if True, sample via reparameterization; if False, return mu
        Returns:
            z:      (batch, latent_dim)
            mu:     (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # h_n: (num_layers * num_directions, batch, hidden_size) = (4, batch, 256)
        _, (h_n, _) = self.lstm(strokes)

        # h_n[-2]: final layer forward direction  (batch, 256)
        # h_n[-1]: final layer backward direction (batch, 256)
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, 512)

        mu = self.fc_mu(h_final)        # (batch, 64)
        logvar = self.fc_logvar(h_final)  # (batch, 64)

        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar
