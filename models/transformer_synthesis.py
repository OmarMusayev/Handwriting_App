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


class StyleVAE(nn.Module):
    """
    Encodes a stroke prefix sequence into a latent style vector z.

    Architecture:
      - Bidirectional LSTM, 2 layers, hidden=256
      - Final layer hidden state: forward + backward concatenated → 512-dim
      - fc_mu:     Linear(512, 64) → μ
      - fc_logvar: Linear(512, 64) → log σ²

    forward(strokes, use_sampling=True) → (z, mu, logvar)
      - strokes: (batch, seq_len, 3)  [eos, dx, dy]
      - use_sampling=True:  z = μ + σ * ε,  ε ~ N(0, I)  (reparameterization)
      - use_sampling=False: z = μ  (deterministic)
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

    def forward(self, strokes: torch.Tensor, use_sampling: bool = True):
        """
        Args:
            strokes:     (batch, seq_len, 3)
            use_sampling: if True, sample via reparameterization; if False, return mu
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

        if use_sampling:
            std = torch.exp(0.5 * logvar.clamp(-10, 10))
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar


class StrokeDecoder(nn.Module):
    """
    Autoregressive causal Transformer decoder for stroke sequences.

    At each step the stroke vector (3-dim) is concatenated with style vector z
    (latent_dim-dim) and projected to d_model. A causal self-attention mask
    prevents attending to future tokens. Cross-attention attends to the text
    embeddings produced by TextEncoder.

    forward(strokes, text_embeddings, text_padding_mask, z, past_kv=None)
        -> (batch, seq_len, d_model)

    past_kv is accepted but currently ignored (reserved for future KV-cache
    optimisation). Callers may always pass past_kv=None.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 64,
        stroke_dim: int = 3,
    ):
        super().__init__()
        self.d_model = d_model

        # Project (stroke_dim + latent_dim) -> d_model
        self.input_proj = nn.Linear(stroke_dim + latent_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN, consistent with TextEncoder
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def _gaussian_memory_mask(self, seq_len: int, text_len: int, device) -> torch.Tensor:
        """Additive attention bias that enforces monotonic left-to-right text alignment.

        mask[t, s] = -((s - mu_t)^2) / (2 * sigma^2)
        mu_t = t * (text_len - 1) / (seq_len - 1)   — linearly spaced centers
        sigma = text_len / 4.0                        — soft window ~half the text wide

        Returns (seq_len, text_len) float tensor added to cross-attention logits.
        TrInk (2025) shows this reduces CER from 70% to 8.5% vs no mask.
        """
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        s = torch.arange(text_len, dtype=torch.float32, device=device)
        mu = t * (text_len - 1) / max(seq_len - 1, 1)   # (seq_len,)
        sigma = max(text_len / 4.0, 1.0)
        bias = -((s.unsqueeze(0) - mu.unsqueeze(1)) ** 2) / (2 * sigma ** 2)
        return bias  # (seq_len, text_len)

    def forward(
        self,
        strokes: torch.Tensor,            # (batch, seq_len, 3)
        text_embeddings: torch.Tensor,    # (batch, text_len, d_model)
        text_padding_mask: torch.Tensor,  # (batch, text_len) float, 1=valid 0=padding
        z: torch.Tensor,                  # (batch, latent_dim)
        past_kv=None,                     # reserved for KV cache; ignored for now
    ) -> torch.Tensor:                    # (batch, seq_len, d_model)
        seq_len = strokes.size(1)
        text_len = text_embeddings.size(1)

        # Expand z to match sequence length, then concatenate with strokes
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([strokes, z_expanded], dim=-1)  # (batch, seq_len, stroke_dim + latent_dim)

        # Project to d_model and add positional encoding
        x = self.input_proj(x)        # (batch, seq_len, d_model)
        x = self.pos_enc(x)           # (batch, seq_len, d_model)

        # Causal self-attention mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=strokes.device
        )

        # Additive key padding mask: 0.0 for valid positions, -inf for padding
        # Using float (same type as memory_mask) avoids PyTorch type mismatch warning
        memory_key_padding_mask = torch.zeros_like(text_padding_mask)
        memory_key_padding_mask = memory_key_padding_mask.masked_fill(
            text_padding_mask == 0, float('-inf')
        )  # (batch, text_len)

        # Gaussian memory mask: forces monotonic left-to-right text-stroke alignment
        memory_mask = self._gaussian_memory_mask(seq_len, text_len, strokes.device)

        out = self.decoder(
            tgt=x,
            memory=text_embeddings,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
        )
        return out  # (batch, seq_len, d_model)


class MDNHead(nn.Module):
    """
    Projects decoder hidden states to MDN output parameters.

    Layout: 1 EOS + 6×20 mixture parameters
      (mixture weights, μ₁, μ₂, σ₁, σ₂, ρ) = 121 total outputs.

    forward(decoder_hidden) → (batch, seq_len, 121)
    """

    def __init__(self, d_model: int = 256, output_size: int = 121):
        super().__init__()
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_hidden: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, 121)
        """
        return self.linear(decoder_hidden)


class HandWritingSynthesisTransformer(nn.Module):
    """
    Top-level Transformer + Style VAE handwriting synthesis model.

    Wraps TextEncoder, StyleVAE, StrokeDecoder, and MDNHead.

    Training:
        forward(strokes, text, text_mask, style_strokes, use_sampling)
            → (y_hat, mu, logvar)

    Inference:
        generate(text, text_mask, style_strokes, bias, max_steps)
            → numpy array of shape (1, T, 3)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        text_layers: int = 4,
        dec_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 64,
        stroke_dim: int = 3,
        output_size: int = 121,
    ):
        super().__init__()
        self.EOS = False

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=text_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.style_vae = StyleVAE(
            input_size=stroke_dim,
            hidden_size=256,
            num_layers=2,
            latent_dim=latent_dim,
        )
        self.stroke_decoder = StrokeDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=dec_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            latent_dim=latent_dim,
            stroke_dim=stroke_dim,
        )
        self.mdn_head = MDNHead(d_model=d_model, output_size=output_size)

    def forward(
        self,
        strokes: torch.Tensor,         # (batch, seq_len, 3) — target strokes (teacher-forced)
        text: torch.Tensor,            # (batch, text_len) int char indices
        text_mask: torch.Tensor,       # (batch, text_len) float 1=valid 0=pad
        style_strokes: torch.Tensor,   # (batch, style_len, 3) — style prefix for VAE
        use_sampling: bool = True,     # passed to StyleVAE
    ) -> tuple:
        """
        Returns:
            y_hat:  (batch, seq_len, 121) — MDN outputs
            mu:     (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        text_embeddings = self.text_encoder(text, text_mask)            # (batch, text_len, d_model)
        z, mu, logvar = self.style_vae(style_strokes, use_sampling)     # z: (batch, latent_dim)
        decoder_out = self.stroke_decoder(strokes, text_embeddings, text_mask, z)  # (batch, seq_len, d_model)
        y_hat = self.mdn_head(decoder_out)                              # (batch, seq_len, 121)
        return y_hat, mu, logvar

    @torch.no_grad()
    def generate(
        self,
        text: torch.Tensor,            # (batch, text_len) int char indices
        text_mask: torch.Tensor,       # (batch, text_len) float 1=valid 0=pad
        style_strokes: torch.Tensor,   # (batch, style_len, 3)
        bias: float = 1.0,
        max_steps: int = 600,
    ) -> np.ndarray:                   # (1, seq_len, 3)
        """
        Autoregressive generation for batch=1.

        Starts decoder from SOS token [1,0,0] matching collate_fn training input.
        Style strokes are only used by the StyleVAE to encode z.

        EOS flag means pen-lift (not end-of-sequence), so generation runs for
        max_steps rather than stopping on the first pen-lift.

        Returns:
            numpy array of shape (1, total_steps, 3)
        """
        assert text.shape[0] == 1, "generate() only supports batch_size=1"
        self.EOS = False

        text_embeddings = self.text_encoder(text, text_mask)              # (1, text_len, d_model)
        z, _, _ = self.style_vae(style_strokes, use_sampling=False)       # deterministic at inference

        # SOS token: [1, 0, 0] — pen-up, zero offsets (matches collate_fn training input)
        inp = torch.zeros(1, 1, 3, device=text.device)
        inp[0, 0, 0] = 1.0  # pen-up flag
        max_steps = min(max_steps, 1190)

        gen_seq = []
        seq_len = 0

        while seq_len < max_steps:
            decoder_out = self.stroke_decoder(inp, text_embeddings, text_mask, z)  # (1, seq_so_far, d_model)
            y_hat = self.mdn_head(decoder_out)                                      # (1, seq_so_far, 121)

            # Take the last time step, squeeze batch dim → (121,)
            y_hat_last = y_hat[0, -1, :]

            Z = sample_from_out_dist(y_hat_last, bias)   # (1, 1, 3)
            gen_seq.append(Z)
            inp = torch.cat([inp, Z], dim=1)             # accumulate along seq dim
            seq_len += 1

        self.EOS = True
        # Stack list of (1, 1, 3) tensors → (1, total_steps, 3)
        result = torch.cat(gen_seq, dim=1)               # (1, total_steps, 3)
        return result.cpu().numpy()
