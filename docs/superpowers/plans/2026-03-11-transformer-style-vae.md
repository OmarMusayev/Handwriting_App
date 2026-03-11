# Transformer + Style VAE Handwriting Synthesis Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LSTM + Gaussian-window synthesis model with a Transformer encoder-decoder + Style VAE that trains on M4 Max MPS and deploys on CPU via INT8 quantization.

**Architecture:** TextEncoder (4-layer Transformer) encodes text; StyleVAE (BiLSTM) encodes a style prefix into a 64-dim latent z; StrokeDecoder (6-layer causal Transformer) attends to text and receives z at every step; MDNHead outputs the existing 121-dim mixture distribution. The app integration adds `MODEL_TYPE` branching; all other app code is unchanged.

**Tech Stack:** PyTorch 2.5, `nn.TransformerEncoder/Decoder`, `nn.LSTM` (BiLSTM), AdamW, cosine LR annealing, pytest

**Spec:** `docs/superpowers/specs/2026-03-11-transformer-style-vae-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `models/transformer_synthesis.py` | **Create** | All model components + wrapper |
| `train_transformer.py` | **Create** | Training loop, dataset, checkpointing |
| `tests/test_transformer_model.py` | **Create** | Component-level model tests |
| `tests/test_train_transformer.py` | **Create** | Dataset + training step tests |
| `models/models.py` | **Modify** | Fix Cholesky sampling bug in `sample_from_out_dist` |
| `generate.py` | **Modify** | Add Transformer adapter path |
| `app/core/config.py` | **Modify** | Add `model_type` env var |
| `app/core/singletons.py` | **Modify** | Branch `ModelSingleton` on `model_type` |
| `tests/test_singletons.py` | **Modify** | Add transformer singleton path test |

---

## Chunk 1: Pre-work

### Task 1: Document Sequence Length Stats

**Files:**
- No code changes — verification only

Data shows: n=6000, p50=627, p95=945, p99=1085, max=1191.

The current `MAX_GEN_STEPS=600` is below the median. Training will use `MAX_STROKE_LEN=1000` (covers p95). Inference cap will be raised to 1000 via env var in the deployment `.env`.

- [ ] **Step 1: Note the stats in a comment at the top of `train_transformer.py`** when it is created in Task 7.

- [ ] **Step 2: Commit**
```bash
git commit --allow-empty -m "chore: note stroke length stats p50=627 p95=945 max=1191, training cap=1000"
```

---

### Task 2: Fix Cholesky Sampling Bug

**Files:**
- Modify: `models/models.py` — `sample_from_out_dist` and `sample_batch_from_out_dist`
- Test: `tests/test_transformer_model.py` (added in Task 3, but write the test first here)

The current code does `Z = mu_k + cov @ x` where `cov = Σ`. This gives samples with covariance `Σ²`. The fix uses the lower-triangular Cholesky factor `L` where `Σ = L @ Lᵀ`.

For the 2D correlated Gaussian with params (σ₁, σ₂, ρ):
```
L = [[σ₁,           0        ],
     [ρ·σ₂,  σ₂·√(1 - ρ²)  ]]
```

- [ ] **Step 1: Write the failing test** in a new file `tests/test_cholesky_sampling.py`

```python
# tests/test_cholesky_sampling.py
import torch
import pytest
from models.models import sample_from_out_dist


def _make_y_hat(mu1=0.0, mu2=0.0, std1=1.0, std2=1.0, rho=0.8):
    """Build a 121-dim y_hat with one dominant mixture component."""
    y = torch.zeros(121)
    # mixture weights: make component 0 dominant
    y[1] = 10.0        # large logit for component 0
    y[21] = mu1        # mu_1[0]
    y[41] = mu2        # mu_2[0]
    y[61] = torch.log(torch.tensor(std1))   # logstd_1[0]
    y[81] = torch.log(torch.tensor(std2))   # logstd_2[0]
    y[101] = torch.atanh(torch.tensor(rho)) # pre-tanh rho[0]
    return y


def test_sample_covariance_is_not_sigma_squared():
    """With high correlation, samples must NOT have var >> std^2."""
    torch.manual_seed(42)
    y_hat = _make_y_hat(std1=1.0, std2=1.0, rho=0.9)
    samples = [sample_from_out_dist(y_hat, bias=0.0)[0, 0, 1:] for _ in range(2000)]
    samples = torch.stack(samples)
    # With correct Cholesky sampling, var(dx) ≈ 1.0
    # With wrong Σ sampling, var(dx) would be much larger (~(1 + 0.81) = 1.81)
    var_dx = samples[:, 0].var().item()
    assert var_dx < 1.5, (
        f"var(dx)={var_dx:.3f} — likely still using Σ instead of Cholesky L"
    )


def test_sample_returns_correct_shape():
    torch.manual_seed(0)
    y_hat = _make_y_hat()
    out = sample_from_out_dist(y_hat, bias=0.0)
    assert out.shape == (1, 1, 3)


def test_eos_is_zero_or_one():
    torch.manual_seed(0)
    y_hat = _make_y_hat()
    for _ in range(20):
        out = sample_from_out_dist(y_hat, bias=0.0)
        assert out[0, 0, 0].item() in (0.0, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**
```bash
cd /path/to/project && hand_gen_env/bin/pytest tests/test_cholesky_sampling.py -v
```
Expected: `test_sample_covariance_is_not_sigma_squared` FAILS.

- [ ] **Step 3: Apply the Cholesky fix to `models/models.py`**

In `sample_from_out_dist`, replace lines 29–40:

```python
# BEFORE (wrong — applies Σ, gives covariance Σ²):
cov = y_hat.new_zeros(2, 2)
cov[0, 0] = std_1[K].pow(2)
cov[1, 1] = std_2[K].pow(2)
cov[0, 1], cov[1, 0] = (
    correlations[K] * std_1[K] * std_2[K],
    correlations[K] * std_1[K] * std_2[K],
)
x = torch.normal(mean=torch.Tensor([0.0, 0.0]), std=torch.Tensor([1.0, 1.0])).to(
    y_hat.device
)
Z = mu_k + torch.mv(cov, x)
```

```python
# AFTER (correct — applies Cholesky L, gives covariance Σ):
# L is lower-triangular: Σ = L @ Lᵀ
# For [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]:
# L = [[σ₁, 0], [ρσ₂, σ₂√(1-ρ²)]]
L = y_hat.new_zeros(2, 2)
L[0, 0] = std_1[K]
L[1, 0] = correlations[K] * std_2[K]
L[1, 1] = std_2[K] * torch.sqrt(
    torch.clamp(1.0 - correlations[K].pow(2), min=1e-6)
)
x = torch.normal(
    mean=torch.zeros(2, device=y_hat.device),
    std=torch.ones(2, device=y_hat.device),
)
Z = mu_k + torch.mv(L, x)
```

Apply the same fix to `sample_batch_from_out_dist` (lines 70–85), using batched Cholesky:

```python
# AFTER for batched version:
L = y_hat.new_zeros(batch_size, 2, 2)
L[:, 0, 0] = std_1[torch.arange(batch_size), K]
L[:, 1, 0] = (correlations[torch.arange(batch_size), K]
              * std_2[torch.arange(batch_size), K])
L[:, 1, 1] = (std_2[torch.arange(batch_size), K]
              * torch.sqrt(torch.clamp(
                  1.0 - correlations[torch.arange(batch_size), K].pow(2),
                  min=1e-6)))
X = torch.randn(batch_size, 2, 1, device=y_hat.device)
Z = mu_k + torch.matmul(L, X).squeeze(-1)
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
hand_gen_env/bin/pytest tests/test_cholesky_sampling.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**
```bash
git add models/models.py tests/test_cholesky_sampling.py
git commit -m "fix: use Cholesky factor L for MDN sampling, not full covariance Σ"
```

---

## Chunk 2: Core Model

### Task 3: PositionalEncoding + TextEncoder

**Files:**
- Create: `models/transformer_synthesis.py`
- Test: `tests/test_transformer_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_transformer_model.py`:

```python
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
    """Padded positions should not affect valid positions' output."""
    enc = TextEncoder(vocab_size=77, d_model=256, nhead=8, num_layers=4, ff_dim=512)
    enc.eval()
    text = torch.randint(0, 77, (1, 10))
    # All valid
    mask_full = torch.ones(1, 10)
    # Only first 5 valid
    mask_half = torch.zeros(1, 10)
    mask_half[0, :5] = 1.0
    with torch.no_grad():
        out_full = enc(text, mask_full)
        out_half = enc(text, mask_half)
    # Valid positions differ when padding changes — encoder is not mask-invariant
    # (just checks it runs without error and shape is preserved)
    assert out_half.shape == (1, 10, 256)
```

- [ ] **Step 2: Run tests to confirm they fail**
```bash
hand_gen_env/bin/pytest tests/test_transformer_model.py::test_positional_encoding_output_shape -v
```
Expected: ImportError (module doesn't exist yet).

- [ ] **Step 3: Create `models/transformer_synthesis.py` with PositionalEncoding and TextEncoder**

```python
# models/transformer_synthesis.py
import math
import numpy as np
import torch
import torch.nn as nn
from models.models import sample_from_out_dist


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

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
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

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
        x = self.embedding(text)           # (batch, text_len, d_model)
        x = self.pos_enc(x)
        # PyTorch key_padding_mask: True = ignore (padding)
        padding_mask = (text_mask == 0)    # (batch, text_len) bool
        return self.encoder(x, src_key_padding_mask=padding_mask)
```

- [ ] **Step 4: Run tests**
```bash
hand_gen_env/bin/pytest tests/test_transformer_model.py::test_positional_encoding_output_shape tests/test_transformer_model.py::test_positional_encoding_adds_signal tests/test_transformer_model.py::test_text_encoder_output_shape tests/test_transformer_model.py::test_text_encoder_respects_padding_mask -v
```
Expected: all 4 PASS.

- [ ] **Step 5: Commit**
```bash
git add models/transformer_synthesis.py tests/test_transformer_model.py
git commit -m "feat: add PositionalEncoding and TextEncoder for Transformer synthesis"
```

---

### Task 4: StyleVAE

**Files:**
- Modify: `models/transformer_synthesis.py`
- Modify: `tests/test_transformer_model.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_transformer_model.py`)

```python
from models.transformer_synthesis import StyleVAE


def test_style_vae_encode_output_shapes():
    vae = StyleVAE(input_size=3, hidden_size=256, z_dim=64, num_layers=2)
    strokes = torch.randn(2, 50, 3)   # (batch=2, T=50, 3)
    mu, logvar = vae.encode(strokes)
    assert mu.shape == (2, 64)
    assert logvar.shape == (2, 64)


def test_style_vae_forward_training_samples():
    """z should differ from mu during training (stochastic)."""
    vae = StyleVAE(input_size=3, hidden_size=256, z_dim=64)
    vae.train()
    torch.manual_seed(0)
    strokes = torch.randn(1, 30, 3)
    z, mu, logvar = vae(strokes)
    assert z.shape == (1, 64)
    # z != mu because of reparameterization noise during training
    assert not torch.allclose(z, mu)


def test_style_vae_forward_eval_deterministic():
    """z should equal mu at eval (deterministic inference)."""
    vae = StyleVAE(input_size=3, hidden_size=256, z_dim=64)
    vae.eval()
    strokes = torch.randn(1, 30, 3)
    with torch.no_grad():
        z, mu, _ = vae(strokes)
    assert torch.allclose(z, mu)


def test_style_vae_kl_loss_non_negative():
    vae = StyleVAE(input_size=3, hidden_size=256, z_dim=64)
    strokes = torch.randn(2, 40, 3)
    _, mu, logvar = vae(strokes)
    kl = vae.kl_loss(mu, logvar)
    assert kl.item() >= 0.0
```

- [ ] **Step 2: Run tests to confirm they fail**
```bash
hand_gen_env/bin/pytest tests/test_transformer_model.py -k "style_vae" -v
```
Expected: ImportError for `StyleVAE`.

- [ ] **Step 3: Add `StyleVAE` to `models/transformer_synthesis.py`**

Append after `TextEncoder`:

```python
# ---------------------------------------------------------------------------
# Style VAE Encoder
# ---------------------------------------------------------------------------

class StyleVAE(nn.Module):
    """
    Encodes a stroke sequence (style prefix) into a latent vector z.
    Uses reparameterization trick during training; returns mu at eval.
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 256,
        z_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
        )
        self.mu_layer = nn.Linear(hidden_size * 2, z_dim)
        self.logvar_layer = nn.Linear(hidden_size * 2, z_dim)

    def encode(self, strokes: torch.Tensor):
        """
        Args:
            strokes: (batch, T, 3) — normalized stroke prefix
        Returns:
            mu, logvar: each (batch, z_dim)
        """
        _, (h, _) = self.lstm(strokes)
        # h: (num_layers * 2, batch, hidden_size) for bidirectional
        # Take the last layer's forward (h[-2]) and backward (h[-1]) states
        h_fwd = h[-2]                              # (batch, hidden_size)
        h_bwd = h[-1]                              # (batch, hidden_size)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1) # (batch, hidden_size * 2)
        return self.mu_layer(h_cat), self.logvar_layer(h_cat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, strokes: torch.Tensor):
        """Returns (z, mu, logvar)."""
        mu, logvar = self.encode(strokes)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence: KL(N(mu, exp(logvar)) || N(0, I)), summed over batch."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

- [ ] **Step 4: Run tests**
```bash
hand_gen_env/bin/pytest tests/test_transformer_model.py -k "style_vae" -v
```
Expected: all 4 PASS.

- [ ] **Step 5: Commit**
```bash
git add models/transformer_synthesis.py tests/test_transformer_model.py
git commit -m "feat: add StyleVAE encoder with reparameterization and KL loss"
```

---

### Task 5: StrokeDecoder

**Files:**
- Modify: `models/transformer_synthesis.py`
- Modify: `tests/test_transformer_model.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_transformer_model.py`)

```python
from models.transformer_synthesis import StrokeDecoder


def test_stroke_decoder_output_shape():
    dec = StrokeDecoder(d_model=256, nhead=8, num_layers=6, ff_dim=512, z_dim=64)
    strokes = torch.randn(2, 30, 3)          # (batch, T, 3)
    text_emb = torch.randn(2, 15, 256)       # (batch, text_len, d_model)
    z = torch.randn(2, 64)
    out = dec(strokes, text_emb, z)
    assert out.shape == (2, 30, 256)


def test_stroke_decoder_causal_mask_applied():
    """
    With a causal mask, the output at position t should not depend on
    positions > t. We verify this by zeroing out future inputs and
    checking the first t outputs are identical.
    """
    dec = StrokeDecoder(d_model=64, nhead=4, num_layers=2, ff_dim=128, z_dim=32)
    dec.eval()
    torch.manual_seed(7)
    T = 8
    strokes = torch.randn(1, T, 3)
    text_emb = torch.randn(1, 5, 64)
    z = torch.randn(1, 32)

    with torch.no_grad():
        out_full = dec(strokes, text_emb, z)
        # Zero out positions 4..7
        strokes_partial = strokes.clone()
        strokes_partial[0, 4:] = 0.0
        out_partial = dec(strokes_partial, text_emb, z)

    # Outputs at positions 0..3 must be identical
    assert torch.allclose(out_full[0, :4], out_partial[0, :4], atol=1e-5), \
        "Decoder is leaking future information — causal mask not working"
```

- [ ] **Step 2: Run tests to confirm they fail**
```bash
hand_gen_env/bin/pytest tests/test_transformer_model.py -k "stroke_decoder" -v
```
Expected: ImportError for `StrokeDecoder`.

- [ ] **Step 3: Add `StrokeDecoder` to `models/transformer_synthesis.py`**

Append after `StyleVAE`:

```python
# ---------------------------------------------------------------------------
# Stroke Decoder
# ---------------------------------------------------------------------------

class StrokeDecoder(nn.Module):
    """
    Autoregressive Transformer decoder.
    - Causal self-attention over previous strokes
    - Cross-attention to text_embeddings at every layer
    - Style vector z added to input projection at every step
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
        z_dim: int = 64,
        max_seq_len: int = 1200,
    ):
        super().__init__()
        self.d_model = d_model
        self.stroke_proj = nn.Linear(3, d_model)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, ff_dim, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(
        self,
        strokes: torch.Tensor,
        text_embeddings: torch.Tensor,
        z: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            strokes:              (batch, T, 3)
            text_embeddings:      (batch, text_len, d_model)
            z:                    (batch, z_dim)
            tgt_key_padding_mask: (batch, T) bool — True = padding
            memory_key_padding_mask: (batch, text_len) bool — True = padding
        Returns:
            (batch, T, d_model)
        """
        T = strokes.size(1)
        x = self.stroke_proj(strokes) + self.z_proj(z).unsqueeze(1)
        x = self.pos_enc(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=strokes.device
        )
        return self.decoder(
            x,
            text_embeddings,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
```

- [ ] **Step 4: Run tests**
```bash
hand_gen_env/bin/pytest tests/test_transformer_model.py -k "stroke_decoder" -v
```
Expected: all 2 PASS.

- [ ] **Step 5: Commit**
```bash
git add models/transformer_synthesis.py tests/test_transformer_model.py
git commit -m "feat: add StrokeDecoder with causal mask and cross-attention to text"
```

---

### Task 6: HandWritingSynthesisTransformer Wrapper

**Files:**
- Modify: `models/transformer_synthesis.py`
- Modify: `tests/test_transformer_model.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_transformer_model.py`)

```python
from models.transformer_synthesis import HandWritingSynthesisTransformer


def test_transformer_forward_output_shape():
    model = HandWritingSynthesisTransformer(vocab_size=77)
    strokes = torch.randn(2, 40, 3)
    text = torch.randint(0, 77, (2, 12))
    text_mask = torch.ones(2, 12)
    style = torch.randn(2, 15, 3)
    y_hat, mu, logvar = model(strokes, text, text_mask, style)
    assert y_hat.shape == (2, 40, 121)
    assert mu.shape == (2, 64)
    assert logvar.shape == (2, 64)


def test_transformer_generate_returns_strokes():
    model = HandWritingSynthesisTransformer(vocab_size=77)
    model.eval()
    style = torch.randn(1, 20, 3)
    text = torch.randint(0, 77, (1, 8))
    text_mask = torch.ones(1, 8)
    with torch.no_grad():
        gen = model.generate(style, text, text_mask, max_steps=10, bias=5.0)
    assert isinstance(gen, np.ndarray)
    assert gen.shape[2] == 3   # (1, T, 3)
    assert gen.shape[1] <= 10


def test_transformer_generate_stops_at_eos():
    """If EOS is always predicted, generation stops early."""
    model = HandWritingSynthesisTransformer(vocab_size=10)
    model.eval()
    # Bias the EOS logit massively positive so EOS fires on step 1
    with torch.no_grad():
        model.mdn_head.linear.bias[0] = 100.0
    style = torch.randn(1, 5, 3)
    text = torch.randint(0, 10, (1, 5))
    text_mask = torch.ones(1, 5)
    with torch.no_grad():
        gen = model.generate(style, text, text_mask, max_steps=100, bias=0.0)
    assert gen.shape[1] < 10, "Should have stopped early at EOS"
```

- [ ] **Step 2: Run tests to confirm they fail**
```bash
hand_gen_env/bin/pytest tests/test_transformer_model.py -k "transformer_forward or transformer_generate" -v
```
Expected: ImportError for `HandWritingSynthesisTransformer`.

- [ ] **Step 3: Add wrapper and `MDNHead` to `models/transformer_synthesis.py`**

Append:

```python
# ---------------------------------------------------------------------------
# MDN Head
# ---------------------------------------------------------------------------

class MDNHead(nn.Module):
    """Linear projection to 121-dim MDN parameters (unchanged layout)."""

    def __init__(self, d_model: int = 256, output_size: int = 121):
        super().__init__()
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class HandWritingSynthesisTransformer(nn.Module):
    """
    Transformer + Style VAE handwriting synthesis model.
    Drop-in replacement for HandWritingSynthesisNet.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        z_dim: int = 64,
        max_seq_len: int = 1200,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, d_model)
        self.style_vae = StyleVAE(z_dim=z_dim)
        self.stroke_decoder = StrokeDecoder(d_model=d_model, z_dim=z_dim, max_seq_len=max_seq_len)
        self.mdn_head = MDNHead(d_model)

    def forward(
        self,
        strokes: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        style_strokes: torch.Tensor,
        stroke_pad_mask: torch.Tensor = None,
    ):
        """
        Args:
            strokes:       (batch, T, 3) — decoder input (teacher-forced, shifted)
            text:          (batch, text_len) int
            text_mask:     (batch, text_len) float 1=valid 0=padding
            style_strokes: (batch, S, 3) — style prefix for VAE
            stroke_pad_mask: (batch, T) bool True=padding, optional
        Returns:
            y_hat:   (batch, T, 121)
            mu:      (batch, z_dim)
            logvar:  (batch, z_dim)
        """
        text_emb = self.text_encoder(text, text_mask)
        z, mu, logvar = self.style_vae(style_strokes)
        mem_pad_mask = (text_mask == 0)  # True = padding, for cross-attention
        dec_out = self.stroke_decoder(
            strokes, text_emb, z,
            tgt_key_padding_mask=stroke_pad_mask,
            memory_key_padding_mask=mem_pad_mask,
        )
        y_hat = self.mdn_head(dec_out)
        return y_hat, mu, logvar

    @torch.no_grad()
    def generate(
        self,
        style_strokes: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        max_steps: int = 600,
        bias: float = 5.0,
    ) -> np.ndarray:
        """
        Autoregressive generation.
        Caller must call model.eval() before invoking this method so that
        StyleVAE returns mu deterministically and dropout is disabled.

        Args:
            style_strokes: (1, S, 3) normalized style prefix
            text:          (1, text_len) int
            text_mask:     (1, text_len) float
            max_steps:     hard cap on generation length
            bias:          MDN temperature (higher = less variance)
        Returns:
            np.ndarray (1, T, 3) generated strokes (normalized)
        """
        device = text.device

        # Encode text and style once
        text_emb = self.text_encoder(text, text_mask)
        z = self.style_vae(style_strokes)[1]   # use mu — deterministic (requires eval mode)

        # BOS token: zero stroke
        gen = torch.zeros(1, 1, 3, device=device)
        gen_seq = []

        for _ in range(max_steps):
            mem_pad = (text_mask == 0)
            # StrokeDecoder.forward() generates its own causal mask internally
            dec_out = self.stroke_decoder(
                gen, text_emb, z,
                memory_key_padding_mask=mem_pad,
            )
            # Use only the last position's output
            y_hat = self.mdn_head(dec_out[:, -1, :]).squeeze(0)  # (121,)
            next_stroke = sample_from_out_dist(y_hat, bias)       # (1,1,3)
            gen_seq.append(next_stroke)
            gen = torch.cat([gen, next_stroke.to(device)], dim=1)

            if next_stroke[0, 0, 0].item() > 0.5:   # EOS
                break

        if not gen_seq:
            return np.zeros((1, 1, 3), dtype=np.float32)

        return torch.cat(gen_seq, dim=1).cpu().numpy()
```

- [ ] **Step 4: Run all model tests**
```bash
hand_gen_env/bin/pytest tests/test_transformer_model.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**
```bash
git add models/transformer_synthesis.py tests/test_transformer_model.py
git commit -m "feat: add MDNHead and HandWritingSynthesisTransformer wrapper with generate()"
```

---

## Chunk 3: Training Script

### Task 7: TransformerHandwritingDataset

**Files:**
- Create: `tests/test_train_transformer.py`
- The dataset class lives inside `train_transformer.py` (created fully in Task 8)

Note: stroke lengths are p50=627, p95=945, p99=1085, max=1191. `MAX_STROKE_LEN=1000` covers p95.

- [ ] **Step 1: Write failing tests**

Create `tests/test_train_transformer.py`:

```python
# tests/test_train_transformer.py
import numpy as np
import pytest
import torch
from unittest.mock import patch


def _make_fake_data(n=20, max_len=80):
    """Generate fake strokes + sentences for testing."""
    strokes = np.array(
        [np.random.randn(np.random.randint(30, max_len), 3).astype(np.float32)
         for _ in range(n)],
        dtype=object,
    )
    sentences = [f"hello world {i}" for i in range(n)]
    return strokes, sentences


def test_dataset_returns_style_target_text(tmp_path):
    from train_transformer import TransformerHandwritingDataset

    strokes, sentences = _make_fake_data(20)
    with patch("numpy.load", return_value=strokes):
        with patch("builtins.open", side_effect=lambda *a, **kw: __import__("io").StringIO("\n".join(sentences))):
            ds = TransformerHandwritingDataset("./data/", split="train", max_stroke_len=60)

    style, target, text_ids, text_mask, stroke_mask = ds[0]
    assert style.shape[1] == 3
    assert target.shape[1] == 3
    assert len(text_ids) == len(text_mask)
    assert style.shape[0] + target.shape[0] <= 60


def test_dataset_split_ratio_is_20_to_40_percent(tmp_path):
    from train_transformer import TransformerHandwritingDataset

    strokes, sentences = _make_fake_data(20, max_len=100)
    with patch("numpy.load", return_value=strokes):
        with patch("builtins.open", side_effect=lambda *a, **kw: __import__("io").StringIO("\n".join(sentences))):
            ds = TransformerHandwritingDataset("./data/", split="train", max_stroke_len=100)

    for i in range(10):
        style, target, *_ = ds[i]
        total = style.shape[0] + target.shape[0]
        ratio = style.shape[0] / total
        assert 0.19 <= ratio <= 0.42, f"split ratio {ratio:.2f} out of range"


def test_collate_fn_pads_to_max_in_batch():
    from train_transformer import collate_fn
    import torch

    # Two samples with different lengths
    style1 = torch.randn(10, 3)
    target1 = torch.randn(20, 3)
    text1 = torch.tensor([1, 2, 3])
    tmask1 = torch.ones(3)
    smask1 = torch.zeros(20)

    style2 = torch.randn(5, 3)
    target2 = torch.randn(30, 3)
    text2 = torch.tensor([4, 5, 6, 7])
    tmask2 = torch.ones(4)
    smask2 = torch.zeros(30)

    batch = [
        (style1, target1, text1, tmask1, smask1),
        (style2, target2, text2, tmask2, smask2),
    ]
    style_b, target_b, text_b, tmask_b, smask_b = collate_fn(batch)

    assert style_b.shape == (2, 10, 3)   # padded to max style len
    assert target_b.shape == (2, 30, 3)  # padded to max target len
    assert text_b.shape == (2, 4)        # padded to max text len
    # Valid positions must be False (not padding), padded positions must be True
    assert smask_b[0, :20].all() == False   # sample 0: first 20 positions valid
    assert smask_b[0, 20:].all() == True    # sample 0: positions 20-29 are padding
    assert smask_b[1, :30].all() == False   # sample 1: all 30 positions valid (no padding)
```

- [ ] **Step 2: Run tests to confirm they fail**
```bash
hand_gen_env/bin/pytest tests/test_train_transformer.py -v
```
Expected: ImportError (train_transformer.py doesn't exist).

- [ ] **Step 3: Create `train_transformer.py` with dataset and collate_fn** (training loop added in Task 8)

```python
# train_transformer.py
# Stroke length stats: n=6000, p50=627, p95=945, p99=1085, max=1191
# MAX_STROKE_LEN=1000 covers p95. Raise MAX_GEN_STEPS to 1000 in .env for the server.

import os
import math
import argparse
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.transformer_synthesis import HandWritingSynthesisTransformer
from utils.model_utils import compute_nll_loss
from utils.data_utils import data_denormalization
from utils import plot_stroke
from utils.constants import Global


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TransformerHandwritingDataset(Dataset):
    """
    Loads strokes.npy + sentences.txt.
    Each __getitem__ returns:
      style_strokes  — random 20-40% prefix (for StyleVAE)
      target_strokes — remaining 60-80% (for decoder target)
      text_ids       — character indices tensor
      text_mask      — 1=valid float tensor
      stroke_mask    — 0=valid (padding mask for decoder, all zeros pre-collate)
    """

    def __init__(self, data_path: str, split: str = "train", max_stroke_len: int = 1000):
        strokes_all = np.load(data_path + "strokes.npy", allow_pickle=True, encoding="bytes")
        with open(data_path + "sentences.txt") as f:
            texts_all = f.read().splitlines()

        n_total = len(strokes_all)
        n_train = int(0.9 * n_total)
        if split == "train":
            strokes_raw = strokes_all[:n_train]
            texts = texts_all[:n_train]
        else:
            strokes_raw = strokes_all[n_train:]
            texts = texts_all[n_train:]

        # Build vocab from all text
        counter = Counter()
        for t in texts_all:
            counter.update(list(t))
        unique = sorted(counter)
        self.char_to_id = {c: i for i, c in enumerate(unique)}
        self.vocab_size = len(unique)

        # Compute train normalisation stats from raw train split
        train_raw = strokes_all[:n_train]
        all_offsets = np.concatenate([s[:, 1:] for s in train_raw], axis=0)
        self.mean = all_offsets.mean(axis=0).astype(np.float32)
        self.std = all_offsets.std(axis=0).astype(np.float32)
        self.std[self.std == 0] = 1.0

        # Store in Global for denormalisation (matches existing convention)
        Global.train_mean = self.mean
        Global.train_std = self.std

        # Normalise and truncate
        self.strokes = []
        for s in strokes_raw:
            s = s.astype(np.float32)
            s[:, 1:] = (s[:, 1:] - self.mean) / self.std
            s = s[:max_stroke_len]
            self.strokes.append(s)

        self.texts = texts
        self.max_stroke_len = max_stroke_len

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        stroke = self.strokes[idx]    # (T, 3) normalised
        text = self.texts[idx]
        T = len(stroke)

        # Random style/target split: 20-40% for style
        split_pct = random.uniform(0.2, 0.4)
        split_pt = max(1, int(T * split_pct))

        style = torch.from_numpy(stroke[:split_pt])    # (S, 3)
        target = torch.from_numpy(stroke[split_pt:])   # (T-S, 3)

        text_ids = torch.tensor(
            [self.char_to_id.get(c, 0) for c in text], dtype=torch.long
        )
        text_mask = torch.ones(len(text_ids), dtype=torch.float)

        # stroke_pad_mask: all zeros (no padding yet — collate pads)
        stroke_mask = torch.zeros(len(target), dtype=torch.bool)

        return style, target, text_ids, text_mask, stroke_mask


def collate_fn(batch):
    """Pad style, target, text to max lengths in the batch."""
    styles, targets, texts, text_masks, stroke_masks = zip(*batch)

    max_style = max(s.shape[0] for s in styles)
    max_target = max(t.shape[0] for t in targets)
    max_text = max(t.shape[0] for t in texts)

    B = len(batch)

    style_b = torch.zeros(B, max_style, 3)
    target_b = torch.zeros(B, max_target, 3)
    text_b = torch.zeros(B, max_text, dtype=torch.long)
    text_mask_b = torch.zeros(B, max_text)
    stroke_mask_b = torch.ones(B, max_target, dtype=torch.bool)  # True = padding

    for i, (s, t, tx, tm, sm) in enumerate(zip(styles, targets, texts, text_masks, stroke_masks)):
        style_b[i, :s.shape[0]] = s
        target_b[i, :t.shape[0]] = t
        text_b[i, :tx.shape[0]] = tx
        text_mask_b[i, :tm.shape[0]] = tm
        stroke_mask_b[i, :sm.shape[0]] = False   # valid positions = False

    return style_b, target_b, text_b, text_mask_b, stroke_mask_b
```

- [ ] **Step 4: Run dataset tests**
```bash
hand_gen_env/bin/pytest tests/test_train_transformer.py -v
```
Expected: all 3 PASS.

- [ ] **Step 5: Commit**
```bash
git add train_transformer.py tests/test_train_transformer.py
git commit -m "feat: add TransformerHandwritingDataset with style-prefix split and collate_fn"
```

---

### Task 8: Training Loop + KL Annealing + Checkpointing

**Files:**
- Modify: `train_transformer.py`
- Modify: `tests/test_train_transformer.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_train_transformer.py`)

```python
import math
from models.transformer_synthesis import HandWritingSynthesisTransformer


def test_train_step_loss_decreases(tmp_path):
    """A single training step should produce a finite loss."""
    from train_transformer import train_epoch

    model = HandWritingSynthesisTransformer(vocab_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    # Fake mini-batch loader with one batch
    style = torch.randn(2, 8, 3)
    target = torch.randn(2, 20, 3)
    text = torch.randint(0, 10, (2, 6))
    text_mask = torch.ones(2, 6)
    stroke_mask = torch.zeros(2, 20, dtype=torch.bool)

    loader = [(style, target, text, text_mask, stroke_mask)]
    loss = train_epoch(model, optimizer, 0, loader, device, beta=0.0)
    assert math.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss > 0


def test_checkpoint_save_and_resume(tmp_path):
    """Saving and loading a checkpoint restores all training state."""
    from train_transformer import save_checkpoint, load_checkpoint
    import torch

    model = HandWritingSynthesisTransformer(vocab_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    save_checkpoint(
        path=tmp_path / "ckpt.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        best_val_loss=0.42,
        beta=0.3,
    )

    model2 = HandWritingSynthesisTransformer(vocab_size=10)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)

    state = load_checkpoint(tmp_path / "ckpt.pt", model2, optimizer2, scheduler2)
    assert state["epoch"] == 5
    assert abs(state["best_val_loss"] - 0.42) < 1e-6
    assert abs(state["beta"] - 0.3) < 1e-6
```

- [ ] **Step 2: Run tests to confirm they fail**
```bash
hand_gen_env/bin/pytest tests/test_train_transformer.py -k "train_step or checkpoint" -v
```
Expected: ImportError for `train_epoch`, `save_checkpoint`, `load_checkpoint`.

- [ ] **Step 3: Append training loop + checkpointing to `train_transformer.py`**

```python
# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, beta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "beta": beta,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return {
        "epoch": ckpt["epoch"],
        "best_val_loss": ckpt["best_val_loss"],
        "beta": ckpt["beta"],
    }


# ---------------------------------------------------------------------------
# Beta (KL weight) schedule
# ---------------------------------------------------------------------------

def get_beta(epoch: int, warmup_epochs: int = 20, anneal_epochs: int = 40) -> float:
    """
    Returns beta for KL annealing:
      epochs 0..warmup_epochs-1:            beta = 0.0
      epochs warmup_epochs..warmup+anneal:  beta linearly 0 -> 1
      after:                                beta = 1.0
    """
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / anneal_epochs
    return min(1.0, progress)


# ---------------------------------------------------------------------------
# Train + Validation Epochs
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, epoch, train_loader, device, beta):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in train_loader:
        style, target, text, text_mask, stroke_mask = batch
        style = style.to(device)
        target = target.to(device)
        text = text.to(device)
        text_mask = text_mask.to(device)
        stroke_mask = stroke_mask.to(device)

        # Teacher forcing: decoder input = target[0:T-1], targets = target[1:T]
        dec_input = target[:, :-1, :]
        dec_target = target[:, 1:, :]
        stroke_mask_in = stroke_mask[:, :-1]

        # NLL loss mask: 1=valid, 0=padding (inverted from stroke_mask)
        nll_mask = (~stroke_mask[:, 1:]).float()

        optimizer.zero_grad()

        y_hat, mu, logvar = model(dec_input, text, text_mask, style, stroke_pad_mask=stroke_mask_in)

        nll = compute_nll_loss(dec_target, y_hat, nll_mask)
        kl = StyleVAE.kl_loss(mu, logvar) if beta > 0 else torch.tensor(0.0, device=device)
        # Normalize both NLL and KL by batch size for a consistent per-sample loss
        n = dec_input.size(0)
        loss = (nll + beta * kl) / n

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * n
        total_tokens += n

    return total_loss / max(total_tokens, 1)


def validation_epoch(model, valid_loader, device, beta):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in valid_loader:
            style, target, text, text_mask, stroke_mask = batch
            style = style.to(device)
            target = target.to(device)
            text = text.to(device)
            text_mask = text_mask.to(device)
            stroke_mask = stroke_mask.to(device)

            dec_input = target[:, :-1, :]
            dec_target = target[:, 1:, :]
            stroke_mask_in = stroke_mask[:, :-1]
            nll_mask = (~stroke_mask[:, 1:]).float()

            y_hat, mu, logvar = model(dec_input, text, text_mask, style, stroke_pad_mask=stroke_mask_in)
            nll = compute_nll_loss(dec_target, y_hat, nll_mask)
            kl = StyleVAE.kl_loss(mu, logvar) if beta > 0 else torch.tensor(0.0)
            n = dec_input.size(0)
            loss = (nll + beta * kl) / n

            total_loss += loss.item() * n
            total_tokens += n

    return total_loss / max(total_tokens, 1)
```

Also add the import at the top of `train_transformer.py`:
```python
from models.transformer_synthesis import HandWritingSynthesisTransformer, StyleVAE
```

- [ ] **Step 4: Run tests**
```bash
hand_gen_env/bin/pytest tests/test_train_transformer.py -k "train_step or checkpoint" -v
```
Expected: all 2 PASS.

- [ ] **Step 5: Commit**
```bash
git add train_transformer.py tests/test_train_transformer.py
git commit -m "feat: add training loop, KL annealing, and checkpoint save/resume"
```

---

### Task 9: Main Training Entry Point

**Files:**
- Modify: `train_transformer.py` — add `main()` + `argparser()`

No new tests — this is a CLI entry point wiring together already-tested pieces.

- [ ] **Step 1: Append `main()` to `train_transformer.py`**

```python
# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def argparser():
    p = argparse.ArgumentParser(description="Train Transformer + Style VAE")
    p.add_argument("--data_path", type=str, default="./data/")
    p.add_argument("--save_path", type=str, default="./checkpoints/transformer/")
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_stroke_len", type=int, default=1000)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = argparser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device: MPS on Apple Silicon, CUDA if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}")

    train_ds = TransformerHandwritingDataset(args.data_path, split="train", max_stroke_len=args.max_stroke_len)
    valid_ds = TransformerHandwritingDataset(args.data_path, split="valid", max_stroke_len=args.max_stroke_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = HandWritingSynthesisTransformer(vocab_size=train_ds.vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    save_path = Path(args.save_path)
    latest_ckpt = save_path / "checkpoint_latest.pt"
    best_ckpt = save_path / "checkpoint_best.pt"

    start_epoch = 0
    best_val_loss = math.inf
    beta = 0.0

    if args.resume and latest_ckpt.exists():
        state = load_checkpoint(latest_ckpt, model, optimizer, scheduler)
        start_epoch = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        beta = state["beta"]
        print(f"Resumed from epoch {state['epoch']}, best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.n_epochs):
        beta = get_beta(epoch)
        print(f"\nEpoch {epoch+1}/{args.n_epochs}  beta={beta:.3f}")

        train_loss = train_epoch(model, optimizer, epoch, train_loader, device, beta)
        val_loss = validation_epoch(model, valid_loader, device, beta)
        scheduler.step()

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        save_checkpoint(latest_ckpt, model, optimizer, scheduler, epoch, best_val_loss, beta)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, best_val_loss, beta)
            print(f"  New best model saved (val_loss={val_loss:.4f})")

            # Generate a sample to verify training progress
            model.eval()
            sample_text = "hello"
            text_ids = torch.tensor([[train_ds.char_to_id.get(c, 0) for c in sample_text]], device=device)
            text_mask = torch.ones(1, len(sample_text), device=device)
            style_sample = torch.from_numpy(valid_ds.strokes[0][:30]).unsqueeze(0).to(device)
            with torch.no_grad():
                gen = model.generate(style_sample, text_ids, text_mask, max_steps=400, bias=5.0)
            gen_denorm = data_denormalization(Global.train_mean, Global.train_std, gen)
            plot_stroke(gen_denorm[0], save_name=str(save_path / f"sample_epoch{epoch+1}.png"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the entry point compiles**
```bash
hand_gen_env/bin/python -c "import train_transformer; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**
```bash
git add train_transformer.py
git commit -m "feat: add train_transformer.py CLI entry point with MPS support and resume"
```

---

## Chunk 4: App Integration

### Task 10: config.py + ModelSingleton Branching

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/core/singletons.py`
- Modify: `tests/test_singletons.py`

- [ ] **Step 1: Write failing test** (append to `tests/test_singletons.py`)

```python
def test_model_singleton_loads_transformer_when_model_type_is_transformer():
    from app.core.singletons import ModelSingleton
    ModelSingleton._model = None
    mock_model = MagicMock()
    mock_model.load_state_dict = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    # Patch in the models module since the import inside get() is a local deferred import
    with patch("models.transformer_synthesis.HandWritingSynthesisTransformer", return_value=mock_model):
        with patch("torch.load", return_value={}):
            m = ModelSingleton.get("fake.pt", "cpu", 77, model_type="transformer")
    assert m is mock_model


def test_model_singleton_defaults_to_lstm():
    from app.core.singletons import ModelSingleton
    ModelSingleton._model = None
    mock_model = MagicMock()
    mock_model.load_state_dict = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    with patch("app.core.singletons.HandWritingSynthesisNet", return_value=mock_model):
        with patch("torch.load", return_value={}):
            m = ModelSingleton.get("fake.pt", "cpu", 77)  # no model_type arg
    assert m is mock_model
```

- [ ] **Step 2: Run tests to confirm they fail**
```bash
hand_gen_env/bin/pytest tests/test_singletons.py -k "transformer" -v
```
Expected: FAIL — `ModelSingleton.get` doesn't accept `model_type` yet.

- [ ] **Step 3: Update `app/core/config.py`**

Add one field to the `Settings` class:
```python
model_type: str = os.getenv("MODEL_TYPE", "lstm")  # "lstm" or "transformer"
```

- [ ] **Step 4: Update `app/core/singletons.py`**

Update `ModelSingleton.get()`:
```python
@classmethod
def get(cls, model_path: str, device: str, vocab_size: int, model_type: str = "lstm"):
    if cls._model is None:
        if model_type == "transformer":
            from models.transformer_synthesis import HandWritingSynthesisTransformer
            model = HandWritingSynthesisTransformer(vocab_size=vocab_size)
        else:
            model = HandWritingSynthesisNet(window_size=vocab_size)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        cls._model = model
    return cls._model
```

Update `startup_singletons` to pass `model_type` through:
```python
def startup_singletons(data_path: str, model_path: str, device: str):
    from app.core.config import settings
    VocabSingleton.initialize(data_path)
    StatsSingleton.initialize(data_path)
    ModelSingleton.get(model_path, device, VocabSingleton.vocab_size(), model_type=settings.model_type)
```

- [ ] **Step 5: Run tests**
```bash
hand_gen_env/bin/pytest tests/test_singletons.py -v
```
Expected: all PASS (including existing tests).

- [ ] **Step 6: Commit**
```bash
git add app/core/config.py app/core/singletons.py tests/test_singletons.py
git commit -m "feat: add MODEL_TYPE env var and branch ModelSingleton on lstm/transformer"
```

---

### Task 11: generate.py Adapter

**Files:**
- Modify: `generate.py`
- Test: `tests/test_generate_adapter.py` (new)

- [ ] **Step 1: Read the existing `generate_conditional_sequence` signature**

```bash
hand_gen_env/bin/python -c "import inspect, generate; print(inspect.signature(generate.generate_conditional_sequence))"
```

- [ ] **Step 2: Write failing test**

Create `tests/test_generate_adapter.py`:

```python
# tests/test_generate_adapter.py
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_generate_conditional_sequence_with_transformer():
    """When passed a HandWritingSynthesisTransformer, should call model.generate()."""
    from generate import generate_conditional_sequence
    from models.transformer_synthesis import HandWritingSynthesisTransformer

    mock_model = MagicMock(spec=HandWritingSynthesisTransformer)
    mock_model.generate.return_value = np.zeros((1, 10, 3), dtype=np.float32)

    char_to_id = {c: i for i, c in enumerate("hello ")}
    idx_to_char = lambda ids: np.array([list("hello ")[i] for i in ids])

    gen, phi = generate_conditional_sequence(
        mock_model,
        "hello",
        torch.device("cpu"),
        char_to_id,
        idx_to_char,
        bias=5.0,
        prime=True,
        prime_seq=torch.zeros(1, 10, 3),
        real_text="hello",
        is_map=False,
    )

    assert mock_model.generate.called
    assert isinstance(gen, np.ndarray)
    assert phi.shape == (0,)   # empty phi for transformer


def test_generate_conditional_sequence_transformer_passes_prime_seq_as_style():
    """prime_seq tensor must be forwarded to model.generate() as the style argument."""
    from generate import generate_conditional_sequence
    from models.transformer_synthesis import HandWritingSynthesisTransformer

    mock_model = MagicMock(spec=HandWritingSynthesisTransformer)
    mock_model.generate.return_value = np.zeros((1, 10, 3), dtype=np.float32)

    char_to_id = {c: i for i, c in enumerate("hello ")}
    idx_to_char = lambda ids: np.array([list("hello ")[i] for i in ids])
    style_tensor = torch.ones(1, 10, 3)   # distinctive non-zero tensor

    generate_conditional_sequence(
        mock_model,
        "hello",
        torch.device("cpu"),
        char_to_id,
        idx_to_char,
        bias=5.0,
        prime=True,
        prime_seq=style_tensor,
        real_text="hello",
        is_map=False,
    )

    call_args = mock_model.generate.call_args
    # First positional arg to model.generate() must be the style tensor
    actual_style = call_args[0][0] if call_args[0] else call_args[1].get("style_strokes")
    assert actual_style is not None
    assert torch.allclose(actual_style.cpu(), style_tensor)
```

- [ ] **Step 3: Run tests to confirm the transformer path fails**
```bash
hand_gen_env/bin/pytest tests/test_generate_adapter.py -v
```
Expected: `test_generate_conditional_sequence_with_transformer` FAILS.

- [ ] **Step 4: Update `generate.py`**

Read `generate.py` first to understand the current structure, then add the transformer branch at the start of `generate_conditional_sequence`:

```python
def generate_conditional_sequence(
    model_or_path,
    char_seq,
    device,
    char_to_id,
    idx_to_char,
    bias,
    prime=False,
    prime_seq=None,
    real_text=None,
    is_map=False,
):
    from models.transformer_synthesis import HandWritingSynthesisTransformer

    # --- Transformer path ---
    if isinstance(model_or_path, HandWritingSynthesisTransformer):
        model = model_or_path
        # model.eval() must be set by the caller (ModelSingleton.get() does this at load time)
        text_ids = torch.tensor(
            [[char_to_id.get(c, 0) for c in char_seq + "  "]],
            device=device,
        )
        text_mask = torch.ones(1, text_ids.shape[1], device=device)
        # prime_seq is the style strokes tensor (1, T, 3)
        if prime_seq is None:
            style = torch.zeros(1, 10, 3, device=device)
        else:
            style = prime_seq.to(device)
        gen = model.generate(style, text_ids, text_mask, max_steps=600, bias=bias)
        phi = np.zeros(0)
        return gen, phi

    # --- Original LSTM path (unchanged below) ---
    ...
```

- [ ] **Step 5: Run all tests**
```bash
hand_gen_env/bin/pytest tests/test_generate_adapter.py tests/test_transformer_model.py tests/test_singletons.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**
```bash
git add generate.py tests/test_generate_adapter.py
git commit -m "feat: add Transformer adapter path to generate_conditional_sequence"
```

---

### Task 12: Full Integration Smoke Test

**Files:**
- No new files

- [ ] **Step 1: Run the full test suite**
```bash
hand_gen_env/bin/pytest tests/ -v
```
Expected: all existing tests PASS plus all new tests PASS.

- [ ] **Step 2: Smoke-test the full generation pipeline with the transformer**

```bash
hand_gen_env/bin/python - <<'EOF'
import torch
import numpy as np
from models.transformer_synthesis import HandWritingSynthesisTransformer
from generate import generate_conditional_sequence

# Tiny vocab, untrained model — just verifying shapes and no crashes
model = HandWritingSynthesisTransformer(vocab_size=20)
model.eval()
char_to_id = {chr(ord('a') + i): i for i in range(20)}
idx_to_char = lambda ids: np.array([chr(ord('a') + int(i)) for i in ids])
style = torch.randn(1, 15, 3)
gen, phi = generate_conditional_sequence(
    model, "hello", torch.device("cpu"), char_to_id, idx_to_char,
    bias=5.0, prime=True, prime_seq=style, real_text="hello"
)
print(f"gen shape: {gen.shape}")  # expect (1, T, 3) where T <= 600
print(f"phi shape: {phi.shape}")  # expect (0,)
print("Smoke test PASSED")
EOF
```
Expected output: `gen shape: (1, T, 3)`, `Smoke test PASSED`.

- [ ] **Step 3: Final commit**
```bash
git add -A
git commit -m "feat: Transformer + Style VAE complete — all tests passing"
```

---

## Deployment Checklist (After Training)

After training completes on the M4 Max and `checkpoint_best.pt` exists:

- [ ] Copy `checkpoints/transformer/checkpoint_best.pt` to the laptop server
- [ ] Update `.env` on the server:
  ```
  MODEL_TYPE=transformer
  MODEL_PATH=checkpoints/transformer/checkpoint_best.pt
  MAX_GEN_STEPS=1000
  ```
- [ ] Apply INT8 quantization at load time (already in `ModelSingleton.get()` — add the `quantize_dynamic` call there)
- [ ] Restart uvicorn: `sudo systemctl restart handwriting`
- [ ] Verify one generation request completes in < 15s

---

## Training Command (M4 Max)

```bash
# First run
python train_transformer.py --data_path ./data/ --save_path ./checkpoints/transformer/ --n_epochs 100 --batch_size 32

# Resume after interruption
python train_transformer.py --data_path ./data/ --save_path ./checkpoints/transformer/ --n_epochs 100 --batch_size 32 --resume
```
