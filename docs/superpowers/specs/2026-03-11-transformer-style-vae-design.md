# Transformer + Style VAE Handwriting Synthesis

**Date:** 2026-03-11
**Status:** Approved
**Train target:** M4 Max MacBook (48GB unified RAM, MPS GPU)
**Inference target:** Dell Latitude 5420 (i7-1185G7, CPU-only, 16GB RAM)

---

## Problem

The current model (Graves 2013 LSTM + Gaussian Window Attention + MDN) has three concrete quality issues:

1. **Shaky strokes** — LSTM struggles with long-range stroke dependencies
2. **Text fidelity** — hand-designed Gaussian window attention misses characters on longer strings
3. **Style drift** — style priming via hidden state copy fades after the first few strokes

---

## Solution

Replace the LSTM backbone with a Transformer encoder-decoder, and replace hidden-state style priming with a VAE-based style encoder. Keep the MDN output head with the Cholesky sampling bug fixed (sampling-only fix — output layout unchanged). The existing data pipeline, loss function, and app are preserved with minimal targeted changes.

---

## Architecture

### Components

```
[Text]           → Text Encoder      → text_embeddings  (seq_len × 256)
[Canvas strokes] → Style VAE Encoder → z                (64-dim vector)
[z + text_emb]   → Stroke Decoder    → MDN outputs      (121-dim per step)
[MDN outputs]    → Sample            → (eos, dx, dy)
```

### Text Encoder
- Transformer encoder, 4 layers, d_model=256, nhead=8, ff_dim=512
- Learned character embeddings (replaces one-hot + Gaussian window)
- Output: sequence of context vectors (text_len × 256) for decoder cross-attention

### Style VAE Encoder
- Bidirectional LSTM, 2 layers, hidden=256
- Final hidden state → two linear projections → μ (64-dim), log σ (64-dim)
- Training: sample z ~ N(μ, σ) via reparameterization trick
- Inference: use μ directly (deterministic, no sampling noise)
- Runs per-request at inference time (user draws new strokes each request)
- Included in INT8 quantization at deployment

### Style Input During Training
The IAM dataset has no writer-identity metadata. The workaround: for each stroke sample, split the sequence at a random point between 20–40% of its length. The first portion is fed to the Style VAE as style context; the remaining portion is the generation target. This forces the VAE to encode general stroke character (slant, pressure, rhythm) from a short prefix, and the decoder generates the rest. The split point is randomized per epoch to prevent overfitting to a fixed split.

### Stroke Decoder
- Autoregressive Transformer decoder, 6 layers, d_model=256, nhead=8, ff_dim=512
- Causal self-attention on previous strokes (upper-triangular causal mask combined with key-padding mask over zero-padded positions, following standard `nn.TransformerDecoder` conventions)
- Cross-attention to text_embeddings at every layer
- Style vector z added to stroke input projection at every step
- KV cache at inference: previous stroke K/V tensors are cached; each new step is O(T) in self-attention over cached history but avoids redundant full re-encoding of all previous steps

### MDN Head
- Linear(256 → 121)
- Output split: 1 EOS + 6×20 mixture params (weights, μ₁, μ₂, σ₁, σ₂, ρ)
- Output layout and `compute_nll_loss` are **unchanged**
- **Cholesky sampling fix**: `sample_from_out_dist` is updated to use the Cholesky factor L of the covariance matrix instead of multiplying by Σ directly. This is a sampling-only fix; the (σ₁, σ₂, ρ) parameterization and 121-dim layout remain the same.

**Total parameters:** ~12M (FP32), ~3MB (INT8 quantized)

---

## Data Flow

### Training
```
For each batch (batch_size=32):
  stroke sequence     → split at random 20-40% point
  style_prefix        → Style VAE → z ~ N(μ, σ)
  target_strokes[0:T-1] → Stroke Decoder (teacher forcing, cross-attn to text, + z)
  target_strokes[1:T]   → MDN loss targets

Loss = NLL(MDN) + β · KL(N(μ,σ) || N(0,I))
```

### Inference
```
1. User draws on canvas → style_strokes (the drawn portion)
2. Style VAE encoder runs → z = μ (deterministic)
3. User types text → Text Encoder → text_embeddings
4. Decoder autoregressively generates (eos, dx, dy) with KV cache
5. Stop at eos=1 or MAX_GEN_STEPS (default 600, matching existing config)
6. Denormalize → plot_stroke → PNG
```

**Note on 600-step cap:** The existing app already uses `MAX_GEN_STEPS=600`. The IAM dataset's 95th-percentile stroke length should be verified against this cap before training — if sequences exceed 600 steps consistently, training will use the same cap for consistency with inference.

### Style Interpolation
Given two style latents z1, z2:
```
z = (1 - α) · z1 + α · z2
```
Works because the VAE regularizes the latent space to be smooth (N(0,I) prior).

---

## Training Schedule

| Stage | Epochs | β (KL weight) | Batch size | Notes |
|---|---|---|---|---|
| 1 | 0–20 | 0.0 | 32 | Pure reconstruction, decoder stabilizes |
| 2 | 20–60 | 0.0 → 1.0 (linear) | 32 | KL annealing |
| 3 | 60–100 | 1.0 | 32 | LR decay, full VAE |

- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **LR schedule:** Cosine annealing over 100 epochs
- **Gradient clipping:** global norm clipping at 1.0
- **Device:** MPS (`torch.device("mps")` on M4 Max)
- **Estimated training time:** ~3-5 hours total on M4 Max MPS

---

## Checkpointing

Saves every epoch to `checkpoints/transformer/checkpoint_latest.pt`:
```python
{
  "epoch": int,
  "model_state": state_dict,
  "optimizer_state": state_dict,
  "scheduler_state": state_dict,
  "best_val_loss": float,
  "beta": float,          # current KL annealing value
}
```

Separately saves `checkpoints/transformer/checkpoint_best.pt` when validation loss improves. Resume loads `checkpoint_latest.pt` and starts from `epoch + 1`. If killed mid-epoch, that epoch restarts cleanly.

---

## Inference Optimization (Laptop Server)

**INT8 Dynamic Quantization** (applied at deployment, no retraining):
```python
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
)
```
Covers both the Transformer's Linear layers and the Style VAE's BiLSTM, which runs per-request to encode canvas strokes.

**Expected inference time on i7-1185G7:**

| | Current LSTM (FP32) | New Transformer (INT8) |
|---|---|---|
| Model weights | 36MB | ~3MB |
| 600 steps, 1 sample | ~15-20s | ~6-9s |
| 600 steps, 3 samples | ~45-60s | ~18-27s |

**PyTorch thread config (unchanged):**
```python
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
```

---

## Codebase Integration

### New files
```
models/transformer_synthesis.py   — TextEncoder, StyleVAE, StrokeDecoder, MDNHead,
                                     HandWritingSynthesisTransformer (wraps all four)
train_transformer.py              — training script with checkpointing
```

### Minimal changes to existing files

**`app/core/config.py`** — add one field:
```python
model_type: str = os.getenv("MODEL_TYPE", "lstm")  # "lstm" or "transformer"
```

**`app/core/singletons.py`** — `ModelSingleton.get()` branches on `model_type`:
```python
if model_type == "transformer":
    from models.transformer_synthesis import HandWritingSynthesisTransformer
    model = HandWritingSynthesisTransformer(...)
else:
    model = HandWritingSynthesisNet(window_size=vocab_size)
```

**`generate.py`** — `generate_conditional_sequence()` adapter: when the model is a `HandWritingSynthesisTransformer`, `prime_seq` is passed to the Style VAE encoder (not used for LSTM priming), `real_text` is ignored (text conditioning is handled internally), and the `phi` return value is an empty array (attention maps are not meaningful for Transformers). The call signature seen by `app/services/generation.py` is unchanged.

### All other app files
```
app/api/          — unchanged
app/services/     — unchanged
app/core/session.py — unchanged
utils/            — unchanged
```

### Swapping in production
```bash
MODEL_PATH=checkpoints/transformer/checkpoint_best.pt
MODEL_TYPE=transformer
```

---

## Quality Improvements Over Current Model

| Issue | Current (LSTM) | New (Transformer + VAE) |
|---|---|---|
| Shaky strokes | LSTM gradient issues on long seqs | Transformer self-attention, full context |
| Text fidelity | Gaussian window misses chars | Learned cross-attention, unconstrained |
| Style drift | Hidden state fades after ~50 steps | z injected at every decoder layer |
| Style interpolation | Not possible | z1 → z2 linear interpolation |

---

## Out of Scope

- Diffusion-based generation (too slow for CPU inference target)
- Hierarchical word/stroke decomposition (overengineered for demo)
- Changing the frontend or API routes
- Rewriting the data loading pipeline
