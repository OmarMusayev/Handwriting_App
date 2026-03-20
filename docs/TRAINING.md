# Training Guide

This document covers the architecture, data pipeline, and training process for both models. You don't need any of this to run the app — the trained weights are included in `weights/`. This is for anyone who wants to understand or reproduce the training.

---

## LSTM Model (Graves 2013)

### Architecture

The LSTM model (`models/models.py`) follows the architecture from [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850).

**HandWritingSynthesisNet:**

| Component | Details |
|---|---|
| LSTM layers | 3 stacked, hidden_size=400 each |
| Input | `[x, y, pen_up]` offsets + attention window vector |
| Skip connections | Input is concatenated to each layer |
| Output layer | Linear(1200 → 121) from all 3 LSTM hidden states |
| Attention | Soft Gaussian window with K=10 mixture components |

**Output distribution (121 dimensions):**
- 1 dimension: end-of-stroke logit (sigmoid → probability)
- 20 dimensions: mixture weights (softmax over M=20 Gaussian components)
- 20 dimensions: mean x (μ₁) for each component
- 20 dimensions: mean y (μ₂) for each component
- 20 dimensions: log std x (log σ₁) for each component
- 20 dimensions: log std y (log σ₂) for each component
- 20 dimensions: correlation coefficient (tanh → ρ)

At each step the model predicts a mixture of 20 bivariate Gaussians for the next pen offset, plus a Bernoulli for end-of-stroke.

**Attention mechanism:**

The window layer takes the top LSTM hidden state and outputs 3×K=30 parameters (α, β, κ) that define K=10 Gaussian functions over text positions. These are used to compute a soft attention distribution φ over the input text, which produces a window vector that tells the model which character it's currently writing.

### Training

| Parameter | Value |
|---|---|
| Optimizer | Adam, lr=0.001 |
| Scheduler | StepLR, step_size=100, gamma=0.1 |
| Gradient clipping | 10.0 on LSTM params, output clamped to [-100, 100] |
| Batch size | 32 |
| Epochs | 100 (early stopping, patience=15) |
| Loss | Negative log-likelihood of the Gaussian mixture |
| Seed | 212 |

**Loss function:**

The NLL loss computes the negative log-probability of the true next offset under the predicted Gaussian mixture distribution, plus a binary cross-entropy term for the end-of-stroke prediction. Computed per-timestep, masked for sequence padding.

### Data

The LSTM was trained on the original Graves handwriting dataset:
- `data/strokes.npy` — NumPy object array of variable-length sequences, each `(T, 3)` with `[eos, dx, dy]` format
- `data/sentences.txt` — Corresponding text, one sentence per line
- ~6,000 samples, 90/10 train/valid split
- Normalization: global mean/std computed on training set dx/dy values

### How to retrain

```bash
python train.py --model_type synthesis --n_epochs 100 --batch_size 32
```

Best model is saved to `results/synthesis/best_model_synthesis.pt`.

---

## Transformer Model (IAMOnDB)

### Architecture

The transformer model (`handwriting/model.py`) is a cross-attention GPT-style decoder that generates handwriting as a sequence of discrete stroke tokens.

**CrossAttentionGPT:**

| Component | Details |
|---|---|
| Text embedding | (vocab_size, d_model=384) + learned positional embedding |
| Stroke embedding | (stroke_vocab_size, d_model=384) + learned positional embedding |
| Decoder blocks | 6 layers |
| Each block | self-attention → cross-attention → MLP |
| Attention heads | 6 per layer |
| MLP | Linear(384 → 1536) → GELU → Dropout → Linear(1536 → 384) |
| Dropout | 0.1 throughout |
| Output | LayerNorm → Linear(384 → stroke_vocab_size) |
| Max text length | 64 characters |
| Max stroke tokens | 1050 |

**Total parameters:** ~15M

Unlike the LSTM which outputs continuous Gaussian mixture parameters, this model outputs logits over a discrete stroke token vocabulary and is trained with standard cross-entropy loss.

### Stroke Tokenization (Polar Offset Tokenizer)

Instead of predicting continuous (dx, dy) offsets, strokes are tokenized into discrete tokens using polar coordinates:

1. **Convert to polar:** Each offset `(dx, dy)` → `(θ, r)` where `θ = atan2(dy, dx)` and `r = sqrt(dx² + dy²)`

2. **Quantize angle:** θ is quantized into 128 bins covering [0, 2π). Token ID = 3 + angle_bin

3. **Quantize radius + pen flag:** The radius is quantized into 64 bins using data-driven codebooks (bin edges learned from training data). Separate codebooks for pen-down vs pen-up strokes. The pen flag is encoded jointly with the radius.

4. **2-token scheme:** Each stroke offset becomes exactly 2 tokens: `[angle_token, radius+flag_token]`

**Vocabulary layout:**
```
ID 0:         PAD
ID 1:         BOS (begin of sequence)
ID 2:         EOS (end of sequence)
IDs 3-130:    128 angle bins
IDs 131-258:  64 radius bins (pen-down)
IDs 259-322:  64 radius bins (pen-up)
```

Total stroke vocab: ~323 tokens.

**Decoding:** Tokens are decoded back to offsets by looking up the angle bin center and the empirical median radius for each bin, then converting back to Cartesian `(dx, dy)` via `dx = r·cos(θ), dy = r·sin(θ)`.

### Text Encoding

Characters are encoded at the character level with a vocabulary of ~98 tokens:
- 4 special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- ~94 characters: letters (uppercase + lowercase), digits, punctuation, space
- Input text gets `<bos>` prepended and `<eos>` appended
- Truncated to max 64 characters

### Generation (Inference)

Generation is autoregressive: starting from a BOS token, the model predicts the next stroke token one at a time until it produces an EOS token or hits the max length (1050 tokens).

**Sampling strategy:**
- Temperature: 0.9 (slightly sharpen the distribution)
- Top-k: 20 (only sample from the 20 most likely next tokens)
- Each sample uses a different random seed for variety

This means each generation request produces multiple unique handwriting samples of the same text.

### Training Data

**IAM On-Line Handwriting Database:**
- Source: [FKI Group, University of Bern](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
- ~13,000 handwriting samples from 500+ writers
- Raw format: XML files with timestamped pen coordinates captured on a Wacom tablet
- Free registration required for download

**Data processing pipeline** (in order):

1. **Raw extraction** (`build_iam_ondb_dataset.py`): Parse XML files, extract stroke coordinates, pair with text transcriptions, convert to offset format `[dx, dy, pen_flag]`

2. **Quality filtering**: Remove samples with timing anomalies (dots/crosses written out of sequence), samples that are too short or too long, and samples with transcription issues

3. **Writer filtering**: Select 75 writers with the most samples. Split per-writer to ensure no writer appears in both train and test sets

4. **Tokenization**: Build radius codebooks from training data (compute quantile-based bin edges), then tokenize all stroke sequences into polar token sequences

5. **Final splits**:
   - Train: ~9MB of tokenized stroke sequences
   - Validation: ~1.2MB
   - Test: ~1.1MB

### Training

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | Cosine schedule with warmup |
| Batch size | Varies by hardware (128-256 on GPU) |
| Epochs | 55 (best checkpoint by validation loss) |
| Loss | Cross-entropy on next-token prediction |
| Hardware | NVIDIA H100 SXM 80GB (RunPod) |
| Training time | ~4 hours for 60 epochs |
| d_model | 384 |
| n_layers | 6 |
| n_heads | 6 |

**How to retrain:**

1. Download IAM On-Line DB from the FKI website (requires registration)
2. Run the data processing pipeline to build tokenized datasets
3. Configure an experiment JSON in `configs/experiments/`
4. Run training (the training code lives in the `handwriting/` package — `training.py` and `optim.py` were stripped from this repo for deployment, but are in the full IAMOnDB research repo)

---

## Model Comparison

| | LSTM | Transformer |
|---|---|---|
| Parameters | ~5M | ~15M |
| Weight file | 14MB | 102MB |
| Output type | Continuous (Gaussian mixture) | Discrete (token vocabulary) |
| Style transfer | Yes (prime with drawn strokes) | No (text-only) |
| Generation speed | ~1-3s per sample (CPU) | ~2-5s per sample (CPU) |
| Training data | 6K samples (Graves dataset) | 10K+ samples (IAM On-Line DB) |
| Inference controls | Bias (0-10), style selection | Top-k sampling (fixed at k=20) |
