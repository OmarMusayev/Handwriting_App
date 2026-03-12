# AI Handwriting Generator

> Generate realistic handwriting from any text using deep learning — with style transfer from your own handwriting.
> Built by **Omar Musayev**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is this?

A deep learning system that generates realistic handwritten text from any input string. Draw your own handwriting style on the canvas and the model will mimic it — or use the default style.

**Two model architectures included:**
- A classic **LSTM + Gaussian Mixture Model** (based on [Alex Graves 2013](https://arxiv.org/abs/1308.0850)) — production-ready
- A modern **Transformer + Style VAE** — trained on 41,000+ handwriting samples, currently in training on an NVIDIA H100 SXM

---

## Demo

- Draw your handwriting style on the canvas
- Type any text
- Get multiple generated handwriting samples, streamed progressively as each one finishes
- Save and manage up to 10 styles per session — no login required

---

## Features

- **Style transfer** — draw on canvas, model mimics your handwriting
- **No login required** — session persists via cookie
- **Multi-style management** — save, rename, delete up to 10 styles
- **Async generation** — samples stream as they're ready
- **Two model backends** — LSTM (fast, stable) or Transformer + Style VAE (richer style control)
- **FastAPI + Uvicorn** backend
- **Retro synthwave UI** — neon glows, CRT scanlines, perspective grid

---

## Quick Start

```bash
git clone https://github.com/OmarMusayev/ai-handwriting-generator.git
cd ai-handwriting-generator
python -m venv env && source env/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Open http://localhost:8000

**Use the Transformer model:**

```bash
MODEL_TYPE=transformer MODEL_PATH=checkpoints/transformer/checkpoint_best.pt \
  python -m uvicorn main:app --reload --port 8000
```

---

## How It Works

### LSTM Model (Graves 2013)

A 3-layer LSTM with a soft attention window over the input text. At each step it predicts a bivariate Gaussian mixture distribution over the next pen offset `(dx, dy)` and an end-of-stroke flag. The attention window tracks which character is being written.

### Transformer + Style VAE

A modern architecture with four components:

| Component | Architecture | Purpose |
|---|---|---|
| **TextEncoder** | 4-layer Transformer encoder, d_model=256, 8 heads | Encodes input text into context vectors |
| **StyleVAE** | BiLSTM (2 layers, hidden=256) → 64-dim latent | Encodes handwriting style into a latent vector z |
| **StrokeDecoder** | 6-layer causal Transformer decoder | Autoregressively generates strokes conditioned on text + style |
| **MDNHead** | Linear(256 → 121) | Outputs Gaussian mixture parameters for next stroke |

Training uses **KL annealing** (β-VAE): β=0 for epochs 0–19, linear ramp 0→1 for epochs 20–60, β=1 after epoch 60. This prevents posterior collapse in the style encoder.

Loss: `(NLL + β·KL) / n`

---

## Training

### Data

| Dataset | Samples | Format |
|---|---|---|
| [IAM On-Line Handwriting DB](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) | 6,000 sentences | `[eos, dx, dy]` stroke arrays |
| [DeepWriting](https://ait.ethz.ch/deepwriting) | 35,000+ samples | Converted + rescaled to IAM format |
| **Combined** | **41,000+** | Used for Transformer training |

### Hardware

**Cloud (Transformer):**
- NVIDIA H100 SXM 80GB — RunPod
- batch_size=256, ~20 min/epoch, 300 epochs in ~4 hours

**Local (LSTM / prototyping):**
- Apple MacBook Pro M4 Max, 48GB unified RAM
- Apple MPS, batch_size=8, ~31 min/epoch

### Train the Transformer

```bash
# Fresh run
python train_transformer.py \
  --epochs 300 --batch_size 256 --max_stroke_len 1000 \
  --checkpoint_dir checkpoints/transformer/ \
  --deepwriting_path deepwriting_dataset/ --tqdm

# Resume
python train_transformer.py \
  --epochs 300 --batch_size 256 --max_stroke_len 1000 \
  --checkpoint_dir checkpoints/transformer/ \
  --deepwriting_path deepwriting_dataset/ --tqdm --resume
```

### Train the LSTM

```bash
python train.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_TYPE` | `lstm` | `lstm` or `transformer` |
| `MODEL_PATH` | `results/synthesis/best_model_synthesis_4.pt` | Checkpoint path |
| `DATA_PATH` | `data/` | Path to stroke data |
| `DISK_STORAGE_PATH` | `./disk` | Session/style storage |
| `MAX_GEN_STEPS` | `600` | Max generation steps |
| `N_SAMPLES` | `5` | Samples per request |
| `SESSION_TTL_DAYS` | `7` | Session cookie lifetime |

---

## Project Structure

```
ai-handwriting-generator/
├── main.py                        # FastAPI app entry point
├── train.py                       # LSTM training
├── train_transformer.py           # Transformer + Style VAE training
├── generate.py                    # CLI generation (both models)
├── app/
│   ├── api/                       # REST endpoints (styles, generate, jobs)
│   ├── core/                      # Config, singletons, session
│   ├── services/                  # Generation worker, job store, cleanup
│   ├── static/                    # CSS + JS
│   └── templates/                 # Jinja2 HTML
├── models/
│   ├── models.py                  # HandWritingSynthesisNet (LSTM)
│   └── transformer_synthesis.py   # HandWritingSynthesisTransformer
├── utils/                         # Dataset, normalization, plotting
├── checkpoints/transformer/       # Transformer checkpoints
├── results/synthesis/             # LSTM checkpoints
└── data/                          # Stroke data (not in repo — see Training)
```

---

## References

- Alex Graves — [Generating Sequences With Recurrent Neural Networks (2013)](https://arxiv.org/abs/1308.0850)
- Aksan et al. — [DeepWriting: Making Digital Ink Editable via Deep Generative Modeling (2018)](https://ait.ethz.ch/deepwriting)
- IAM On-Line Handwriting Database — [FKI Group, University of Bern](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)

---

## License

MIT — see [LICENSE](LICENSE).
