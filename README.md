# AI Handwriting Generator

> Generate realistic handwriting from any text using deep learning — with style transfer from your own handwriting.
> Built by **Omar Musayev**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is this?

A web app that generates realistic handwritten text from any input string. Two model architectures:

- **LSTM + Gaussian Mixture Model** — classic approach (Graves 2013), supports style transfer from your own handwriting drawn on-canvas
- **Transformer + Polar Tokenizer** — trained on IAM On-Line Handwriting DB, generates multiple samples via top-k sampling

Toggle between models in the UI.

---

## Quick Start (Server)

```bash
git clone https://github.com/OmarMusayev/ai-handwriting-generator.git
cd ai-handwriting-generator
python -m venv env && source env/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://your-server:8000`

Everything needed to run is in the repo — both model weights are included in `weights/`.

---

## Features

- **Two model modes** — switch between LSTM and Transformer in the UI
- **Style transfer** (LSTM) — draw on canvas, model mimics your handwriting
- **Multi-sample generation** (Transformer) — each sample is unique via top-k=20 sampling
- **No login required** — session persists via cookie
- **Multi-style management** — save, rename, delete up to 10 styles
- **Async generation** — samples stream progressively as each finishes

---

## How It Works

### LSTM Model

A 3-layer LSTM with a soft attention window over the input text. At each step it predicts a bivariate Gaussian mixture distribution over the next pen offset `(dx, dy)` and an end-of-stroke flag. Style transfer works by priming the model with your drawn strokes.

Based on [Alex Graves — Generating Sequences With Recurrent Neural Networks (2013)](https://arxiv.org/abs/1308.0850).

### Transformer Model

A cross-attention GPT-style decoder (6 layers, 384-dim, 6 heads). Text is encoded character-by-character, and strokes are tokenized into polar coordinates (angle + radius tokens). Generation is autoregressive with top-k=20 sampling at temperature 0.9.

Trained on [IAM On-Line Handwriting DB](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) (epoch 55, best validation loss).

---

## Project Structure

```
ai-handwriting-generator/
├── main.py                  # FastAPI app entry point
├── generate.py              # CLI generation
├── train.py                 # LSTM training script
├── app/
│   ├── api/                 # REST endpoints (styles, generate, jobs)
│   ├── core/                # Config, singletons, session management
│   ├── services/            # Generation workers, job store, cleanup
│   ├── static/              # CSS + JS frontend
│   └── templates/           # Jinja2 HTML
├── handwriting/             # Transformer inference package (IAMOnDB)
│   ├── model.py             # CrossAttentionGPT architecture
│   ├── generation.py        # Token generation + plotting
│   ├── tokenizers.py        # Polar offset tokenizer
│   ├── data.py              # Text vocabulary
│   └── checkpoint.py        # Checkpoint loading
├── models/
│   └── models.py            # LSTM model (HandWritingSynthesisNet)
├── utils/                   # Dataset, normalization, plotting
├── weights/
│   ├── lstm.pt              # LSTM best model (14MB)
│   └── transformer.pt       # Transformer best model (102MB, Git LFS)
├── data/
│   ├── sentences.txt        # Text vocabulary for LSTM
│   └── strokes.npy          # Stroke data for LSTM normalization
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── tests/
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `weights/lstm.pt` | LSTM checkpoint |
| `TRANSFORMER_CHECKPOINT` | `weights/transformer.pt` | Transformer checkpoint |
| `DATA_PATH` | `./data/` | Path to LSTM stroke data |
| `DISK_STORAGE_PATH` | `./disk` | Session/style/job storage |
| `MAX_GEN_STEPS` | `600` | Max LSTM generation steps |
| `N_SAMPLES` | `5` | Samples per request |
| `SESSION_TTL_DAYS` | `7` | Session cookie lifetime |

---

## Training Data Sources

The model weights are included in the repo so you don't need the training data to run the app. If you want to retrain:

### IAM On-Line Handwriting Database
- **Source**: [FKI Group, University of Bern](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
- **What**: ~13,000 handwriting samples from 500+ writers, captured as pen coordinates
- **Format**: XML files with stroke coordinates, converted to `[dx, dy, pen_flag]` offset arrays
- **Used for**: Transformer model training (75 writers, filtered for quality)
- **Access**: Free registration required on the FKI website

### DeepWriting Dataset
- **Source**: [ETH Zurich — DeepWriting (Aksan et al. 2018)](https://ait.ethz.ch/deepwriting)
- **What**: 35,000+ handwriting samples
- **Format**: Converted and rescaled to match IAM offset format `[eos, dx, dy]`
- **Used for**: Additional LSTM training data

### LSTM Training Data (`data/`)
- `strokes.npy` — NumPy object array of variable-length stroke sequences, each `(T, 3)` with `[eos, dx, dy]`
- `sentences.txt` — Corresponding text for each stroke sequence (one per line)
- These come from the original Graves 2013 handwriting dataset

---

## References

- Alex Graves — [Generating Sequences With Recurrent Neural Networks (2013)](https://arxiv.org/abs/1308.0850)
- Aksan et al. — [DeepWriting: Making Digital Ink Editable via Deep Generative Modeling (2018)](https://ait.ethz.ch/deepwriting)
- IAM On-Line Handwriting Database — [FKI Group, University of Bern](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)

---

## License

MIT — see [LICENSE](LICENSE).
