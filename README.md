# ✦ HAND MAGIC ✦

> Retro-synth handwriting synthesis web app powered by LSTM + Gaussian Mixture Models.  
> Made by **Omar Musayev**

Generate realistic handwriting from any text using deep learning. Draw your own style on the canvas and prime the model to mimic it.

---

## Features

- **No login required** — session persists via a long-lived cookie
- **Multi-style management** — save up to 10 hand-drawn styles, rename or delete them
- **Async generation** — samples stream progressively as each one is ready
- **Retro synthwave UI** — neon glows, CRT scanlines, perspective grid
- **FastAPI + Uvicorn** backend (replaces Flask)
- **RAM-efficient singletons** — model, vocab, and normalization stats loaded once at startup

---

## Quick Start (local)

```bash
git clone https://github.com/your-handle/hand-magic.git
cd hand-magic
pip install -e ".[dev]"
cp .env.example .env
uvicorn main:app --reload
```

Open http://localhost:8000

---

## Docker

```bash
docker compose up --build
```

Open http://localhost:8000

---

## Environment Variables

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `DISK_STORAGE_PATH` | `/disk` | Where session/style data is stored |
| `MODEL_PATH` | `results/synthesis/best_model_synthesis_4.pt` | Path to model checkpoint |
| `DATA_PATH` | `data/` | Path to `strokes.npy` + `sentences.txt` |
| `MAX_GEN_STEPS` | `600` | Max LSTM generation steps |
| `SESSION_TTL_DAYS` | `7` | Days before session cleanup |
| `MAX_STYLES_PER_SESSION` | `10` | Styles per session |
| `N_SAMPLES` | `5` | Samples generated per request |

---

## Model Weights & Data

| File | Description |
|---|---|
| `data/strokes.npy` | IAM On-Line Handwriting stroke arrays `[eos, dx, dy]` |
| `data/sentences.txt` | Corresponding transcriptions |
| `results/synthesis/best_model_synthesis_4.pt` | Trained `HandWritingSynthesisNet` checkpoint |

The model is a 3-layer LSTM with a soft attention window, trained to predict bivariate Gaussian mixture offsets conditioned on text. Architecture from [Alex Graves (2013)](https://arxiv.org/abs/1308.0850).

---

## Running Tests

```bash
pytest -v
```

---

## Project Structure

```
hand-magic/
├── main.py                   # FastAPI app entry point
├── app/
│   ├── api/                  # Route handlers (styles, generate, jobs)
│   ├── core/                 # Config, singletons, session management
│   ├── services/             # Generation worker, job store, cleanup
│   ├── static/               # CSS + JS
│   └── templates/            # Jinja2 HTML
├── models/models.py          # HandWritingSynthesisNet (pure PyTorch)
├── utils/                    # Dataset, normalization, plotting helpers
├── generate.py               # CLI generation script
├── train.py                  # CLI training script
└── data/                     # Stroke data
```

---

## License

MIT — see [LICENSE](LICENSE).
