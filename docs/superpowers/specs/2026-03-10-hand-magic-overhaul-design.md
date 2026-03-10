# Hand Magic — Full Overhaul Design Spec
**Date:** 2026-03-10
**Author:** Omar Musayev
**Status:** Approved

---

## Overview

Complete overhaul of the handwriting synthesis web app:
- Rename to **Hand Magic**
- Replace Flask with **FastAPI**
- Remove all authentication (login, signup, user accounts, database)
- Replace per-request dataset loading with **singletons**
- Add **async generation** with live sample streaming
- Add **multi-style session management** (persistent, no login)
- New **retro synth UI** design
- Package as a clean **open source project**

---

## 1. Architecture

### Stack
- **FastAPI** + Uvicorn (replaces Flask)
- **Jinja2** templates (same engine, new designs)
- **No database** — SQLAlchemy, Flask-Bcrypt, Flask-WTF removed entirely
- **Long-lived cookie** (30-day UUID token) identifies sessions on disk

### Singletons (loaded once at startup)
| Singleton | Contents | Current problem it fixes |
|---|---|---|
| `ModelSingleton` | `HandWritingSynthesisNet` weights | Already existed; port to FastAPI |
| `VocabSingleton` | `char_to_id`, `idx_to_char` | Full `HandwritingDataset` loaded every request |
| `StatsSingleton` | `train_mean`, `train_std` | Recomputed from style data per request (wrong) |

The `VocabSingleton` reads only the vocabulary from the dataset once at startup and discards all stroke data. This eliminates the primary RAM spike.

### Key Bug Fixes
1. `data_normalization()` in `priming.py` computes local mean/std from the user's style strokes and uses that for denormalization — wrong. The model was trained on dataset-wide stats. Fix: always use `StatsSingleton.train_mean/std`.
2. Double `SQLAlchemy` instantiation in `__init__.py` and `routes.py` — eliminated by removing the DB entirely.
3. Generation capped at **600 steps** (down from 2000) — configurable via env var.

---

## 2. Session & Style Storage

### Session Identity
- On first visit, server sets a `session_token` cookie (UUID4, 30-day expiry)
- All user data lives under `/disk/sessions/{token}/`
- Sessions older than 7 days cleaned up on startup
- Max 10 styles per session

### Disk Layout
```
/disk/sessions/{token}/
  styles/
    {style_id}/
      stroke.npy
      preview.png
      meta.json     # {"name": "Style 1", "created_at": "2026-03-10T12:00:00"}
  jobs/
    {job_id}/
      status.json   # {"status": "running|done|error", "done": 2, "total": 5}
      sample_0.png
      sample_1.png
      ...
```

### Style Naming
- Auto-named "Style 1", "Style 2", etc. on creation
- Users can rename inline (click name → edit in place)
- Delete button per style

---

## 3. API Routes

```
GET  /                          → Serve main single-page HTML
GET  /api/styles                → List session styles (JSON)
POST /api/styles                → Save new style from canvas path data
PATCH /api/styles/{id}          → Rename style
DELETE /api/styles/{id}         → Delete style

POST /api/generate              → Start generation job → {job_id}
GET  /api/jobs/{id}             → Poll job status → {status, done, total}
GET  /api/jobs/{id}/sample/{n}  → Fetch sample n as base64 PNG
```

---

## 4. Async Generation Flow

1. User submits Generate form → `POST /api/generate` → server creates `job_id`, spawns background thread, returns `{job_id}` immediately
2. Frontend polls `GET /api/jobs/{id}` every 1.5 seconds
3. Background thread generates samples one-by-one:
   - Writes `sample_N.png` to disk as each completes
   - Increments `done` counter in `status.json`
4. Frontend fetches and renders each new sample as it becomes available — samples appear progressively, not all at once
5. When `status = "done"`, polling stops
6. On error, `status = "error"` with message

---

## 5. UI Design — Retro Synth

### Brand
- **Site name:** HAND MAGIC
- **Credit:** Made by Omar Musayev
- **Aesthetic:** Retro synthwave — neon glows, dark backgrounds, CRT effects, perspective grid

### Palette
```
Background:   #08080f
Grid overlay: #1a1a3a
Cyan:         #00f5ff   (primary — borders, highlights, glow)
Magenta:      #ff00cc   (secondary — accents, hover)
Purple:       #7700ff   (tertiary)
Text:         #c8d0ff
Fonts:        "VT323" (headers/logo), "Share Tech Mono" (body/inputs)
```

### Layout (single page, three zones)
```
┌─────────────────────────────────────────────┐
│  ✦ HAND MAGIC ✦          Made by Omar Musayev│  ← Fixed header
├──────────────┬──────────────────────────────┤
│  STYLE LAB   │   DRAW YOUR STYLE            │  ← Studio panel
│              │   ┌─────────────────────┐    │
│  [Style 1 ▼] │   │  canvas             │    │
│  [Style 2 ▼] │   └─────────────────────┘    │
│  [+ New]     │   [CLEAR]  [SAVE STYLE]       │
├──────────────┴──────────────────────────────┤
│  GENERATE                                   │  ← Generate panel
│  Text: [________________________]           │
│  Style: ( Style 1 ) ( Style 2 )             │
│  Bias: ──●────────  3.5                     │
│  [ ▶ SYNTHESIZE ]                           │
│  ░░░░░░░░░░  generating...  [progress bar]  │
│  [ img1 ] [ img2 ] [ img3 ] ...             │  ← appear one by one
└─────────────────────────────────────────────┘
```

### Visual Effects
- "HAND MAGIC" title: animated cyan→magenta gradient with outer glow pulse
- Panels: dark card, `1px` cyan border, subtle cyan `box-shadow`
- Buttons: dark fill + neon border; hover = neon fill + bloom
- Canvas: dark background, white stroke lines, cyan border
- CRT scanline overlay: `repeating-linear-gradient` at 15% opacity over full page
- Background: CSS perspective grid fading toward center
- Generated images: neon-framed grid; click to expand

---

## 6. Open Source Packaging

### Files Added/Updated
| File | Purpose |
|---|---|
| `README.md` | Setup, local dev, deployment, model weights info |
| `LICENSE` | MIT |
| `pyproject.toml` | Replaces `environment.yml`; defines dependencies and package |
| `.env.example` | Documents all environment variables |
| `Dockerfile` | Cleaned up for both local dev and self-hosting |
| `docker-compose.yml` | Local dev with volume mounts |
| `.gitignore` | Updated — excludes sessions, generated files, `.env` |
| `CONTRIBUTING.md` | Brief contribution guide |

### Model Weights & Data
- `strokes.npy`, `sentences.txt`, and `.pt` checkpoints remain in the repo
- `.gitignore` updated to exclude generated session files and temp outputs
- README documents what each data file is and where it came from

### Environment Variables (`.env.example`)
```
DISK_STORAGE_PATH=/disk         # Where session/style data is stored
MAX_GEN_STEPS=600               # Max LSTM generation steps
SESSION_TTL_DAYS=7              # Days before session cleanup
MAX_STYLES_PER_SESSION=10
SECRET_KEY=changeme             # FastAPI session signing
```

---

## 7. Files Removed
- `app/models.py` — User, SavedSample DB models
- `app/forms.py` — LoginForm
- `config.py` — SQLAlchemy config
- `app/__init__.py` → replaced by FastAPI app factory
- All login/signup/logout/profile routes and templates

## 8. Files Kept (ported/cleaned)
- `models/models.py` — unchanged (pure PyTorch)
- `utils/` — unchanged except `constants.py` (StatsSingleton)
- `generate.py` — unchanged
- `train.py` — unchanged
- `app/xml_parser.py` — kept, minor cleanup
- `results/synthesis/*.pt` — kept
- `data/` — kept as-is
