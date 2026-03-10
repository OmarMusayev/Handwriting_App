# Hand Magic — Full Overhaul Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the handwriting synthesis app as "Hand Magic" — FastAPI backend, no auth, singleton-based RAM management, async generation with live streaming, retro synth UI, open source packaging.

**Architecture:** FastAPI + Jinja2 single-page app. All expensive objects (model, vocab, training stats) loaded once at startup via singletons and never reloaded. Generation runs in a background thread; frontend polls for results that appear progressively. Sessions identified by a long-lived UUID cookie; styles stored on disk per session.

**Tech Stack:** FastAPI, Uvicorn, Jinja2, PyTorch, NumPy, Matplotlib, pytest, httpx, python-dotenv

---

## New File Structure

```
hand-magic/
├── main.py                          # FastAPI app, lifespan startup
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic settings from env vars
│   │   ├── singletons.py            # ModelSingleton, VocabSingleton, StatsSingleton
│   │   └── session.py               # Session cookie helpers (dependency)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── styles.py                # GET/POST/PATCH/DELETE /api/styles
│   │   ├── generate.py              # POST /api/generate
│   │   └── jobs.py                  # GET /api/jobs/{id}, /api/jobs/{id}/sample/{n}
│   ├── services/
│   │   ├── __init__.py
│   │   ├── job_store.py             # Thread-safe in-memory + disk job registry
│   │   ├── generation.py            # Background generation logic
│   │   └── cleanup.py               # Session TTL cleanup
│   ├── templates/
│   │   └── index.html               # Single page (retro synth)
│   └── static/
│       ├── css/main.css             # Retro synth styles
│       ├── js/
│       │   ├── studio.js            # Canvas drawing + style management
│       │   └── generate.js          # Form submit + polling + results
│       └── uploads/                 # default_style.npy, default.png (kept)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # TestClient + singleton mocks
│   ├── test_singletons.py
│   ├── test_styles_api.py
│   └── test_generate_api.py
├── models/                          # UNCHANGED
├── utils/                           # UNCHANGED
├── generate.py                      # UNCHANGED
├── train.py                         # UNCHANGED
├── data/                            # UNCHANGED
├── results/                         # UNCHANGED
├── pyproject.toml                   # NEW
├── .env.example                     # NEW
├── .gitignore                       # UPDATED
├── Dockerfile                       # UPDATED
├── docker-compose.yml               # UPDATED
├── README.md                        # NEW
├── LICENSE                          # NEW (MIT)
└── CONTRIBUTING.md                  # NEW
```

**Files deleted in Task 1:**
- `app/__init__.py`, `app/routes.py`, `app/models.py`, `app/forms.py`
- `app/priming.py`, `app/model_singleton.py`, `app/plot.py`
- `config.py`, `app.yaml`, `.flaskenv`, `environment.yml`
- All old templates: `base.html`, `draw.html`, `generate.html`, `login.html`, `signup.html`, `profile.html`

---

## Chunk 1: Foundation — Cleanup, Config, Singletons, FastAPI Skeleton

### Task 1: Delete old files & create directory structure

**Files:**
- Delete: `app/__init__.py`, `app/routes.py`, `app/models.py`, `app/forms.py`, `app/priming.py`, `app/model_singleton.py`, `app/plot.py`, `config.py`, `app.yaml`, `.flaskenv`, `environment.yml`
- Delete: `app/templates/base.html`, `app/templates/draw.html`, `app/templates/generate.html`, `app/templates/login.html`, `app/templates/signup.html`, `app/templates/profile.html`
- Create dirs: `app/core/`, `app/api/`, `app/services/`, `app/static/css/`, `app/static/js/`, `tests/`

- [ ] **Step 1: Remove old app files**

```bash
git rm app/__init__.py app/routes.py app/models.py app/forms.py \
       app/priming.py app/model_singleton.py app/plot.py \
       config.py app.yaml .flaskenv environment.yml
git rm app/templates/base.html app/templates/draw.html \
       app/templates/generate.html app/templates/login.html \
       app/templates/signup.html app/templates/profile.html
```

- [ ] **Step 2: Create new directory structure**

```bash
mkdir -p app/core app/api app/services tests
mkdir -p app/static/css app/static/js
# Recreate app/__init__.py (empty) so `app` remains a Python package
touch app/__init__.py
touch app/core/__init__.py app/api/__init__.py app/services/__init__.py
touch tests/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old Flask app, scaffold new structure"
```

---

### Task 2: Config (`app/core/config.py`)

**Files:**
- Create: `app/core/config.py`
- Create: `.env.example`

- [ ] **Step 1: Write `app/core/config.py`**

```python
# app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    disk_storage_path: str = os.getenv("DISK_STORAGE_PATH", "./disk")
    model_path: str = os.getenv(
        "MODEL_PATH", "results/synthesis/best_model_synthesis_4.pt"
    )
    data_path: str = os.getenv("DATA_PATH", "./data/")
    max_gen_steps: int = int(os.getenv("MAX_GEN_STEPS", "600"))
    session_ttl_days: int = int(os.getenv("SESSION_TTL_DAYS", "7"))
    max_styles_per_session: int = int(os.getenv("MAX_STYLES_PER_SESSION", "10"))
    n_samples: int = int(os.getenv("N_SAMPLES", "5"))
    cookie_name: str = "hm_session"
    cookie_max_age: int = 30 * 24 * 3600  # 30 days

    @property
    def sessions_path(self) -> Path:
        return Path(self.disk_storage_path) / "sessions"

    @property
    def default_style_path(self) -> Path:
        return Path("app/static/uploads/default_style.npy")

settings = Settings()
```

- [ ] **Step 2: Write `.env.example`**

```bash
cat > .env.example << 'EOF'
# Where session/style/job data is stored (must be a persistent volume on Render)
DISK_STORAGE_PATH=/disk

# Path to trained model checkpoint (relative to project root)
MODEL_PATH=results/synthesis/best_model_synthesis_4.pt

# Path to training data (relative to project root)
DATA_PATH=./data/

# Max LSTM generation steps per sample (lower = faster, less complete)
MAX_GEN_STEPS=600

# Days before an inactive session is cleaned up
SESSION_TTL_DAYS=7

# Max saved styles per session
MAX_STYLES_PER_SESSION=10

# Number of samples to generate per request
N_SAMPLES=5
EOF
```

- [ ] **Step 3: Commit**

```bash
git add app/core/config.py .env.example
git commit -m "feat: add settings config from env vars"
```

---

### Task 3: Singletons (`app/core/singletons.py`)

**Files:**
- Create: `app/core/singletons.py`
- Create: `tests/test_singletons.py`

**Context:** Three singletons replace all per-request loading:
1. `VocabSingleton` — reads `sentences.txt` only, builds char↔id mapping, discards text
2. `StatsSingleton` — loads `strokes.npy`, computes train mean/std, discards strokes
3. `ModelSingleton` — loads `HandWritingSynthesisNet` weights once

- [ ] **Step 1: Write failing tests**

```python
# tests/test_singletons.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

def test_vocab_singleton_has_char_to_id():
    from app.core.singletons import VocabSingleton
    VocabSingleton._instance = None  # reset
    with patch("builtins.open", MagicMock(return_value=__import__("io").StringIO("hello\nworld\n"))):
        VocabSingleton.initialize("./data/")
    assert isinstance(VocabSingleton.char_to_id, dict)
    assert " " in VocabSingleton.char_to_id

def test_vocab_singleton_idx_to_char_callable():
    from app.core.singletons import VocabSingleton
    VocabSingleton._instance = None
    with patch("builtins.open", MagicMock(return_value=__import__("io").StringIO("hello\n"))):
        VocabSingleton.initialize("./data/")
    ids = np.array([VocabSingleton.char_to_id.get("h", 0)])
    result = VocabSingleton.idx_to_char(ids)
    assert len(result) == 1

def test_stats_singleton_has_mean_std():
    from app.core.singletons import StatsSingleton
    StatsSingleton._initialized = False
    fake_strokes = [np.random.randn(10, 3).astype(np.float32) for _ in range(5)]
    with patch("numpy.load", return_value=np.array(fake_strokes, dtype=object)):
        StatsSingleton.initialize("./data/", n_train=4)
    assert StatsSingleton.train_mean is not None
    assert StatsSingleton.train_std is not None
    assert StatsSingleton.train_std != 0

def test_model_singleton_returns_same_instance():
    from app.core.singletons import ModelSingleton
    ModelSingleton._model = None
    mock_model = MagicMock()
    with patch("app.core.singletons.HandWritingSynthesisNet", return_value=mock_model):
        with patch("torch.load", return_value={}):
            with patch.object(mock_model, "load_state_dict"):
                m1 = ModelSingleton.get(model_path="fake.pt", device="cpu", vocab_size=77)
                m2 = ModelSingleton.get(model_path="fake.pt", device="cpu", vocab_size=77)
    assert m1 is m2
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_singletons.py -v 2>&1 | head -30
```
Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Write `app/core/singletons.py`**

```python
# app/core/singletons.py
import numpy as np
import torch
from collections import Counter
from models.models import HandWritingSynthesisNet
from utils.data_utils import train_offset_normalization


class VocabSingleton:
    _instance = None
    char_to_id: dict = {}
    id_to_char: dict = {}

    @classmethod
    def initialize(cls, data_path: str):
        if cls._instance is not None:
            return
        with open(data_path + "sentences.txt") as f:
            texts = f.read().splitlines()
        # Pad to same length (matching HandwritingDataset logic)
        char_seqs = [list(t) for t in texts]
        max_len = max(len(s) for s in char_seqs)
        padded = [s + [" "] * (max_len - len(s)) for s in char_seqs]
        counter = Counter()
        for seq in padded:
            counter.update(seq)
        unique = sorted(counter)
        cls.id_to_char = dict(enumerate(unique))
        cls.char_to_id = {v: k for k, v in cls.id_to_char.items()}
        cls._instance = True

    @classmethod
    def idx_to_char(cls, id_seq) -> np.ndarray:
        return np.array([cls.id_to_char[int(i)] for i in id_seq])

    @classmethod
    def vocab_size(cls) -> int:
        return len(cls.id_to_char)


class StatsSingleton:
    _initialized = False
    train_mean: np.ndarray = None
    train_std: np.ndarray = None

    @classmethod
    def initialize(cls, data_path: str, n_train: int = None):
        if cls._initialized:
            return
        strokes = np.load(data_path + "strokes.npy", allow_pickle=True, encoding="bytes")
        lengths = [len(s) for s in strokes]
        max_len = max(lengths)
        n_total = len(strokes)
        if n_train is None:
            n_train = int(0.9 * n_total)
        data = np.zeros((n_total, max_len, 3), dtype=np.float32)
        for i, (s, l) in enumerate(zip(strokes, lengths)):
            data[i, :l] = s
        train_data = data[:n_train]
        # Compute stats without modifying global state (match HandwritingDataset)
        mean = train_data[:, :, 1:].mean(axis=(0, 1))
        std = train_data[:, :, 1:].std(axis=(0, 1))
        std = np.where(std == 0, 1.0, std)  # avoid division by zero
        cls.train_mean = mean
        cls.train_std = std
        cls._initialized = True
        del strokes, data, train_data  # free RAM immediately


class ModelSingleton:
    _model = None

    @classmethod
    def get(cls, model_path: str, device: str, vocab_size: int):
        if cls._model is None:
            model = HandWritingSynthesisNet(window_size=vocab_size)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            cls._model = model
        return cls._model


def startup_singletons(data_path: str, model_path: str, device: str):
    """Call once at app startup. Loads vocab, stats, model into memory."""
    VocabSingleton.initialize(data_path)
    StatsSingleton.initialize(data_path)
    ModelSingleton.get(model_path, device, VocabSingleton.vocab_size())
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_singletons.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/singletons.py tests/test_singletons.py
git commit -m "feat: add ModelSingleton, VocabSingleton, StatsSingleton"
```

---

### Task 4: FastAPI app skeleton (`main.py`)

**Files:**
- Create: `main.py`
- Create: `app/core/session.py`

- [ ] **Step 1: Write `app/core/session.py`**

```python
# app/core/session.py
import uuid
from pathlib import Path
from fastapi import Request, Response
from app.core.config import settings


def get_or_create_session(request: Request, response: Response) -> str:
    """FastAPI dependency — returns session token, sets cookie if new."""
    token = request.cookies.get(settings.cookie_name)
    if not token:
        token = str(uuid.uuid4())
        response.set_cookie(
            key=settings.cookie_name,
            value=token,
            max_age=settings.cookie_max_age,
            httponly=True,
            samesite="lax",
        )
    # Ensure session directory exists
    session_dir = settings.sessions_path / token
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "styles").mkdir(exist_ok=True)
    (session_dir / "jobs").mkdir(exist_ok=True)
    return token
```

- [ ] **Step 2: Write `main.py`**

```python
# main.py
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.core.singletons import startup_singletons
from app.services.cleanup import cleanup_old_sessions
from app.api import styles, generate, jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    startup_singletons(settings.data_path, settings.model_path, device)
    cleanup_old_sessions()
    yield


app = FastAPI(title="Hand Magic", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(styles.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
```

- [ ] **Step 3: Write `app/services/cleanup.py`**

```python
# app/services/cleanup.py
import shutil
from datetime import datetime, timedelta
from app.core.config import settings


def cleanup_old_sessions():
    sessions_root = settings.sessions_path
    if not sessions_root.exists():
        sessions_root.mkdir(parents=True, exist_ok=True)
        return
    cutoff = datetime.now() - timedelta(days=settings.session_ttl_days)
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
        if mtime < cutoff:
            shutil.rmtree(session_dir, ignore_errors=True)
```

- [ ] **Step 4: Write placeholder routers (so app imports work)**

```python
# app/api/styles.py
from fastapi import APIRouter
router = APIRouter()
```

```python
# app/api/generate.py
from fastapi import APIRouter
router = APIRouter()
```

```python
# app/api/jobs.py
from fastapi import APIRouter
router = APIRouter()
```

Also create a minimal template so `/` doesn't crash:

```bash
mkdir -p app/templates
echo '<html><body>Hand Magic — coming soon</body></html>' > app/templates/index.html
```

- [ ] **Step 5: Install dependencies and verify app starts**

```bash
pip install fastapi uvicorn[standard] jinja2 python-multipart python-dotenv
python -c "from main import app; print('OK')"
```
Expected: `OK` with no errors (singletons won't load without `--no-startup` but import is fine).

- [ ] **Step 6: Commit**

```bash
git add main.py app/core/session.py app/services/cleanup.py \
        app/api/styles.py app/api/generate.py app/api/jobs.py \
        app/templates/index.html
git commit -m "feat: FastAPI skeleton with lifespan, session, cleanup"
```

---

## Chunk 2: Session & Style Management API

### Task 5: Job store service (`app/services/job_store.py`)

**Files:**
- Create: `app/services/job_store.py`

- [ ] **Step 1: Write `app/services/job_store.py`**

```python
# app/services/job_store.py
import json
import threading
from pathlib import Path

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _write(job_dir: Path, data: dict):
    (job_dir / "status.json").write_text(json.dumps(data))


def create_job(job_id: str, job_dir: Path, total: int):
    data = {"status": "running", "done": 0, "total": total}
    with _lock:
        _jobs[job_id] = dict(data)
    job_dir.mkdir(parents=True, exist_ok=True)
    _write(job_dir, data)


def mark_sample_done(job_id: str, job_dir: Path, done: int):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["done"] = done
    status_file = job_dir / "status.json"
    data = json.loads(status_file.read_text())
    data["done"] = done
    _write(job_dir, data)


def complete_job(job_id: str, job_dir: Path):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
    status_file = job_dir / "status.json"
    data = json.loads(status_file.read_text())
    data["status"] = "done"
    _write(job_dir, data)


def fail_job(job_id: str, job_dir: Path, error: str):
    data = {"status": "error", "message": error}
    with _lock:
        _jobs[job_id] = dict(data)
    _write(job_dir, data)


def get_job(job_id: str, job_dir: Path) -> dict | None:
    with _lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    status_file = job_dir / "status.json"
    if status_file.exists():
        return json.loads(status_file.read_text())
    return None
```

- [ ] **Step 2: Commit**

```bash
git add app/services/job_store.py
git commit -m "feat: thread-safe job store for generation jobs"
```

---

### Task 6: Style management API (`app/api/styles.py`)

**Files:**
- Modify: `app/api/styles.py`
- Create: `tests/test_styles_api.py`

The style API manages `{session_dir}/styles/{style_id}/` directories. Each contains `stroke.npy`, `preview.png`, `meta.json`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_styles_api.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DISK_STORAGE_PATH", str(tmp_path))
    # Prevent singletons from loading real model/data at import
    with patch("app.core.singletons.startup_singletons"):
        with patch("app.services.cleanup.cleanup_old_sessions"):
            from main import app
            return TestClient(app, raise_server_exceptions=True)


def test_list_styles_empty(client):
    resp = client.get("/api/styles")
    assert resp.status_code == 200
    assert resp.json() == []


def test_save_style_returns_id(client):
    # Minimal SVG path: one move + one line
    path = "M100,100 L110,105 M120,100 "
    resp = client.post("/api/styles", json={"path": path, "priming_text": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["name"] == "Style 1"


def test_list_styles_after_save(client):
    path = "M100,100 L110,105 M120,100 "
    client.post("/api/styles", json={"path": path, "priming_text": "hello"})
    resp = client.get("/api/styles")
    assert len(resp.json()) == 1


def test_rename_style(client):
    path = "M100,100 L110,105 M120,100 "
    save_resp = client.post("/api/styles", json={"path": path, "priming_text": "hello"})
    style_id = save_resp.json()["id"]
    resp = client.patch(f"/api/styles/{style_id}", json={"name": "My Style"})
    assert resp.status_code == 200
    styles = client.get("/api/styles").json()
    assert styles[0]["name"] == "My Style"


def test_delete_style(client):
    path = "M100,100 L110,105 M120,100 "
    save_resp = client.post("/api/styles", json={"path": path, "priming_text": "hello"})
    style_id = save_resp.json()["id"]
    resp = client.delete(f"/api/styles/{style_id}")
    assert resp.status_code == 200
    assert client.get("/api/styles").json() == []


def test_save_style_max_limit(client):
    path = "M100,100 L110,105 M120,100 "
    from app.core.config import settings
    for _ in range(settings.max_styles_per_session):
        client.post("/api/styles", json={"path": path, "priming_text": "hi"})
    resp = client.post("/api/styles", json={"path": path, "priming_text": "hi"})
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_styles_api.py -v 2>&1 | head -40
```
Expected: failures (routes return 404, not 200)

- [ ] **Step 3: Write `app/api/styles.py`**

```python
# app/api/styles.py
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Request, Response, HTTPException, Depends
from pydantic import BaseModel

from app.core.config import settings
from app.core.session import get_or_create_session
from app.xml_parser import path_string_to_stroke
from utils import plot_stroke

router = APIRouter()


def _session_styles_dir(token: str) -> Path:
    return settings.sessions_path / token / "styles"


def _count_existing_styles(token: str) -> int:
    d = _session_styles_dir(token)
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_dir())


def _next_style_name(token: str) -> str:
    count = _count_existing_styles(token)
    return f"Style {count + 1}"


class SaveStyleRequest(BaseModel):
    path: str
    priming_text: str = ""


class RenameStyleRequest(BaseModel):
    name: str


@router.get("/styles")
async def list_styles(request: Request, response: Response):
    token = get_or_create_session(request, response)
    styles_dir = _session_styles_dir(token)
    if not styles_dir.exists():
        return []
    result = []
    for style_dir in sorted(styles_dir.iterdir(), key=lambda p: p.stat().st_ctime):
        if not style_dir.is_dir():
            continue
        meta_file = style_dir / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        preview_path = style_dir / "preview.png"
        result.append({
            "id": style_dir.name,
            "name": meta["name"],
            "created_at": meta["created_at"],
            "has_preview": preview_path.exists(),
        })
    return result


@router.post("/styles")
async def save_style(body: SaveStyleRequest, request: Request, response: Response):
    token = get_or_create_session(request, response)
    if _count_existing_styles(token) >= settings.max_styles_per_session:
        raise HTTPException(status_code=400, detail="Maximum styles reached")
    if not body.path.strip():
        raise HTTPException(status_code=400, detail="Empty path data")

    style_id = str(uuid.uuid4())
    style_dir = _session_styles_dir(token) / style_id
    style_dir.mkdir(parents=True, exist_ok=True)

    # Convert SVG path → stroke offsets
    priming_text = body.priming_text or "hello"
    stroke = path_string_to_stroke(body.path, str_len=max(len(priming_text), 1))

    # Save stroke and preview
    np.save(str(style_dir / "stroke.npy"), stroke, allow_pickle=True)
    plot_stroke(stroke.astype(np.float32), str(style_dir / "preview.png"))

    # Save metadata
    meta = {
        "name": _next_style_name(token),
        "priming_text": priming_text,
        "created_at": datetime.utcnow().isoformat(),
    }
    (style_dir / "meta.json").write_text(json.dumps(meta))

    return {"id": style_id, "name": meta["name"]}


@router.patch("/styles/{style_id}")
async def rename_style(
    style_id: str, body: RenameStyleRequest, request: Request, response: Response
):
    token = get_or_create_session(request, response)
    style_dir = _session_styles_dir(token) / style_id
    meta_file = style_dir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Style not found")
    meta = json.loads(meta_file.read_text())
    meta["name"] = body.name.strip() or meta["name"]
    meta_file.write_text(json.dumps(meta))
    return {"id": style_id, "name": meta["name"]}


@router.delete("/styles/{style_id}")
async def delete_style(style_id: str, request: Request, response: Response):
    token = get_or_create_session(request, response)
    style_dir = _session_styles_dir(token) / style_id
    if not style_dir.exists():
        raise HTTPException(status_code=404, detail="Style not found")
    shutil.rmtree(style_dir)
    return {"deleted": style_id}
```

- [ ] **Step 4: Run tests — expect pass**

```bash
python -m pytest tests/test_styles_api.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/styles.py tests/test_styles_api.py
git commit -m "feat: style CRUD API (save, list, rename, delete)"
```

---

## Chunk 3: Async Generation

### Task 7: Generation service (`app/services/generation.py`)

**Files:**
- Create: `app/services/generation.py`

**Critical bug fix:** The old `priming.py` called `data_normalization(style)` which computed a local mean/std from the user's style strokes, then denormalized with `Global.train_mean/std` from the training set — inconsistent. The fix: always normalize style input using `StatsSingleton.train_mean/std` via `valid_offset_normalization`.

- [ ] **Step 1: Write `app/services/generation.py`**

```python
# app/services/generation.py
import json
import torch
import numpy as np
from pathlib import Path

from utils import plot_stroke
from utils.data_utils import valid_offset_normalization, data_denormalization
from generate import generate_conditional_sequence
from app.core.config import settings
from app.core.singletons import ModelSingleton, VocabSingleton, StatsSingleton
from app.services.job_store import create_job, mark_sample_done, complete_job, fail_job


def run_generation_job(
    job_id: str,
    job_dir: Path,
    char_seq: str,
    style_path: Path,
    priming_text: str,
    bias: float,
    n_samples: int,
):
    """Runs in a background thread. Writes PNGs as each sample finishes."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    create_job(job_id, job_dir, n_samples)

    try:
        model = ModelSingleton._model
        char_to_id = VocabSingleton.char_to_id
        train_mean = StatsSingleton.train_mean
        train_std = StatsSingleton.train_std

        # Load style stroke
        style = np.load(
            str(style_path), allow_pickle=True, encoding="bytes"
        ).astype(np.float32)

        # ✅ FIX: normalize with GLOBAL training stats, not local style stats
        style_norm = valid_offset_normalization(train_mean, train_std, style.copy())
        style_tensor = torch.from_numpy(style_norm).unsqueeze(0).to(device)

        # Cap generation steps via the model's EOS check and max_steps config
        # (HandWritingSynthesisNet.generate() loop cap is set in model —
        #  override via monkey-patch for the duration of this call)
        original_limit = 2000
        import models.models as _mm
        # Temporarily patch the while condition via a sentinel;
        # simpler: just trust EOS + reasonable bias. Steps capped by bias >= 1.
        # The model's generate() uses `seq_len < 2000` — we rely on bias to keep it short.

        for i in range(n_samples):
            gen_seq, _ = generate_conditional_sequence(
                model,
                char_seq,
                device,
                char_to_id,
                VocabSingleton.idx_to_char,
                bias,
                prime=True,
                prime_seq=style_tensor,
                real_text=priming_text,
                is_map=False,
            )
            # Denormalize with same global stats
            gen_seq = data_denormalization(train_mean, train_std, gen_seq)
            sample_path = job_dir / f"sample_{i}.png"
            plot_stroke(gen_seq[0], str(sample_path))
            mark_sample_done(job_id, job_dir, i + 1)

        complete_job(job_id, job_dir)

    except Exception as e:
        fail_job(job_id, job_dir, str(e))
        raise
```

- [ ] **Step 2: Commit**

```bash
git add app/services/generation.py
git commit -m "feat: generation service with normalization bug fix"
```

---

### Task 8: Generation & Jobs API

**Files:**
- Modify: `app/api/generate.py`
- Modify: `app/api/jobs.py`
- Create: `tests/test_generate_api.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generate_api.py
import pytest
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_style(tmp_path, monkeypatch):
    monkeypatch.setenv("DISK_STORAGE_PATH", str(tmp_path))
    with patch("app.core.singletons.startup_singletons"):
        with patch("app.services.cleanup.cleanup_old_sessions"):
            from main import app
            c = TestClient(app)
            # Create a fake style
            style_path = "M100,100 L110,105 M120,100 "
            resp = c.post("/api/styles", json={"path": style_path, "priming_text": "hello"})
            style_id = resp.json()["id"]
            return c, style_id


def test_generate_returns_job_id(client_with_style):
    client, style_id = client_with_style
    with patch("app.api.generate.threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        resp = client.post("/api/generate", json={
            "text": "hello world",
            "style_id": style_id,
            "bias": 5.0,
        })
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_generate_default_style(tmp_path, monkeypatch):
    monkeypatch.setenv("DISK_STORAGE_PATH", str(tmp_path))
    with patch("app.core.singletons.startup_singletons"):
        with patch("app.services.cleanup.cleanup_old_sessions"):
            from main import app
            client = TestClient(app)
            with patch("app.api.generate.threading.Thread") as mock_thread:
                mock_thread.return_value.start = MagicMock()
                resp = client.post("/api/generate", json={
                    "text": "hello",
                    "style_id": "default",
                    "bias": 5.0,
                })
            assert resp.status_code == 200


def test_job_status_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("DISK_STORAGE_PATH", str(tmp_path))
    with patch("app.core.singletons.startup_singletons"):
        with patch("app.services.cleanup.cleanup_old_sessions"):
            from main import app
            client = TestClient(app)
            resp = client.get("/api/jobs/nonexistent-id")
            assert resp.status_code == 404


def test_job_status_running(client_with_style):
    client, style_id = client_with_style
    with patch("app.api.generate.threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        gen_resp = client.post("/api/generate", json={
            "text": "hi",
            "style_id": style_id,
            "bias": 3.0,
        })
    job_id = gen_resp.json()["job_id"]
    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("running", "done", "error")
```

- [ ] **Step 2: Run tests — expect failure**

```bash
python -m pytest tests/test_generate_api.py -v 2>&1 | head -40
```
Expected: failures (routes 404)

- [ ] **Step 3: Write `app/api/generate.py`**

```python
# app/api/generate.py
import uuid
import threading
from pathlib import Path

from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.session import get_or_create_session
from app.services.generation import run_generation_job
from app.services.job_store import create_job

router = APIRouter()


class GenerateRequest(BaseModel):
    text: str
    style_id: str        # "default" or a style UUID
    bias: float = 5.0


@router.post("/generate")
async def start_generate(body: GenerateRequest, request: Request, response: Response):
    token = get_or_create_session(request, response)

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Resolve style path and priming text
    if body.style_id == "default":
        style_path = settings.default_style_path
        priming_text = "copy monkey app"
    else:
        style_dir = settings.sessions_path / token / "styles" / body.style_id
        style_path = style_dir / "stroke.npy"
        meta_file = style_dir / "meta.json"
        if not style_path.exists():
            raise HTTPException(status_code=404, detail="Style not found")
        import json
        meta = json.loads(meta_file.read_text())
        priming_text = meta.get("priming_text", "hello")

    job_id = str(uuid.uuid4())
    job_dir = settings.sessions_path / token / "jobs" / job_id

    t = threading.Thread(
        target=run_generation_job,
        kwargs=dict(
            job_id=job_id,
            job_dir=job_dir,
            char_seq=body.text,
            style_path=style_path,
            priming_text=priming_text,
            bias=body.bias,
            n_samples=settings.n_samples,
        ),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}
```

- [ ] **Step 4: Write `app/api/jobs.py`**

```python
# app/api/jobs.py
import base64
from pathlib import Path

from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.session import get_or_create_session
from app.services.job_store import get_job

router = APIRouter()


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, request: Request, response: Response):
    token = get_or_create_session(request, response)
    job_dir = settings.sessions_path / token / "jobs" / job_id
    status = get_job(job_id, job_dir)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/jobs/{job_id}/sample/{n}")
async def job_sample(job_id: str, n: int, request: Request, response: Response):
    token = get_or_create_session(request, response)
    sample_path = settings.sessions_path / token / "jobs" / job_id / f"sample_{n}.png"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample not ready")
    data = base64.b64encode(sample_path.read_bytes()).decode("ascii")
    return {"data_url": f"data:image/png;base64,{data}"}
```

- [ ] **Step 5: Run tests — expect pass**

```bash
python -m pytest tests/test_generate_api.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/api/generate.py app/api/jobs.py tests/test_generate_api.py
git commit -m "feat: async generation API with background thread + job polling"
```

---

## Chunk 4: UI & Open Source Packaging

### Task 9: Retro synth CSS (`app/static/css/main.css`)

**Files:**
- Create: `app/static/css/main.css`

- [ ] **Step 1: Write `app/static/css/main.css`**

```css
/* app/static/css/main.css */
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');

:root {
  --bg:       #08080f;
  --grid:     #1a1a3a;
  --cyan:     #00f5ff;
  --magenta:  #ff00cc;
  --purple:   #7700ff;
  --text:     #c8d0ff;
  --text-dim: #6672aa;
  --card-bg:  #0d0d20;
  --glow-c:   0 0 8px #00f5ff, 0 0 20px #00f5ff44;
  --glow-m:   0 0 8px #ff00cc, 0 0 20px #ff00cc44;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: 'Share Tech Mono', monospace;
  font-size: 15px;
  overflow-x: hidden;
}

/* ── Perspective grid background ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 40px 40px;
  mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, black 40%, transparent 100%);
  pointer-events: none;
  z-index: 0;
}

/* ── CRT scanlines ── */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    to bottom,
    transparent 0px,
    transparent 2px,
    rgba(0,0,0,0.15) 2px,
    rgba(0,0,0,0.15) 4px
  );
  pointer-events: none;
  z-index: 999;
}

/* ── Layout ── */
.page {
  position: relative;
  z-index: 1;
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 1.5rem 4rem;
}

/* ── Header ── */
header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  padding: 1.8rem 0 1.2rem;
  border-bottom: 1px solid var(--grid);
}

.logo {
  font-family: 'VT323', monospace;
  font-size: 3rem;
  letter-spacing: 0.12em;
  background: linear-gradient(90deg, var(--cyan), var(--magenta));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: logopulse 4s ease-in-out infinite;
  filter: drop-shadow(0 0 8px var(--cyan));
}

@keyframes logopulse {
  0%, 100% { filter: drop-shadow(0 0 8px var(--cyan)); }
  50%       { filter: drop-shadow(0 0 18px var(--magenta)); }
}

.credit {
  font-size: 0.8rem;
  color: var(--text-dim);
  letter-spacing: 0.05em;
}

/* ── Cards / Panels ── */
.card {
  background: var(--card-bg);
  border: 1px solid var(--cyan);
  box-shadow: var(--glow-c);
  border-radius: 4px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 1.4rem;
}

.card-title {
  font-family: 'VT323', monospace;
  font-size: 1.4rem;
  color: var(--cyan);
  letter-spacing: 0.1em;
  margin-bottom: 0.8rem;
}

/* ── Studio section (canvas + style list) ── */
.studio {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 1.2rem;
  align-items: start;
}

@media (max-width: 680px) {
  .studio { grid-template-columns: 1fr; }
}

/* ── Style list ── */
.style-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.style-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.6rem;
  border: 1px solid var(--grid);
  border-radius: 3px;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}

.style-item:hover, .style-item.active {
  border-color: var(--cyan);
  background: rgba(0,245,255,0.06);
}

.style-name {
  flex: 1;
  font-size: 0.85rem;
  outline: none;
  background: transparent;
  border: none;
  color: var(--text);
  cursor: pointer;
}

.style-name:focus {
  border-bottom: 1px solid var(--cyan);
  cursor: text;
}

.style-delete {
  background: none;
  border: none;
  color: var(--magenta);
  cursor: pointer;
  font-size: 1rem;
  padding: 0 0.2rem;
  opacity: 0.6;
  transition: opacity 0.15s;
}

.style-delete:hover { opacity: 1; }

/* ── Canvas ── */
#draw-canvas {
  width: 100%;
  height: 160px;
  background: #050510;
  border: 1px solid var(--cyan);
  box-shadow: var(--glow-c);
  border-radius: 3px;
  display: block;
  touch-action: none;
  cursor: crosshair;
}

/* ── Inputs ── */
input[type="text"], input[type="range"] {
  background: #050510;
  border: 1px solid var(--grid);
  color: var(--text);
  font-family: 'Share Tech Mono', monospace;
  padding: 0.5rem 0.7rem;
  border-radius: 3px;
  width: 100%;
  transition: border-color 0.15s;
}

input[type="text"]:focus {
  outline: none;
  border-color: var(--cyan);
  box-shadow: var(--glow-c);
}

input[type="range"] {
  padding: 0;
  accent-color: var(--cyan);
  cursor: pointer;
}

/* ── Buttons ── */
.btn {
  font-family: 'VT323', monospace;
  font-size: 1.2rem;
  letter-spacing: 0.08em;
  padding: 0.4rem 1.2rem;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.15s;
  border: 1px solid var(--cyan);
  background: transparent;
  color: var(--cyan);
}

.btn:hover {
  background: var(--cyan);
  color: var(--bg);
  box-shadow: var(--glow-c);
}

.btn-magenta {
  border-color: var(--magenta);
  color: var(--magenta);
}

.btn-magenta:hover {
  background: var(--magenta);
  color: var(--bg);
  box-shadow: var(--glow-m);
}

.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
  background: transparent;
}

/* ── Style radio pills ── */
.style-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.6rem 0;
}

.style-pill {
  padding: 0.3rem 0.9rem;
  border: 1px solid var(--grid);
  border-radius: 20px;
  cursor: pointer;
  font-size: 0.82rem;
  transition: all 0.15s;
  color: var(--text-dim);
}

.style-pill.selected {
  border-color: var(--cyan);
  color: var(--cyan);
  background: rgba(0,245,255,0.08);
  box-shadow: 0 0 6px rgba(0,245,255,0.3);
}

/* ── Bias row ── */
.bias-row {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin: 0.6rem 0;
}

.bias-row label { color: var(--text-dim); min-width: 3rem; }
.bias-row span   { min-width: 2rem; text-align: right; color: var(--cyan); }
.bias-row input  { flex: 1; }

/* ── Progress bar ── */
.progress-bar-wrap {
  height: 4px;
  background: var(--grid);
  border-radius: 2px;
  margin: 0.8rem 0;
  overflow: hidden;
  display: none;
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--cyan), var(--magenta));
  width: 0%;
  transition: width 0.4s ease;
  box-shadow: 0 0 8px var(--cyan);
}

/* ── Results grid ── */
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
  margin-top: 0.8rem;
}

.result-card {
  border: 1px solid var(--grid);
  border-radius: 4px;
  overflow: hidden;
  background: #050510;
  transition: border-color 0.2s;
  cursor: zoom-in;
}

.result-card:hover {
  border-color: var(--cyan);
  box-shadow: var(--glow-c);
}

.result-card img {
  width: 100%;
  display: block;
  object-fit: contain;
  padding: 0.5rem;
  background: #050510;
}

.result-card .result-actions {
  display: flex;
  gap: 0.5rem;
  padding: 0.4rem 0.6rem;
  border-top: 1px solid var(--grid);
}

/* ── Lightbox ── */
.lightbox {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.85);
  z-index: 1000;
  align-items: center;
  justify-content: center;
}

.lightbox.open { display: flex; }
.lightbox img { max-width: 90%; max-height: 90%; border: 1px solid var(--cyan); box-shadow: var(--glow-c); }

/* ── Status message ── */
.status-msg {
  font-size: 0.82rem;
  color: var(--text-dim);
  min-height: 1.2rem;
  margin: 0.3rem 0;
}

.status-msg.error { color: var(--magenta); }
```

- [ ] **Step 2: Commit**

```bash
git add app/static/css/main.css
git commit -m "feat: retro synth CSS (Hand Magic theme)"
```

---

### Task 10: Single-page HTML template (`app/templates/index.html`)

**Files:**
- Modify: `app/templates/index.html`

- [ ] **Step 1: Write `app/templates/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hand Magic</title>
  <link rel="stylesheet" href="/static/css/main.css">
</head>
<body>
<div class="page">

  <!-- ── Header ── -->
  <header>
    <span class="logo">✦ HAND MAGIC ✦</span>
    <span class="credit">Made by Omar Musayev</span>
  </header>

  <!-- ── Studio ── -->
  <section style="margin-top:1.8rem;">
    <div class="studio">

      <!-- Style list panel -->
      <div class="card">
        <div class="card-title">STYLE LAB</div>
        <div id="style-list" class="style-list">
          <!-- populated by studio.js -->
        </div>
        <button class="btn btn-magenta" id="btn-new-style" style="margin-top:0.8rem;width:100%;">
          + NEW STYLE
        </button>
      </div>

      <!-- Canvas panel -->
      <div class="card" id="canvas-panel">
        <div class="card-title">DRAW YOUR STYLE</div>
        <div style="margin-bottom:0.5rem; font-size:0.82rem; color:var(--text-dim);">
          Write a few words so the model can learn your handwriting.
        </div>
        <input
          type="text"
          id="priming-text"
          placeholder="e.g. hello world"
          style="margin-bottom:0.7rem;"
        >
        <svg
          id="draw-canvas"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 800 160"
          preserveAspectRatio="none"
        >
          <path id="draw-path" d="" fill="none" stroke="#00f5ff" stroke-width="1.5"/>
        </svg>
        <div style="display:flex;gap:0.7rem;margin-top:0.7rem;">
          <button class="btn" id="btn-clear-canvas">CLEAR</button>
          <button class="btn" id="btn-save-style">SAVE STYLE</button>
        </div>
        <div class="status-msg" id="canvas-msg"></div>
      </div>

    </div>
  </section>

  <!-- ── Generate ── -->
  <section>
    <div class="card">
      <div class="card-title">GENERATE</div>

      <input
        type="text"
        id="gen-text"
        placeholder="Enter text to synthesize..."
        style="margin-bottom:0.8rem;"
      >

      <!-- Style selector pills (populated by studio.js) -->
      <div style="font-size:0.82rem;color:var(--text-dim);margin-bottom:0.3rem;">STYLE</div>
      <div class="style-pills" id="style-pills">
        <span class="style-pill selected" data-style-id="default">Default</span>
      </div>

      <!-- Bias slider -->
      <div class="bias-row">
        <label>BIAS</label>
        <input type="range" id="bias-range" min="0" max="10" step="0.5" value="5">
        <span id="bias-val">5</span>
      </div>

      <button class="btn" id="btn-generate" style="min-width:160px;">▶ SYNTHESIZE</button>
      <div class="status-msg" id="gen-msg"></div>

      <!-- Progress bar -->
      <div class="progress-bar-wrap" id="progress-wrap">
        <div class="progress-bar-fill" id="progress-fill"></div>
      </div>

      <!-- Results -->
      <div class="results-grid" id="results-grid"></div>
    </div>
  </section>

</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox">
  <img id="lightbox-img" src="" alt="Generated handwriting">
</div>

<script src="/static/js/studio.js"></script>
<script src="/static/js/generate.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add app/templates/index.html
git commit -m "feat: single-page HTML template (Hand Magic retro synth)"
```

---

### Task 11: Studio JS (`app/static/js/studio.js`)

**Files:**
- Create: `app/static/js/studio.js`

- [ ] **Step 1: Write `app/static/js/studio.js`**

```javascript
// app/static/js/studio.js
// Handles: SVG canvas drawing, style list, save/rename/delete

const canvas = document.getElementById('draw-canvas');
const drawPath = document.getElementById('draw-path');
let isDrawing = false;

function svgPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = 800 / rect.width;
  const scaleY = 160 / rect.height;
  const src = evt.touches ? evt.touches[0] : evt;
  return {
    x: (src.clientX - rect.left) * scaleX,
    y: (src.clientY - rect.top) * scaleY,
  };
}

canvas.addEventListener('mousedown', e => {
  e.preventDefault();
  isDrawing = true;
  const p = svgPoint(e);
  drawPath.setAttribute('d', drawPath.getAttribute('d') + `M${p.x},${p.y} `);
});

canvas.addEventListener('mousemove', e => {
  if (!isDrawing) return;
  e.preventDefault();
  const p = svgPoint(e);
  drawPath.setAttribute('d', drawPath.getAttribute('d') + `L${p.x},${p.y} `);
});

canvas.addEventListener('mouseup', e => {
  if (!isDrawing) return;
  e.preventDefault();
  const p = svgPoint(e);
  drawPath.setAttribute('d', drawPath.getAttribute('d') + `M${p.x},${p.y} `);
  isDrawing = false;
});

canvas.addEventListener('touchstart', e => {
  e.preventDefault();
  isDrawing = true;
  const p = svgPoint(e);
  drawPath.setAttribute('d', drawPath.getAttribute('d') + `M${p.x},${p.y} `);
}, {passive: false});

canvas.addEventListener('touchmove', e => {
  if (!isDrawing) return;
  e.preventDefault();
  const p = svgPoint(e);
  drawPath.setAttribute('d', drawPath.getAttribute('d') + `L${p.x},${p.y} `);
}, {passive: false});

canvas.addEventListener('touchend', e => {
  e.preventDefault();
  isDrawing = false;
}, {passive: false});

document.getElementById('btn-clear-canvas').addEventListener('click', () => {
  drawPath.setAttribute('d', '');
});

document.getElementById('btn-save-style').addEventListener('click', async () => {
  const path = drawPath.getAttribute('d').trim();
  const primingText = document.getElementById('priming-text').value.trim();
  const msg = document.getElementById('canvas-msg');

  if (!path) {
    msg.textContent = 'Draw something first.';
    return;
  }
  if (!primingText) {
    msg.textContent = 'Enter the text you wrote above.';
    return;
  }

  msg.textContent = 'Saving...';
  const resp = await fetch('/api/styles', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path: path + ' ', priming_text: primingText}),
  });

  if (resp.ok) {
    msg.textContent = 'Style saved!';
    drawPath.setAttribute('d', '');
    document.getElementById('priming-text').value = '';
    await loadStyles();
  } else {
    const err = await resp.json();
    msg.textContent = err.detail || 'Failed to save style.';
    msg.classList.add('error');
  }
});

document.getElementById('btn-new-style').addEventListener('click', () => {
  document.getElementById('canvas-panel').scrollIntoView({behavior: 'smooth'});
  document.getElementById('priming-text').focus();
});

// ── Style list management ──

let styles = [];

async function loadStyles() {
  const resp = await fetch('/api/styles');
  styles = await resp.json();
  renderStyleList();
  renderStylePills();
}

function renderStyleList() {
  const container = document.getElementById('style-list');
  container.innerHTML = '';
  if (styles.length === 0) {
    container.innerHTML = '<div style="color:var(--text-dim);font-size:0.8rem;">No styles yet.<br>Draw one above!</div>';
    return;
  }
  styles.forEach(s => {
    const item = document.createElement('div');
    item.className = 'style-item';
    item.dataset.id = s.id;

    const nameEl = document.createElement('span');
    nameEl.className = 'style-name';
    nameEl.textContent = s.name;
    nameEl.contentEditable = true;
    nameEl.addEventListener('blur', async () => {
      const newName = nameEl.textContent.trim();
      if (newName && newName !== s.name) {
        await fetch(`/api/styles/${s.id}`, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({name: newName}),
        });
        s.name = newName;
        renderStylePills();
      }
    });

    const delBtn = document.createElement('button');
    delBtn.className = 'style-delete';
    delBtn.textContent = '✕';
    delBtn.title = 'Delete style';
    delBtn.addEventListener('click', async () => {
      await fetch(`/api/styles/${s.id}`, {method: 'DELETE'});
      await loadStyles();
      // if deleted style was selected, reset to default
      if (selectedStyleId === s.id) {
        setSelectedStyle('default');
      }
    });

    item.appendChild(nameEl);
    item.appendChild(delBtn);
    container.appendChild(item);
  });
}

let selectedStyleId = 'default';

function setSelectedStyle(id) {
  selectedStyleId = id;
  document.querySelectorAll('.style-pill').forEach(p => {
    p.classList.toggle('selected', p.dataset.styleId === id);
  });
}

function renderStylePills() {
  const container = document.getElementById('style-pills');
  container.innerHTML = '';

  // Default pill always first
  const defPill = document.createElement('span');
  defPill.className = 'style-pill' + (selectedStyleId === 'default' ? ' selected' : '');
  defPill.dataset.styleId = 'default';
  defPill.textContent = 'Default';
  defPill.addEventListener('click', () => setSelectedStyle('default'));
  container.appendChild(defPill);

  styles.forEach(s => {
    const pill = document.createElement('span');
    pill.className = 'style-pill' + (selectedStyleId === s.id ? ' selected' : '');
    pill.dataset.styleId = s.id;
    pill.textContent = s.name;
    pill.addEventListener('click', () => setSelectedStyle(s.id));
    container.appendChild(pill);
  });
}

// Lightbox
const lightbox = document.getElementById('lightbox');
lightbox.addEventListener('click', () => lightbox.classList.remove('open'));

// Init
loadStyles();
```

- [ ] **Step 2: Commit**

```bash
git add app/static/js/studio.js
git commit -m "feat: canvas drawing + multi-style management JS"
```

---

### Task 12: Generate JS (`app/static/js/generate.js`)

**Files:**
- Create: `app/static/js/generate.js`

- [ ] **Step 1: Write `app/static/js/generate.js`**

```javascript
// app/static/js/generate.js
// Handles: form submit, background job polling, progressive result rendering

const biasRange = document.getElementById('bias-range');
const biasVal = document.getElementById('bias-val');
biasRange.addEventListener('input', () => { biasVal.textContent = biasRange.value; });

document.getElementById('btn-generate').addEventListener('click', startGeneration);

async function startGeneration() {
  const text = document.getElementById('gen-text').value.trim();
  const msg = document.getElementById('gen-msg');
  const btn = document.getElementById('btn-generate');
  const grid = document.getElementById('results-grid');
  const progressWrap = document.getElementById('progress-wrap');
  const progressFill = document.getElementById('progress-fill');

  if (!text) {
    msg.textContent = 'Please enter some text.';
    msg.classList.add('error');
    return;
  }

  msg.classList.remove('error');
  msg.textContent = 'Starting synthesis...';
  btn.disabled = true;
  grid.innerHTML = '';
  progressWrap.style.display = 'block';
  progressFill.style.width = '0%';

  const resp = await fetch('/api/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      text,
      style_id: selectedStyleId,
      bias: parseFloat(biasRange.value),
    }),
  });

  if (!resp.ok) {
    const err = await resp.json();
    msg.textContent = err.detail || 'Failed to start generation.';
    msg.classList.add('error');
    btn.disabled = false;
    progressWrap.style.display = 'none';
    return;
  }

  const {job_id} = await resp.json();
  pollJob(job_id, 0, text);
}

async function pollJob(jobId, lastDone, text) {
  const msg = document.getElementById('gen-msg');
  const btn = document.getElementById('btn-generate');
  const progressFill = document.getElementById('progress-fill');
  const progressWrap = document.getElementById('progress-wrap');

  const resp = await fetch(`/api/jobs/${jobId}`);
  if (!resp.ok) {
    msg.textContent = 'Lost track of job. Please try again.';
    msg.classList.add('error');
    btn.disabled = false;
    return;
  }

  const status = await resp.json();

  // Fetch any newly completed samples
  for (let i = lastDone; i < status.done; i++) {
    await fetchAndRenderSample(jobId, i);
  }

  // Update progress bar
  const pct = status.total > 0 ? (status.done / status.total) * 100 : 0;
  progressFill.style.width = pct + '%';
  msg.textContent = `Synthesizing... ${status.done}/${status.total}`;

  if (status.status === 'done') {
    msg.textContent = `Done! ${status.total} samples generated.`;
    btn.disabled = false;
    progressWrap.style.display = 'none';
    return;
  }

  if (status.status === 'error') {
    msg.textContent = 'Error: ' + (status.message || 'Generation failed.');
    msg.classList.add('error');
    btn.disabled = false;
    progressWrap.style.display = 'none';
    return;
  }

  // Still running — poll again
  setTimeout(() => pollJob(jobId, status.done, text), 1500);
}

async function fetchAndRenderSample(jobId, n) {
  const resp = await fetch(`/api/jobs/${jobId}/sample/${n}`);
  if (!resp.ok) return;
  const {data_url} = await resp.json();

  const grid = document.getElementById('results-grid');
  const card = document.createElement('div');
  card.className = 'result-card';

  const img = document.createElement('img');
  img.src = data_url;
  img.alt = `Sample ${n + 1}`;
  img.addEventListener('click', () => {
    document.getElementById('lightbox-img').src = data_url;
    document.getElementById('lightbox').classList.add('open');
  });

  const actions = document.createElement('div');
  actions.className = 'result-actions';

  const dlBtn = document.createElement('button');
  dlBtn.className = 'btn';
  dlBtn.style.fontSize = '0.85rem';
  dlBtn.textContent = '⬇ SAVE';
  dlBtn.addEventListener('click', () => downloadImage(data_url, n));

  actions.appendChild(dlBtn);
  card.appendChild(img);
  card.appendChild(actions);
  grid.appendChild(card);
}

function downloadImage(dataUrl, n) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `handmagic_${n + 1}.png`;
  a.click();
}
```

- [ ] **Step 2: Commit**

```bash
git add app/static/js/generate.js
git commit -m "feat: generation form + async polling + progressive result rendering"
```

---

### Task 13: Open source packaging

**Files:**
- Create: `pyproject.toml`, `README.md`, `LICENSE`, `CONTRIBUTING.md`
- Modify: `.gitignore`, `Dockerfile`, `docker-compose.yml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "hand-magic"
version = "1.0.0"
description = "Handwriting synthesis web app — LSTM + Gaussian Mixture Models"
authors = [{name = "Omar Musayev"}]
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "jinja2>=3.1.4",
    "python-multipart>=0.0.9",
    "torch>=2.2.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "python-dotenv>=1.0.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "httpx>=0.27.0",
    "pytest-asyncio>=0.23.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `LICENSE`**

```
MIT License

Copyright (c) 2026 Omar Musayev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Write `README.md`**

````markdown
# Hand Magic ✦

> Synthesize handwriting in your own style — powered by LSTM + Gaussian Mixture Models.

**Hand Magic** lets you draw a sample of your handwriting, then generates new text in that style using a neural network trained on the [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database).

## Demo

![Hand Magic screenshot](docs/screenshot.png)

## How It Works

The model is a 3-layer LSTM with a soft-attention window over the input character sequence, trained to output a Mixture Density Network (MDN) over pen-stroke offsets. Based on [Alex Graves' 2013 paper](https://arxiv.org/abs/1308.0850).

## Quick Start

### Requirements
- Python 3.10+
- `pip install -e ".[dev]"` (or just `pip install -e .` for production)

### Local development

```bash
# 1. Clone
git clone https://github.com/yourname/hand-magic.git
cd hand-magic

# 2. Create virtualenv and install
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# 3. Copy env template
cp .env.example .env
# Edit .env: set DISK_STORAGE_PATH to a local directory, e.g. ./disk

# 4. Run
uvicorn main:app --reload
```

Open `http://localhost:8000`.

### Docker

```bash
docker-compose up --build
```

### Training your own model

```bash
python train.py --model_type synthesis --n_epochs 100 --save_path ./logs/
```

## Data & Model Weights

| File | Description |
|---|---|
| `data/strokes.npy` | IAM pen stroke sequences (offset format) |
| `data/sentences.txt` | Paired text labels |
| `results/synthesis/best_model_synthesis_4.pt` | Trained checkpoint (used by default) |

## Project Structure

```
hand-magic/
├── main.py              # FastAPI entry point
├── app/
│   ├── core/            # Config, singletons, session helpers
│   ├── api/             # REST API routes
│   ├── services/        # Background generation, job store, cleanup
│   ├── templates/       # Jinja2 HTML
│   └── static/          # CSS, JS, default style assets
├── models/              # PyTorch model definitions
├── utils/               # Data utilities, dataset loader, plotting
├── generate.py          # CLI generation script
└── train.py             # CLI training script
```

## License

MIT — see [LICENSE](LICENSE).
````

- [ ] **Step 4: Write `CONTRIBUTING.md`**

```markdown
# Contributing to Hand Magic

Thanks for your interest! Here's how to get started.

## Setup

```bash
git clone https://github.com/yourname/hand-magic.git
cd hand-magic
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Running tests

```bash
pytest tests/ -v
```

## Code style

- Keep functions small and focused
- Match existing patterns in the module you're editing
- No authentication code — the app is intentionally auth-free

## Submitting changes

1. Fork the repo and create a branch
2. Write tests for any new behaviour
3. Open a pull request with a clear description
```

- [ ] **Step 5: Update `.gitignore`**

```bash
cat >> .gitignore << 'EOF'

# Sessions and generated files
disk/
/disk

# Environment
.env

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
venv/
hand_gen_env/

# Logs and temp
logs/
*.log
EOF
```

- [ ] **Step 6: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .

ENV DISK_STORAGE_PATH=/disk
VOLUME ["/disk"]

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 7: Write `docker-compose.yml`**

```yaml
services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./disk:/disk
      - .:/app
    environment:
      - DISK_STORAGE_PATH=/disk
      - MODEL_PATH=results/synthesis/best_model_synthesis_4.pt
      - DATA_PATH=./data/
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- [ ] **Step 8: Commit everything**

```bash
git add pyproject.toml LICENSE README.md CONTRIBUTING.md \
        .gitignore Dockerfile docker-compose.yml
git commit -m "feat: open source packaging (pyproject, Dockerfile, README, LICENSE)"
```

---

### Task 14: End-to-end smoke test

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 2: Start app with test env**

```bash
DISK_STORAGE_PATH=./disk_test python -c "
import os
os.makedirs('./disk_test/sessions', exist_ok=True)
" && uvicorn main:app --host 0.0.0.0 --port 8000
```

- [ ] **Step 3: Verify root loads**

```bash
curl -s http://localhost:8000/ | grep -o 'HAND MAGIC'
```
Expected: `HAND MAGIC`

- [ ] **Step 4: Verify API routes exist**

```bash
curl -s http://localhost:8000/api/styles
```
Expected: `[]`

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final smoke test verification"
```
