# app/api/styles.py
import base64
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.session import get_or_create_session
from utils import plot_stroke

router = APIRouter()


def _styles_dir(token: str) -> Path:
    return settings.sessions_path / token / "styles"


def _count_styles(token: str) -> int:
    d = _styles_dir(token)
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_dir())


def _next_name(token: str) -> str:
    return f"Style {_count_styles(token) + 1}"


class SaveStyleRequest(BaseModel):
    stroke_data: list   # [[eos, dx, dy], ...]  already converted by the canvas JS
    priming_text: str = ""


class RenameRequest(BaseModel):
    name: str


@router.get("/styles")
async def list_styles(request: Request, response: Response):
    token = get_or_create_session(request, response)
    d = _styles_dir(token)
    if not d.exists():
        return []
    result = []
    for sd in sorted(d.iterdir(), key=lambda p: p.stat().st_ctime):
        if not sd.is_dir():
            continue
        mf = sd / "meta.json"
        if not mf.exists():
            continue
        meta = json.loads(mf.read_text())
        result.append({
            "id": sd.name,
            "name": meta["name"],
            "created_at": meta["created_at"],
            "has_preview": (sd / "preview.png").exists(),
        })
    return result


@router.post("/styles")
async def save_style(body: SaveStyleRequest, request: Request, response: Response):
    token = get_or_create_session(request, response)
    if _count_styles(token) >= settings.max_styles_per_session:
        raise HTTPException(status_code=400, detail="Maximum styles reached")
    if not body.stroke_data:
        raise HTTPException(status_code=400, detail="Empty stroke data")

    style_id = str(uuid.uuid4())
    # Compute name BEFORE creating the directory so _count_styles is not off by one
    next_name = _next_name(token)
    sd = _styles_dir(token) / style_id
    sd.mkdir(parents=True, exist_ok=True)

    priming_text = body.priming_text or "hello"
    # stroke_data is [[eos, dx, dy], ...] — convert directly to float32 array
    stroke = np.array(body.stroke_data, dtype=np.float32)
    np.save(str(sd / "stroke.npy"), stroke, allow_pickle=True)
    plot_stroke(stroke, str(sd / "preview.png"))

    meta = {
        "name": next_name,
        "priming_text": priming_text,
        "created_at": datetime.utcnow().isoformat(),
    }
    (sd / "meta.json").write_text(json.dumps(meta))
    return {"id": style_id, "name": meta["name"]}


@router.patch("/styles/{style_id}")
async def rename_style(style_id: str, body: RenameRequest, request: Request, response: Response):
    token = get_or_create_session(request, response)
    mf = _styles_dir(token) / style_id / "meta.json"
    if not mf.exists():
        raise HTTPException(status_code=404, detail="Style not found")
    meta = json.loads(mf.read_text())
    meta["name"] = body.name.strip() or meta["name"]
    mf.write_text(json.dumps(meta))
    return {"id": style_id, "name": meta["name"]}


@router.delete("/styles/{style_id}")
async def delete_style(style_id: str, request: Request, response: Response):
    token = get_or_create_session(request, response)
    sd = _styles_dir(token) / style_id
    if not sd.exists():
        raise HTTPException(status_code=404, detail="Style not found")
    shutil.rmtree(sd)
    return {"deleted": style_id}


@router.get("/styles/{style_id}/preview")
async def style_preview(style_id: str, request: Request, response: Response):
    token = get_or_create_session(request, response)
    preview = _styles_dir(token) / style_id / "preview.png"
    if not preview.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    data = base64.b64encode(preview.read_bytes()).decode("ascii")
    return {"data_url": f"data:image/png;base64,{data}"}
