# app/api/generate.py
import json
import uuid
import threading
from pathlib import Path

from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.session import get_or_create_session
from app.services.generation import run_generation_job
from app.services.transformer_generation import run_transformer_generation_job
from app.core.transformer_singleton import TransformerSingleton

router = APIRouter()


class GenerateRequest(BaseModel):
    text: str
    style_id: str = "default"
    bias: float = 5.0
    mode: str = "lstm"  # "lstm" or "transformer"


@router.post("/generate")
async def start_generate(body: GenerateRequest, request: Request, response: Response):
    token = get_or_create_session(request, response)
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    job_id = str(uuid.uuid4())
    job_dir = settings.sessions_path / token / "jobs" / job_id

    if body.mode == "transformer":
        if not TransformerSingleton.is_available():
            raise HTTPException(status_code=503, detail="Transformer model not loaded")

        t = threading.Thread(
            target=run_transformer_generation_job,
            kwargs=dict(
                job_id=job_id,
                job_dir=job_dir,
                text=body.text,
                n_samples=settings.n_samples,
            ),
            daemon=True,
        )
        t.start()
        return {"job_id": job_id}

    # LSTM mode
    if body.style_id == "default":
        style_path = settings.default_style_path
        priming_text = "copy monkey app"
    else:
        style_dir = settings.sessions_path / token / "styles" / body.style_id
        style_path = style_dir / "stroke.npy"
        meta_file = style_dir / "meta.json"
        if not style_path.exists():
            raise HTTPException(status_code=404, detail="Style not found")
        meta = json.loads(meta_file.read_text())
        priming_text = meta.get("priming_text", "hello")

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
