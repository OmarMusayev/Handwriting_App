# app/api/jobs.py
import base64
from pathlib import Path

from fastapi import APIRouter, Request, Response, HTTPException

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
