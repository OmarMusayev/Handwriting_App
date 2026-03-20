# main.py
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.singletons import startup_singletons
from app.core.transformer_singleton import TransformerSingleton
from app.services.cleanup import cleanup_old_sessions
from app.api import styles, generate, jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    startup_singletons(settings.data_path, settings.model_path, device)
    TransformerSingleton.initialize(settings.transformer_checkpoint, device)
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
