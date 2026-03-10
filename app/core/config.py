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
