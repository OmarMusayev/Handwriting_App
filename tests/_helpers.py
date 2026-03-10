# tests/_helpers.py — shared test utilities
import importlib
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch):
    """Build a fresh FastAPI TestClient with singletons mocked out."""
    monkeypatch.setenv("DISK_STORAGE_PATH", str(tmp_path))

    # Reload config first so settings picks up the new env var
    import app.core.config as cfg
    importlib.reload(cfg)

    # Reload all API modules so they re-import the fresh settings
    import app.core.session as sess_mod
    import app.services.cleanup as cleanup_mod
    import app.api.styles as styles_mod
    import app.api.generate as gen_mod
    import app.api.jobs as jobs_mod
    for mod in (sess_mod, cleanup_mod, styles_mod, gen_mod, jobs_mod):
        importlib.reload(mod)

    import main as m
    importlib.reload(m)

    # Patch lifespan hooks that were imported into main's namespace
    m.startup_singletons = MagicMock()
    m.cleanup_old_sessions = MagicMock()

    return TestClient(m.app, raise_server_exceptions=True)
