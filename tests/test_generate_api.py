# tests/test_generate_api.py
import json
from unittest.mock import patch, MagicMock
from tests._helpers import make_client


def test_empty_text_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        with patch("app.api.generate.threading.Thread"):
            resp = client.post("/api/generate", json={"text": "", "style_id": "default", "bias": 5.0})
    assert resp.status_code == 400


def test_generate_returns_job_id(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        with patch("app.api.generate.threading.Thread", return_value=MagicMock()):
            resp = client.post("/api/generate", json={"text": "hello", "style_id": "default", "bias": 5.0})
    assert resp.status_code == 200
    assert "job_id" in resp.json()


def test_unknown_style_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        with patch("app.api.generate.threading.Thread"):
            resp = client.post("/api/generate", json={"text": "hi", "style_id": "bad-id", "bias": 3.0})
    assert resp.status_code == 404


def test_job_not_found(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404


def test_job_status_readable(tmp_path, monkeypatch):
    import app.core.config as cfg
    client = make_client(tmp_path, monkeypatch)
    with client:
        with patch("app.api.generate.threading.Thread", return_value=MagicMock()):
            gen_resp = client.post("/api/generate", json={"text": "test", "style_id": "default", "bias": 3.0})
        assert gen_resp.status_code == 200
        job_id = gen_resp.json()["job_id"]
        token = client.cookies.get("hm_session")
        if token:
            job_dir = cfg.settings.sessions_path / token / "jobs" / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "status.json").write_text(
                json.dumps({"status": "running", "done": 0, "total": 5})
            )
            resp = client.get(f"/api/jobs/{job_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "running"
