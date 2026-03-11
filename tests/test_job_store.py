"""Tests for app/services/job_store.py"""
import importlib
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def reset_jobs():
    """Clear in-memory job dict between tests."""
    import app.services.job_store as js
    js._jobs.clear()
    yield
    js._jobs.clear()


def test_create_job_writes_status_file(tmp_path):
    import app.services.job_store as js
    job_dir = tmp_path / "j1"
    js.create_job("j1", job_dir, total=5)

    assert (job_dir / "status.json").exists()
    data = js.get_job("j1", job_dir)
    assert data["status"] == "running"
    assert data["done"] == 0
    assert data["total"] == 5


def test_mark_sample_done_updates_count(tmp_path):
    import app.services.job_store as js
    job_dir = tmp_path / "j2"
    js.create_job("j2", job_dir, total=3)
    js.mark_sample_done("j2", job_dir, done=2)

    data = js.get_job("j2", job_dir)
    assert data["done"] == 2
    assert data["status"] == "running"


def test_complete_job(tmp_path):
    import app.services.job_store as js
    job_dir = tmp_path / "j3"
    js.create_job("j3", job_dir, total=1)
    js.mark_sample_done("j3", job_dir, done=1)
    js.complete_job("j3", job_dir)

    data = js.get_job("j3", job_dir)
    assert data["status"] == "done"


def test_fail_job(tmp_path):
    import app.services.job_store as js
    job_dir = tmp_path / "j4"
    job_dir.mkdir()
    js.fail_job("j4", job_dir, error="boom")

    data = js.get_job("j4", job_dir)
    assert data["status"] == "error"
    assert data["message"] == "boom"


def test_get_job_falls_back_to_file_when_not_in_memory(tmp_path):
    """If job not in _jobs dict, should read from disk."""
    import app.services.job_store as js
    job_dir = tmp_path / "j5"
    js.create_job("j5", job_dir, total=2)
    # Evict from memory
    js._jobs.clear()

    data = js.get_job("j5", job_dir)
    assert data is not None
    assert data["total"] == 2


def test_get_job_returns_none_for_missing(tmp_path):
    import app.services.job_store as js
    result = js.get_job("nonexistent", tmp_path / "missing")
    assert result is None
