# tests/test_styles_api.py
from unittest.mock import patch
from tests._helpers import make_client


def test_list_styles_empty(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.get("/api/styles")
    assert resp.status_code == 200
    assert resp.json() == []


def test_save_style_returns_id(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        resp = client.post("/api/styles", json={"path": "M100,100 L110,105 M120,100 ", "priming_text": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["name"] == "Style 1"


def test_list_styles_after_save(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        client.post("/api/styles", json={"path": "M100,100 L110,105", "priming_text": "hi"})
        resp = client.get("/api/styles")
    assert len(resp.json()) == 1


def test_rename_style(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        save_resp = client.post("/api/styles", json={"path": "M100,100 L110,105", "priming_text": "hi"})
        style_id = save_resp.json()["id"]
        resp = client.patch(f"/api/styles/{style_id}", json={"name": "My Style"})
        assert resp.status_code == 200
        assert client.get("/api/styles").json()[0]["name"] == "My Style"


def test_delete_style(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    with client:
        save_resp = client.post("/api/styles", json={"path": "M100,100 L110,105", "priming_text": "hi"})
        style_id = save_resp.json()["id"]
        client.delete(f"/api/styles/{style_id}")
        assert client.get("/api/styles").json() == []
