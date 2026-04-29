"""Tests for logs_view endpoint.

Nuevo contrato (no usa docker CLI — no está disponible dentro del contenedor
processor-api):
- ``processor-api``: lee del ring buffer en memoria del propio proceso.
- resto de servicios: pide al backend via httpx GET {base}/logs.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import logs_view


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(logs_view.router)
    return TestClient(app)


def _seed_local(logger_name: str = "tests.logs"):
    logs_view._LOCAL_BUFFER.buffer.clear()
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    log.info("Starting service")
    log.debug("Loaded config")
    log.warning("Slow DB query")
    log.error("Connection refused")


def test_local_ring_buffer_returns_lines(client):
    _seed_local()
    r = client.get("/api/logs/", params={"service": "processor-api"})
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "processor-api"
    levels = [l["level"] for l in data["lines"]]
    assert "INFO" in levels and "ERROR" in levels and "WARNING" in levels


def test_filter_by_level_error(client):
    _seed_local()
    r = client.get("/api/logs/", params={"service": "processor-api", "level": "ERROR"})
    assert r.status_code == 200
    lines = r.json()["lines"]
    assert len(lines) == 1
    assert lines[0]["level"] == "ERROR"
    assert "Connection refused" in lines[0]["message"]


def test_level_warn_is_normalized_to_warning(client):
    _seed_local()
    r = client.get("/api/logs/", params={"service": "processor-api", "level": "WARN"})
    assert r.status_code == 200
    lines = r.json()["lines"]
    assert len(lines) == 1
    assert lines[0]["level"] == "WARNING"


def test_invalid_service_returns_400(client):
    r = client.get("/api/logs/", params={"service": "hacker-container"})
    assert r.status_code == 400
    assert "Unknown service" in r.json()["detail"]


def test_invalid_level_returns_400(client):
    r = client.get("/api/logs/", params={"service": "processor-api", "level": "BOGUS"})
    assert r.status_code == 400


def test_remote_service_uses_httpx(client, monkeypatch):
    """Para servicios no locales, se consulta el backend via HTTP."""
    captured: dict = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return {
                "service": "chapter-splitter",
                "lines": [{"timestamp": 1.0, "level": "INFO", "message": "hola"}],
                "truncated": False,
            }

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResp()

    monkeypatch.setattr(logs_view.httpx, "get", fake_get)
    r = client.get("/api/logs/", params={"service": "chapter-splitter", "tail": 10})
    assert r.status_code == 200
    assert captured["url"].endswith("/logs")
    assert captured["params"]["tail"] == 10
    assert r.json()["lines"][0]["message"] == "hola"


def test_remote_backend_unreachable_returns_502(client, monkeypatch):
    import httpx as _httpx
    def boom(*a, **kw):
        raise _httpx.ConnectError("nope")
    monkeypatch.setattr(logs_view.httpx, "get", boom)
    r = client.get("/api/logs/", params={"service": "chapter-splitter"})
    assert r.status_code == 502


def test_all_level_returns_everything(client):
    _seed_local()
    r = client.get("/api/logs/", params={"service": "processor-api", "level": "ALL"})
    assert r.status_code == 200
    # ALL es sinónimo de "sin filtro"; incluye INFO, DEBUG, WARNING, ERROR.
    levels = {l["level"] for l in r.json()["lines"]}
    assert {"INFO", "DEBUG", "WARNING", "ERROR"}.issubset(levels)
