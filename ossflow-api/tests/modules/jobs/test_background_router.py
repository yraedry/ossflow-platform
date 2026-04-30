"""Tests del router ``/api/background-jobs/*``.

Verifica:
* Listado vacío y poblado.
* GET por id, 404 si no existe.
* **Redirect 307** desde ``/api/background-jobs/`` (con trailing slash) a
  la canónica sin slash. Decisión documentada en spec anexo §0:
  eliminamos el endpoint duplicado del legacy.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.jobs.dependencies import get_background_jobs_service
from ossflow_api.modules.jobs.models import BackgroundJob
from ossflow_api.modules.jobs.routers.background import router as bg_router


class _StubService:
    """Servicio fake con respuestas canned."""

    def __init__(self, jobs: list[BackgroundJob]):
        self._jobs = {j.id: j for j in jobs}

    def list_all(self, type_filter=None):
        items = list(self._jobs.values())
        if type_filter:
            items = [j for j in items if j.type == type_filter]
        return items

    def get(self, job_id):
        return self._jobs.get(job_id)


def _make_client(jobs: list[BackgroundJob]) -> TestClient:
    app = FastAPI()
    app.include_router(bg_router)
    app.dependency_overrides[get_background_jobs_service] = lambda: _StubService(jobs)
    return TestClient(app)


def test_list_empty_returns_empty_jobs():
    client = _make_client([])
    r = client.get("/api/background-jobs")
    assert r.status_code == 200
    assert r.json() == {"jobs": []}


def test_list_returns_all_jobs():
    jobs = [
        BackgroundJob(id="a", type="cleanup_scan", status="completed"),
        BackgroundJob(id="b", type="duplicates_scan", status="running", progress=42.0),
    ]
    client = _make_client(jobs)
    r = client.get("/api/background-jobs")
    assert r.status_code == 200
    body = r.json()
    assert len(body["jobs"]) == 2
    ids = {j["id"] for j in body["jobs"]}
    assert ids == {"a", "b"}


def test_list_filters_by_type():
    jobs = [
        BackgroundJob(id="a", type="cleanup_scan", status="queued"),
        BackgroundJob(id="b", type="duplicates_scan", status="queued"),
    ]
    client = _make_client(jobs)
    r = client.get("/api/background-jobs?type=cleanup_scan")
    assert r.status_code == 200
    body = r.json()
    assert len(body["jobs"]) == 1
    assert body["jobs"][0]["id"] == "a"


def test_get_existing_job():
    jobs = [BackgroundJob(id="abc", type="cleanup_scan", status="running", progress=42.0)]
    client = _make_client(jobs)
    r = client.get("/api/background-jobs/abc")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "abc"
    assert body["type"] == "cleanup_scan"
    assert body["status"] == "running"
    assert body["progress"] == 42.0


def test_get_missing_job_returns_404():
    client = _make_client([])
    r = client.get("/api/background-jobs/ghost")
    assert r.status_code == 404
    assert r.json() == {"detail": "Job not found"}


def test_trailing_slash_redirects_307_to_canonical():
    """Forma con trailing slash: FastAPI redirige automáticamente a la
    forma canónica sin slash con ``redirect_slashes=True`` (default)."""
    client = _make_client([])
    # follow_redirects=False para verificar la redirección, no el destino.
    r = client.get("/api/background-jobs/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].endswith("/api/background-jobs")


def test_response_shape_matches_legacy_contract():
    """El JSON externo debe seguir teniendo las mismas keys que el legacy
    para no romper el frontend."""
    jobs = [BackgroundJob(
        id="z",
        type="cleanup_scan",
        status="completed",
        progress=100.0,
        message="done",
        params={"path": "/x"},
        result={"deleted": 5},
    )]
    client = _make_client(jobs)
    r = client.get("/api/background-jobs/z")
    body = r.json()
    expected_keys = {
        "id", "type", "status", "progress", "message",
        "result", "error", "created_at", "completed_at", "params",
    }
    assert set(body.keys()) == expected_keys
