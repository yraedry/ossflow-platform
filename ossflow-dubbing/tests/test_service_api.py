"""Service-level tests for dubbing-generator FastAPI wrapper (real pipeline mocked)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PY_ROOT = Path(__file__).resolve().parents[2]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from bjj_service_kit import JobEvent, RunRequest, create_app  # noqa: E402


SERVICE_NAME = "dubbing-generator"


def _fake_task(req: RunRequest, emit) -> None:
    emit(JobEvent(type="log", data={"message": "fake dub"}))
    emit(JobEvent(type="progress", data={"pct": 100}))


@pytest.fixture()
def client():
    app = create_app(service_name=SERVICE_NAME, task_fn=_fake_task)
    return TestClient(app)


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": SERVICE_NAME}


def test_run_returns_job_id(client, tmp_path):
    r = client.post("/run", json={"input_path": str(tmp_path)})
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_events_streams_done(client, tmp_path):
    job_id = client.post("/run", json={"input_path": str(tmp_path)}).json()["job_id"]
    with client.stream("GET", f"/events/{job_id}") as resp:
        text = "".join(resp.iter_text())
    assert "event: log" in text
    assert "event: done" in text


def test_events_streams_error_on_invalid_path(client):
    r = client.post("/run", json={"input_path": "not-absolute"})
    assert r.status_code == 400


def test_events_streams_error_on_task_failure(tmp_path):
    def boom(req, emit):
        raise RuntimeError("xtts missing")

    app = create_app(service_name=SERVICE_NAME, task_fn=boom)
    c = TestClient(app)
    job_id = c.post("/run", json={"input_path": str(tmp_path)}).json()["job_id"]
    with c.stream("GET", f"/events/{job_id}") as resp:
        text = "".join(resp.iter_text())
    assert "event: error" in text
    assert "xtts" in text


def test_run_rejects_path_traversal(client):
    r = client.post("/run", json={"input_path": "C:/proyectos/../secret"})
    assert r.status_code == 400
