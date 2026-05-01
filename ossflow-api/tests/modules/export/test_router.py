"""Tests del módulo export.

Usa ``app.dependency_overrides`` con factories fake para no tocar la red
ni el filesystem real (``OssFlowClient`` ni ``PlexExporter``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.export import export_router
from ossflow_api.modules.export.dependencies import get_export_service
from ossflow_api.modules.export.service import ExportService


class _FakeOssFlowClient:
    def __init__(self, *, base_url: str | None = None, fail: bool = False) -> None:
        self.base_url = base_url
        self.fail = fail

    def export_full_instructional(self, path: Path, instructor: str) -> dict:
        if self.fail:
            raise RuntimeError("simulated upstream error")
        return {"path": str(path), "instructor": instructor, "files": 3}

    def health_check(self) -> bool:
        return not self.fail


class _FakePlexExporter:
    def __init__(self, *, missing_source: bool = False) -> None:
        self.missing_source = missing_source
        self.calls: list[dict] = []

    def export(
        self,
        name: str,
        chapters: list,
        source: Path,
        output: Path,
    ) -> None:
        if self.missing_source:
            raise FileNotFoundError(str(source))
        self.calls.append(
            {"name": name, "chapters": chapters, "source": source, "output": output}
        )


@pytest.fixture
def env(tmp_path):
    ossflow = _FakeOssFlowClient()
    plex = _FakePlexExporter()

    svc = ExportService(
        ossflow_factory=lambda *, base_url=None: ossflow,
        plex_factory=lambda: plex,
    )

    app = FastAPI()
    app.include_router(export_router)
    app.dependency_overrides[get_export_service] = lambda: svc

    src = tmp_path / "src"
    src.mkdir()

    return {
        "client": TestClient(app),
        "ossflow": ossflow,
        "plex": plex,
        "source_dir": str(src),
        "output_dir": str(tmp_path / "out"),
    }


# ---------------------------------------------------------------------------
# /api/export/ossflow
# ---------------------------------------------------------------------------


def test_ossflow_export_happy_path(env):
    r = env["client"].post(
        "/api/export/ossflow",
        json={
            "path": env["source_dir"],
            "instructor": "Danaher",
            "base_url": "http://ossflow.test",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["summary"]["instructor"] == "Danaher"


def test_ossflow_export_missing_fields_returns_400(env):
    r = env["client"].post("/api/export/ossflow", json={"instructor": ""})
    assert r.status_code == 400
    assert "error" in r.json()


def test_ossflow_export_path_not_exists_returns_404(env):
    r = env["client"].post(
        "/api/export/ossflow",
        json={"path": "/no/such/dir", "instructor": "X"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/export/ossflow/status
# ---------------------------------------------------------------------------


def test_ossflow_status_reachable(env):
    r = env["client"].get("/api/export/ossflow/status")
    assert r.status_code == 200
    assert r.json() == {"reachable": True}


def test_ossflow_status_unreachable_returns_payload_with_error(env):
    env["ossflow"].fail = True
    r = env["client"].get("/api/export/ossflow/status")
    assert r.status_code == 200
    body = r.json()
    assert body["reachable"] is False


# ---------------------------------------------------------------------------
# /api/export/plex
# ---------------------------------------------------------------------------


def test_plex_export_happy_path(env):
    r = env["client"].post(
        "/api/export/plex",
        json={
            "name": "MyInstructional",
            "chapters": [{"title": "Cap 1"}],
            "source_dir": env["source_dir"],
            "output_dir": env["output_dir"],
        },
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert len(env["plex"].calls) == 1


def test_plex_export_missing_fields_returns_400(env):
    r = env["client"].post("/api/export/plex", json={"name": ""})
    assert r.status_code == 400
    assert "error" in r.json()


def test_plex_export_source_dir_missing_returns_404(env):
    r = env["client"].post(
        "/api/export/plex",
        json={
            "name": "X",
            "chapters": [{"title": "c"}],
            "source_dir": "/no/such/dir",
            "output_dir": env["output_dir"],
        },
    )
    assert r.status_code == 404
