"""Tests del endpoint /api/burn-subs (absorbido en modules/dubbing)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.dubbing import burn_subs_router
from ossflow_api.modules.dubbing.burn_subs_router import get_burn_subs_service
from ossflow_api.modules.dubbing.burn_subs_service import BurnSubsService
from ossflow_api.modules.jobs._internal.scheduler import JobsScheduler
from ossflow_api.modules.jobs.repositories.background import BackgroundJobsRepository
from ossflow_api.modules.jobs.services.background import BackgroundJobsService


@pytest.fixture
def client(tmp_path, monkeypatch):
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    from ossflow_service_kit.db import engine as eng_mod, session as sess_mod
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    bg_repo = BackgroundJobsRepository()
    bg_svc = BackgroundJobsService(bg_repo, JobsScheduler())
    bg_svc.init()

    burn_svc = BurnSubsService(
        jobs=bg_svc,
        library_path_loader=lambda: str(library_dir),
    )

    app = FastAPI()
    app.include_router(burn_subs_router)
    app.dependency_overrides[get_burn_subs_service] = lambda: burn_svc

    tc = TestClient(app)
    tc.library_dir = library_dir  # type: ignore[attr-defined]
    yield tc

    eng_mod.reset_engine()
    sess_mod.reset_factory()


def _touch(p):
    p.write_text("x", encoding="utf-8")


def test_missing_path_returns_422(client):
    r = client.post("/api/burn-subs", json={})
    # 503 si ffmpeg no está en CI; 422 si el JSON falta path.
    assert r.status_code in (422, 503)


def test_path_outside_library_forbidden(client, tmp_path):
    outside = tmp_path / "outside.mp4"
    _touch(outside)
    r = client.post("/api/burn-subs", json={"path": str(outside)})
    assert r.status_code in (403, 503)


def test_no_matching_srt_returns_409(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    season = lib / "Season 01"
    season.mkdir()
    _touch(season / "video.mp4")

    r = client.post("/api/burn-subs", json={"path": str(season)})
    assert r.status_code in (409, 503)


def test_accepts_folder_with_matching_srt(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    season = lib / "Season 01"
    season.mkdir()
    video = season / "S01E01 - Intro.mp4"
    srt = season / "S01E01 - Intro.ES.srt"
    _touch(video)
    _touch(srt)

    r = client.post("/api/burn-subs", json={"path": str(season)})
    # 200 si ffmpeg existe; 503 si no.
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert body["type"] == "burn_subs"
        assert body["status"] in ("queued", "running", "completed", "failed")
