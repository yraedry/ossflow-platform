"""Tests for /api/burn-subs."""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()

    monkeypatch.setenv("CONFIG_DIR", str(config_dir))

    import api.settings as settings_mod
    importlib.reload(settings_mod)
    settings_mod.save_settings(
        {**settings_mod.load_settings(), "library_path": str(library_dir)}
    )

    import api.background_jobs as bg_mod
    importlib.reload(bg_mod)
    import api.burn_subs as burn_mod
    importlib.reload(burn_mod)

    app = FastAPI()
    app.include_router(burn_mod.router)
    app.include_router(bg_mod.router)

    tc = TestClient(app)
    tc.library_dir = library_dir  # type: ignore[attr-defined]
    return tc


def _touch(p):
    p.write_text("x", encoding="utf-8")


def test_missing_path_returns_422(client):
    r = client.post("/api/burn-subs", json={})
    assert r.status_code in (422, 503)  # 503 if ffmpeg missing on CI


def test_path_outside_library_forbidden(client, tmp_path):
    outside = tmp_path / "outside.mp4"
    _touch(outside)
    r = client.post("/api/burn-subs", json={"path": str(outside)})
    assert r.status_code in (403, 503)


def test_no_matching_srt_returns_409(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    season = lib / "Season 01"
    season.mkdir()
    _touch(season / "video.mp4")  # sin .ES.srt

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
    # 200 si ffmpeg existe y arranca el job; 503 si falta ffmpeg en CI
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        body = r.json()
        assert body["type"] == "burn_subs"
        assert body["status"] in ("queued", "running", "completed", "failed")
