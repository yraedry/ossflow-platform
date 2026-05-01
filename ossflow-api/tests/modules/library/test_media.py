"""Tests del módulo media (T23.5).

Cubren ``/api/video-info``, ``/api/thumbnail`` y ``/api/media``. ffprobe
y ffmpeg se mockean a nivel ``subprocess.run`` — los tests no
dependen de tener ffmpeg instalado.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.library import library_router
from ossflow_api.modules.library import media as media_mod
from ossflow_api.modules.library.cache import LibraryCache
from ossflow_api.modules.library.dependencies import get_library_service
from ossflow_api.modules.library.service import LibraryService


@pytest.fixture
def env(tmp_path, monkeypatch):
    media_root = tmp_path / "media"
    media_root.mkdir()
    library_dir = media_root / "library"
    library_dir.mkdir()
    monkeypatch.setenv("MEDIA_ROOT", str(media_root))

    cache = LibraryCache(tmp_path / "library.json")
    svc = LibraryService(
        cache=cache,
        library_path_loader=lambda: str(library_dir),
    )

    app = FastAPI()
    app.include_router(library_router)
    app.dependency_overrides[get_library_service] = lambda: svc

    return {
        "client": TestClient(app),
        "media": media_root,
        "library": library_dir,
    }


# ---------------------------------------------------------------------------
# /api/video-info
# ---------------------------------------------------------------------------


def test_video_info_404_when_file_missing(env):
    resp = env["client"].get("/api/video-info", params={"path": "/no/such/file.mp4"})
    assert resp.status_code == 404


def test_video_info_returns_metadata(env, monkeypatch):
    video = env["library"] / "ep1.mp4"
    video.write_bytes(b"\x00" * 100)

    class _Result:
        returncode = 0
        stdout = (
            '{"format": {"duration": "65.5", "size": "10485760"},'
            ' "streams": [{"codec_type": "video", "codec_name": "h264",'
            ' "width": 1920, "height": 1080, "r_frame_rate": "30/1"}]}'
        )

    monkeypatch.setattr(media_mod.subprocess, "run", lambda *a, **kw: _Result())

    resp = env["client"].get("/api/video-info", params={"path": str(video)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["duration"] == 65.5
    assert body["duration_formatted"] == "1:05"
    assert body["codec"] == "h264"
    assert body["width"] == 1920
    assert body["fps"] == 30


def test_video_info_returns_defaults_on_ffprobe_failure(env, monkeypatch):
    video = env["library"] / "broken.mp4"
    video.write_bytes(b"\x00")

    class _Fail:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(media_mod.subprocess, "run", lambda *a, **kw: _Fail())

    resp = env["client"].get("/api/video-info", params={"path": str(video)})
    assert resp.status_code == 200
    assert resp.json()["duration"] == 0


# ---------------------------------------------------------------------------
# /api/thumbnail
# ---------------------------------------------------------------------------


def test_thumbnail_returns_image(env, monkeypatch):
    video = env["library"] / "ep1.mp4"
    video.write_bytes(b"\x00" * 100)

    class _OK:
        returncode = 0
        stdout = b"\xff\xd8\xff\xe0fake-jpeg"

    monkeypatch.setattr(media_mod.subprocess, "run", lambda *a, **kw: _OK())

    resp = env["client"].get("/api/thumbnail", params={"path": str(video)})
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/jpeg"
    assert resp.content.startswith(b"\xff\xd8")


def test_thumbnail_404_when_file_missing(env):
    """Path bajo library_path pero el fichero no existe → 404."""
    ghost = env["library"] / "ghost.mp4"
    resp = env["client"].get("/api/thumbnail", params={"path": str(ghost)})
    assert resp.status_code == 404


def test_thumbnail_400_when_path_outside_library(env, tmp_path):
    """Path fuera de library_path y MEDIA_ROOT → 400 (ValueError)."""
    outside = tmp_path / "outside.mp4"
    resp = env["client"].get("/api/thumbnail", params={"path": str(outside)})
    assert resp.status_code == 400


def test_thumbnail_500_when_ffmpeg_fails(env, monkeypatch):
    video = env["library"] / "ep1.mp4"
    video.write_bytes(b"\x00" * 100)

    class _Fail:
        returncode = 1
        stdout = b""

    monkeypatch.setattr(media_mod.subprocess, "run", lambda *a, **kw: _Fail())

    resp = env["client"].get("/api/thumbnail", params={"path": str(video)})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /api/media — full file
# ---------------------------------------------------------------------------


def test_media_serves_full_video_without_range(env):
    video = env["library"] / "movie.mp4"
    payload = b"\x00\x01\x02\x03" * 1024
    video.write_bytes(payload)

    resp = env["client"].get("/api/media", params={"path": str(video)})
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "video/mp4"
    assert resp.content == payload


def test_media_404_when_outside_root(env, tmp_path):
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"x")
    resp = env["client"].get("/api/media", params={"path": str(outside)})
    assert resp.status_code == 404


def test_media_404_when_not_a_file(env):
    resp = env["client"].get("/api/media", params={"path": str(env["library"])})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/media — Range requests
# ---------------------------------------------------------------------------


def test_media_serves_partial_with_range(env):
    video = env["library"] / "rng.mp4"
    payload = bytes(range(256)) * 4  # 1024 bytes ordered
    video.write_bytes(payload)

    resp = env["client"].get(
        "/api/media",
        params={"path": str(video)},
        headers={"Range": "bytes=10-19"},
    )
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == f"bytes 10-19/{len(payload)}"
    assert resp.content == payload[10:20]


def test_media_416_on_invalid_range(env):
    video = env["library"] / "bad.mp4"
    video.write_bytes(b"\x00" * 50)

    resp = env["client"].get(
        "/api/media",
        params={"path": str(video)},
        headers={"Range": "bytes=999-1999"},
    )
    assert resp.status_code == 416


# ---------------------------------------------------------------------------
# /api/media — subtitles
# ---------------------------------------------------------------------------


def test_media_serves_srt_as_subrip(env):
    srt = env["library"] / "subs.srt"
    srt.write_text("1\n00:00:01,500 --> 00:00:03,000\nHello\n", encoding="utf-8")

    resp = env["client"].get("/api/media", params={"path": str(srt)})
    assert resp.status_code == 200
    assert "subrip" in resp.headers["Content-Type"]


def test_media_converts_srt_to_vtt_with_query_param(env):
    srt = env["library"] / "subs.srt"
    srt.write_text("1\n00:00:01,500 --> 00:00:03,000\nHello\n", encoding="utf-8")

    resp = env["client"].get(
        "/api/media", params={"path": str(srt), "as": "vtt"},
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("text/vtt")
    body = resp.text
    assert body.startswith("WEBVTT")
    assert "00:00:01.500" in body  # comma → dot


# ---------------------------------------------------------------------------
# Compat shim para pipeline.py
# ---------------------------------------------------------------------------


def test_compat_alias_exists():
    """``api.app.get_video_info`` debe seguir importable (pipeline.py)."""
    from api.app import get_video_info, generate_thumbnail
    assert callable(get_video_info)
    assert callable(generate_thumbnail)
