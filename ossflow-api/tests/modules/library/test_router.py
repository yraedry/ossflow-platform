"""Tests del router del módulo library (T23.3).

Cubren los 7 endpoints migrados de ``api/app.py``:
``POST /api/scan``, ``GET /api/library``, ``GET /api/library/{name}``,
``POST /api/library/{name}/refresh``, ``GET/POST /api/library/{name}/poster``,
``POST /api/library/{name}/poster/redownload``.

Usan ``app.dependency_overrides[get_library_service]`` con un servicio
construido con ``LibraryCache`` real (tmp_path) para no acoplar tests al
filesystem CONFIG_DIR.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.library import library_router
from ossflow_api.modules.library.cache import LibraryCache
from ossflow_api.modules.library.dependencies import get_library_service
from ossflow_api.modules.library.service import LibraryService


@pytest.fixture
def env(tmp_path):
    """Servicio con cache real y library_path apuntando a tmp_path/library."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    cache = LibraryCache(tmp_path / "cache" / "library.json")

    poster_calls: list[tuple] = []

    async def _fake_downloader(folder: Path, url: Optional[str], *, force: bool = False):
        poster_calls.append((folder, url, force))
        if not url:
            return None
        # Simula que el downloader escribe poster.jpg.
        dest = folder / "poster.jpg"
        dest.write_bytes(b"\x89PNG\r\n\x1a\n")
        return "poster.jpg"

    svc = LibraryService(
        cache=cache,
        library_path_loader=lambda: str(library_dir),
        poster_downloader=_fake_downloader,
    )

    app = FastAPI()
    app.include_router(library_router)
    app.dependency_overrides[get_library_service] = lambda: svc

    return {
        "client": TestClient(app),
        "library": library_dir,
        "cache": cache,
        "service": svc,
        "poster_calls": poster_calls,
    }


# ---------------------------------------------------------------------------
# POST /api/scan
# ---------------------------------------------------------------------------


def test_scan_walks_library_and_persists_cache(env):
    course = env["library"] / "Course A"
    course.mkdir()
    (course / "Season 01").mkdir()
    (course / "Season 01" / "S01E01.mp4").write_bytes(b"\0" * 16)

    resp = env["client"].post("/api/scan", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert "instructionals" in body
    names = [it["name"] for it in body["instructionals"]]
    assert "Course A" in names

    # Persistido en cache.
    persisted = env["cache"].load()
    assert persisted is not None
    assert any(it["name"] == "Course A" for it in persisted["instructionals"])


def test_scan_returns_422_when_no_path_configured(tmp_path):
    cache = LibraryCache(tmp_path / "library.json")
    svc = LibraryService(
        cache=cache,
        library_path_loader=lambda: None,
    )
    app = FastAPI()
    app.include_router(library_router)
    app.dependency_overrides[get_library_service] = lambda: svc
    client = TestClient(app)

    resp = client.post("/api/scan", json={})
    assert resp.status_code == 422
    assert "Library path not configured" in resp.json()["error"]


def test_scan_returns_422_when_path_missing(env):
    resp = env["client"].post("/api/scan", json={"path": "/nonexistent/path"})
    assert resp.status_code == 422
    assert "Path not accessible" in resp.json()["error"]


def test_scan_accepts_explicit_path(env, tmp_path):
    other = tmp_path / "other_library"
    other.mkdir()
    (other / "Course X").mkdir()
    (other / "Course X" / "ep1.mp4").write_bytes(b"\0" * 8)

    resp = env["client"].post("/api/scan", json={"path": str(other)})
    assert resp.status_code == 200
    names = [it["name"] for it in resp.json()["instructionals"]]
    assert "Course X" in names


# ---------------------------------------------------------------------------
# GET /api/library
# ---------------------------------------------------------------------------


def test_library_returns_empty_when_cold_cache(env):
    resp = env["client"].get("/api/library")
    assert resp.status_code == 200
    body = resp.json()
    assert body["instructionals"] == []
    assert body["refreshing"] is True


def test_library_returns_cached_data(env):
    env["cache"].save([{"name": "Cached Course", "videos": []}])
    resp = env["client"].get("/api/library")
    assert resp.status_code == 200
    body = resp.json()
    assert any(it["name"] == "Cached Course" for it in body["instructionals"])
    assert "refreshing" in body


# ---------------------------------------------------------------------------
# GET /api/library/{name}
# ---------------------------------------------------------------------------


def test_library_detail_returns_videos_grouped_by_season(env):
    course = env["library"] / "Course Detail"
    course.mkdir()
    env["cache"].save([{
        "name": "Course Detail",
        "path": str(course),
        "has_poster": False,
        "poster_filename": None,
        "poster_mtime": None,
        "videos": [
            {
                "filename": "S01E01.mp4",
                "path": str(course / "Season 01" / "S01E01.mp4"),
                "duration": 100.0,
                "has_subtitles_en": True,
                "has_subtitles_es": False,
                "has_dubbing": False,
                "is_chapter": True,
            },
        ],
    }])

    resp = env["client"].get("/api/library/Course Detail?refresh=false")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Course Detail"
    assert len(body["videos"]) == 1
    assert body["videos"][0]["season"] == "Season 1"
    assert body["videos"][0]["has_subtitles_en"] is True


def test_library_detail_404_when_missing(env):
    env["cache"].save([{"name": "Other", "videos": []}])
    resp = env["client"].get("/api/library/Missing?refresh=false")
    assert resp.status_code == 404


def test_library_detail_404_when_cold_cache(env):
    resp = env["client"].get("/api/library/Anything?refresh=false")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/library/{name}/refresh
# ---------------------------------------------------------------------------


def test_library_refresh_works(env):
    course = env["library"] / "Course Refresh"
    course.mkdir()
    (course / "Season 01").mkdir()
    (course / "Season 01" / "S01E01.mp4").write_bytes(b"\0" * 8)

    env["cache"].save([{
        "name": "Course Refresh",
        "path": str(course),
        "videos": [],
    }])

    resp = env["client"].post("/api/library/Course Refresh/refresh")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_library_refresh_404_when_missing(env):
    env["cache"].save([{"name": "Other"}])
    resp = env["client"].post("/api/library/Missing/refresh")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/library/{name}/poster
# ---------------------------------------------------------------------------


def test_poster_serves_file_with_etag(env):
    course = env["library"] / "Poster Course"
    course.mkdir()
    poster = course / "poster.jpg"
    poster.write_bytes(b"\x89PNG\r\n\x1a\n")

    env["cache"].save([{
        "name": "Poster Course",
        "path": str(course),
        "poster_filename": "poster.jpg",
    }])

    resp = env["client"].get("/api/library/Poster Course/poster")
    assert resp.status_code == 200
    assert "ETag" in resp.headers
    assert resp.headers["Content-Type"] == "image/jpeg"


def test_poster_returns_304_when_etag_matches(env):
    course = env["library"] / "Poster ETag"
    course.mkdir()
    poster = course / "poster.png"
    poster.write_bytes(b"\x89PNG\r\n\x1a\n")
    env["cache"].save([{"name": "Poster ETag", "path": str(course)}])

    resp1 = env["client"].get("/api/library/Poster ETag/poster")
    assert resp1.status_code == 200
    etag = resp1.headers["ETag"]

    resp2 = env["client"].get(
        "/api/library/Poster ETag/poster",
        headers={"If-None-Match": etag},
    )
    assert resp2.status_code == 304


def test_poster_404_when_not_found(env):
    course = env["library"] / "No Poster"
    course.mkdir()
    env["cache"].save([{"name": "No Poster", "path": str(course)}])

    resp = env["client"].get("/api/library/No Poster/poster")
    assert resp.status_code == 404


def test_poster_403_on_traversal(env):
    resp = env["client"].get("/api/library/..%2Fetc/poster")
    # Puede caer en 403 o 404 según resolución; ambos son aceptables como
    # "anti-traversal denied".
    assert resp.status_code in (403, 404)


# ---------------------------------------------------------------------------
# POST /api/library/{name}/poster (upload)
# ---------------------------------------------------------------------------


def test_poster_upload_saves_file(env):
    course = env["library"] / "Upload Course"
    course.mkdir()
    env["cache"].save([{"name": "Upload Course", "path": str(course)}])

    resp = env["client"].post(
        "/api/library/Upload Course/poster",
        files={"file": ("custom.jpg", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["saved"] == "poster.jpg"
    assert (course / "poster.jpg").exists()


def test_poster_upload_rejects_bad_extension(env):
    course = env["library"] / "Upload Bad Ext"
    course.mkdir()
    env["cache"].save([{"name": "Upload Bad Ext", "path": str(course)}])

    resp = env["client"].post(
        "/api/library/Upload Bad Ext/poster",
        files={"file": ("custom.gif", io.BytesIO(b"GIF89a"), "image/gif")},
    )
    assert resp.status_code == 415


def test_poster_upload_404_when_instructional_missing(env):
    resp = env["client"].post(
        "/api/library/Ghost/poster",
        files={"file": ("custom.jpg", io.BytesIO(b"\x89PNG"), "image/jpeg")},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/library/{name}/poster/redownload
# ---------------------------------------------------------------------------


def test_poster_redownload_reads_sidecar_and_calls_downloader(env):
    course = env["library"] / "Redownload Course"
    course.mkdir()
    sidecar = course / ".bjj-meta.json"
    sidecar.write_text(
        '{"oracle": {"poster_url": "https://example.com/p.jpg"}}',
        encoding="utf-8",
    )
    env["cache"].save([{"name": "Redownload Course", "path": str(course)}])

    resp = env["client"].post("/api/library/Redownload Course/poster/redownload")
    assert resp.status_code == 200
    assert resp.json()["saved"] == "poster.jpg"
    assert env["poster_calls"]
    assert env["poster_calls"][0][2] is True  # force=True


def test_poster_redownload_404_when_no_sidecar(env):
    course = env["library"] / "No Sidecar"
    course.mkdir()
    env["cache"].save([{"name": "No Sidecar", "path": str(course)}])

    resp = env["client"].post("/api/library/No Sidecar/poster/redownload")
    assert resp.status_code == 404


def test_poster_redownload_404_when_no_poster_url(env):
    course = env["library"] / "No URL"
    course.mkdir()
    (course / ".bjj-meta.json").write_text(
        '{"oracle": {}}', encoding="utf-8",
    )
    env["cache"].save([{"name": "No URL", "path": str(course)}])

    resp = env["client"].post("/api/library/No URL/poster/redownload")
    assert resp.status_code == 404
