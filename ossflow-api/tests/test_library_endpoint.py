"""Tests for /api/library and /api/library/{name}/poster."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate CONFIG_DIR and library path before importing the app
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()

    monkeypatch.setenv("CONFIG_DIR", str(config_dir))

    # Import after env is set
    import importlib

    import api.settings as settings_mod
    importlib.reload(settings_mod)
    settings_mod.save_settings({**settings_mod.load_settings(), "library_path": str(library_dir)})

    import api.app as app_mod
    importlib.reload(app_mod)

    tc = TestClient(app_mod.app)
    tc.library_dir = library_dir  # type: ignore[attr-defined]
    tc.config_dir = config_dir  # type: ignore[attr-defined]
    return tc


def test_library_returns_204_when_no_cache(client):
    r = client.get("/api/library")
    assert r.status_code in (204, 404)


def test_library_returns_cache_when_present(client):
    cache_file = client.config_dir / "library.json"  # type: ignore[attr-defined]
    cache_file.write_text(json.dumps({
        "instructionals": [
            {"name": "Foo", "path": str(client.library_dir / "Foo"), "has_poster": True, "poster_filename": "poster.jpg"},
        ]
    }))
    r = client.get("/api/library")
    assert r.status_code == 200
    assert r.json()["instructionals"][0]["name"] == "Foo"


def test_poster_returns_file_when_present(client):
    folder = client.library_dir / "MyInstr"  # type: ignore[attr-defined]
    folder.mkdir()
    (folder / "poster.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    r = client.get("/api/library/MyInstr/poster")
    assert r.status_code == 200
    assert "image" in r.headers["content-type"]
    assert "max-age" in r.headers.get("cache-control", "")


def test_poster_404_when_missing(client):
    folder = client.library_dir / "NoPoster"  # type: ignore[attr-defined]
    folder.mkdir()
    r = client.get("/api/library/NoPoster/poster")
    assert r.status_code == 404


def test_poster_rejects_path_traversal(client):
    r = client.get("/api/library/..%2F..%2Fetc/poster")
    assert r.status_code in (403, 404)


def test_poster_case_insensitive(client):
    folder = client.library_dir / "CaseTest"  # type: ignore[attr-defined]
    folder.mkdir()
    (folder / "POSTER.JPG").write_bytes(b"\xff\xd8\xff\xd9")
    r = client.get("/api/library/CaseTest/poster")
    assert r.status_code == 200


def test_scan_enrich_detects_cover_webp_case_insensitive(tmp_path):
    """enrich_with_poster should detect Cover.webp (mixed case) as a poster."""
    from api.scan_cache import enrich_with_poster

    folder = tmp_path / "Inst1"
    folder.mkdir()
    (folder / "Cover.webp").write_bytes(b"webp-bytes")

    items = [{"name": "Inst1", "path": str(folder)}]
    enriched = enrich_with_poster(items)
    assert enriched[0]["has_poster"] is True
    assert enriched[0]["poster_filename"].lower() == "cover.webp"


def test_poster_uses_cached_filename_without_iterdir(client, monkeypatch):
    """When scan-cache has poster_filename, the endpoint must NOT call iterdir."""
    folder = client.library_dir / "Cached"  # type: ignore[attr-defined]
    folder.mkdir()
    (folder / "poster.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    # Prime scan cache with the poster_filename.
    (client.config_dir / "library.json").write_text(json.dumps({  # type: ignore[attr-defined]
        "instructionals": [
            {"name": "Cached", "path": str(folder), "has_poster": True, "poster_filename": "poster.jpg"},
        ]
    }))

    # Count calls to Path.iterdir — must stay at zero.
    calls = {"n": 0}
    import pathlib
    orig_iterdir = pathlib.Path.iterdir

    def counting_iterdir(self):
        calls["n"] += 1
        return orig_iterdir(self)

    monkeypatch.setattr(pathlib.Path, "iterdir", counting_iterdir)

    r = client.get("/api/library/Cached/poster")
    assert r.status_code == 200
    assert calls["n"] == 0, f"iterdir must not be called on hot path, got {calls['n']}"


def test_poster_falls_back_to_iterdir_without_cache(client):
    """With no cached poster_filename, endpoint still works via iterdir."""
    folder = client.library_dir / "NoCache"  # type: ignore[attr-defined]
    folder.mkdir()
    (folder / "poster.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    r = client.get("/api/library/NoCache/poster")
    assert r.status_code == 200


def test_poster_etag_and_304(client):
    """ETag returned and If-None-Match triggers 304."""
    folder = client.library_dir / "EtagInst"  # type: ignore[attr-defined]
    folder.mkdir()
    (folder / "poster.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    r = client.get("/api/library/EtagInst/poster")
    assert r.status_code == 200
    etag = r.headers.get("etag")
    assert etag, "ETag header must be present"
    cc = r.headers.get("cache-control", "")
    assert "max-age=86400" in cc
    assert "stale-while-revalidate" in cc

    r2 = client.get("/api/library/EtagInst/poster", headers={"If-None-Match": etag})
    assert r2.status_code == 304
    assert r2.headers.get("etag") == etag


def test_poster_detects_uppercase_poster_jpg(client):
    """/api/library/{name}/poster must find Poster.JPG (mixed case)."""
    folder = client.library_dir / "MyInst"  # type: ignore[attr-defined]
    folder.mkdir()
    (folder / "Poster.JPG").write_bytes(b"\xff\xd8\xff\xd9")
    r = client.get("/api/library/MyInst/poster")
    assert r.status_code == 200
    assert "image" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# /api/library/{name} detail endpoint
# ---------------------------------------------------------------------------

def _write_cache(client, instructionals):
    import json as _json
    (client.config_dir / "library.json").write_text(_json.dumps({"instructionals": instructionals}))  # type: ignore[attr-defined]


def test_library_detail_404_when_not_found(client):
    _write_cache(client, [])
    r = client.get("/api/library/Missing")
    assert r.status_code == 404


def test_library_detail_groups_videos_by_season(client):
    folder = client.library_dir / "Danaher"  # type: ignore[attr-defined]
    folder.mkdir()
    _write_cache(client, [{
        "name": "Danaher",
        "path": str(folder),
        "has_poster": False,
        "poster_filename": None,
        "videos": [
            {"path": str(folder / "Season 01" / "ep1.mkv"), "filename": "ep1.mkv"},
            {"path": str(folder / "Season 01" / "ep2.mkv"), "filename": "ep2.mkv"},
            {"path": str(folder / "Season 02" / "ep1.mkv"), "filename": "ep1.mkv"},
            {"path": str(folder / "extras" / "bonus.mkv"), "filename": "bonus.mkv"},
        ],
    }])
    # Detail endpoint refreshes filesystem-derived flags by default; pass
    # refresh=false so the cached (non-existent-on-disk) videos survive.
    r = client.get("/api/library/Danaher?refresh=false")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "Danaher"
    seasons = {v["season"] for v in data["videos"]}
    assert "Season 1" in seasons
    assert "Season 2" in seasons
    assert "Sin temporada" in seasons
