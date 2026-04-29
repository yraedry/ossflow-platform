"""Tests for /api/duplicates/scan."""

from __future__ import annotations

import importlib

import pytest
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
    settings_mod.save_settings({
        **settings_mod.load_settings(),
        "library_path": str(library_dir),
    })

    import api.app as app_mod
    importlib.reload(app_mod)
    import api.duplicates as dup_mod
    importlib.reload(dup_mod)

    # Include router if app.py hasn't been wired yet (keeps test hermetic).
    if not any(getattr(r, "path", "").startswith("/api/duplicates")
               for r in app_mod.app.routes):
        app_mod.app.include_router(dup_mod.router)

    # Mock ffprobe: duration is encoded in the first byte of the file for
    # deterministic tests. Fallback to 0 if unreadable.
    def fake_get_video_info(path: str) -> dict:
        try:
            with open(path, "rb") as fh:
                b = fh.read(1)
                duration = float(b[0]) if b else 0.0
        except OSError:
            duration = 0.0
        return {"duration": duration, "size_mb": 0}

    monkeypatch.setattr(app_mod, "get_video_info", fake_get_video_info)
    monkeypatch.setattr(
        "api.duplicates.get_video_info",
        fake_get_video_info,
        raising=False,
    )

    tc = TestClient(app_mod.app)
    tc.library_dir = library_dir  # type: ignore[attr-defined]
    return tc


def _mkvideo(path, size: int, duration_byte: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes([duration_byte]) + b"\x00" * max(0, size - 1)
    path.write_bytes(payload)


def test_two_equal_videos_form_one_group(client):
    lib = client.library_dir
    _mkvideo(lib / "a" / "v1.mkv", 1024, 30)
    _mkvideo(lib / "b" / "v2.mkv", 1024, 30)

    r = client.get(f"/api/duplicates/scan?path={lib}")
    assert r.status_code == 200
    data = r.json()
    assert len(data["groups"]) == 1
    assert len(data["groups"][0]) == 2
    assert data["stats"]["total_videos"] == 2
    assert data["stats"]["groups_found"] == 1
    assert data["stats"]["wasted_bytes"] == 1024


def test_distinct_videos_no_groups(client):
    lib = client.library_dir
    _mkvideo(lib / "v1.mkv", 1024, 30)
    _mkvideo(lib / "v2.mkv", 2048, 30)
    _mkvideo(lib / "v3.mkv", 1024, 45)

    r = client.get(f"/api/duplicates/scan?path={lib}")
    assert r.status_code == 200
    data = r.json()
    assert data["groups"] == []
    assert data["stats"]["groups_found"] == 0
    assert data["stats"]["wasted_bytes"] == 0


def test_traversal_outside_library_returns_403(client, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    r = client.get(f"/api/duplicates/scan?path={outside}")
    assert r.status_code == 403


def test_response_shape(client):
    lib = client.library_dir
    _mkvideo(lib / "only.mkv", 512, 10)
    r = client.get(f"/api/duplicates/scan?path={lib}")
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"groups", "stats"}
    assert set(data["stats"].keys()) == {"total_videos", "groups_found", "wasted_bytes"}
    assert isinstance(data["groups"], list)


def test_start_launches_background_job(client, tmp_path, monkeypatch):
    lib = client.library_dir
    _mkvideo(lib / "a" / "v1.mkv", 1024, 30)
    _mkvideo(lib / "b" / "v2.mkv", 1024, 30)

    from api import background_jobs as bg
    from api.background_jobs import JobRegistry
    fresh = JobRegistry(history_file=tmp_path / "bg.json")
    monkeypatch.setattr(bg, "registry", fresh)
    import api.duplicates as dup_mod
    monkeypatch.setattr(dup_mod, "_jobs_registry", fresh)

    r = client.post(f"/api/duplicates/start?path={lib}")
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert job_id

    import time as _t
    deadline = _t.time() + 3.0
    final = None
    while _t.time() < deadline:
        jr = client.get(f"/api/duplicates/job/{job_id}")
        assert jr.status_code == 200
        j = jr.json()
        if j["status"] in ("completed", "failed"):
            final = j
            break
        _t.sleep(0.02)

    assert final is not None, "job never completed"
    assert final["status"] == "completed", final
    assert final["type"] == "duplicates_scan"
    assert final["result"]["stats"]["groups_found"] == 1


def test_start_rejects_traversal(client, tmp_path):
    outside = tmp_path / "outside-dup"
    outside.mkdir()
    r = client.post(f"/api/duplicates/start?path={outside}")
    assert r.status_code == 403


def test_duplicates_job_endpoint_404(client):
    r = client.get("/api/duplicates/job/no-such-id")
    assert r.status_code == 404


def test_deep_mode_filters_by_partial_md5(client):
    lib = client.library_dir
    # Same size + duration but different content → deep must drop the group.
    p1 = lib / "a" / "v1.mkv"
    p2 = lib / "b" / "v2.mkv"
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)
    p1.write_bytes(bytes([30]) + b"\xaa" * 1023)
    p2.write_bytes(bytes([30]) + b"\xbb" * 1023)

    shallow = client.get(f"/api/duplicates/scan?path={lib}").json()
    assert shallow["stats"]["groups_found"] == 1

    deep = client.get(f"/api/duplicates/scan?path={lib}&deep=true").json()
    assert deep["stats"]["groups_found"] == 0
