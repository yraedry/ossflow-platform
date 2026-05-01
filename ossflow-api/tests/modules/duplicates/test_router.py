"""Tests del módulo duplicates."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.duplicates import duplicates_router
from ossflow_api.modules.duplicates.dependencies import get_duplicates_service
from ossflow_api.modules.duplicates.service import DuplicatesService
from ossflow_api.modules.jobs._internal.scheduler import JobsScheduler
from ossflow_api.modules.jobs.repositories.background import BackgroundJobsRepository
from ossflow_api.modules.jobs.services.background import BackgroundJobsService


def _fake_video_info(path: str) -> dict:
    """Mock determinista: duración = primer byte del archivo."""
    try:
        with open(path, "rb") as fh:
            b = fh.read(1)
            duration = float(b[0]) if b else 0.0
    except OSError:
        duration = 0.0
    return {"duration": duration, "size_mb": 0}


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

    dup_svc = DuplicatesService(
        jobs=bg_svc,
        library_path_loader=lambda: str(library_dir),
        video_info_loader=_fake_video_info,
    )

    app = FastAPI()
    app.include_router(duplicates_router)
    app.dependency_overrides[get_duplicates_service] = lambda: dup_svc

    tc = TestClient(app)
    tc.library_dir = library_dir  # type: ignore[attr-defined]
    yield tc

    eng_mod.reset_engine()
    sess_mod.reset_factory()


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


def test_start_launches_background_job(client):
    lib = client.library_dir
    _mkvideo(lib / "a" / "v1.mkv", 1024, 30)
    _mkvideo(lib / "b" / "v2.mkv", 1024, 30)

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


def test_job_endpoint_404(client):
    r = client.get("/api/duplicates/job/no-such-id")
    assert r.status_code == 404


def test_deep_mode_filters_by_partial_md5(client):
    lib = client.library_dir
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
