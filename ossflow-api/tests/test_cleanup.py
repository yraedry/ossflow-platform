"""Tests para api.cleanup."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import cleanup as cleanup_mod
from api.cleanup import router as cleanup_router


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Monta la API y apunta library_path al tmp_path."""
    monkeypatch.setattr(cleanup_mod, "get_library_path", lambda: str(tmp_path))
    app = FastAPI()
    app.include_router(cleanup_router)
    return TestClient(app), tmp_path


def _touch(p: Path, content: bytes = b"x", mtime: float | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def test_scan_detects_orphan_srt(client):
    c, root = client
    _touch(root / "pair.mkv", b"video")
    _touch(root / "pair.en.srt", b"sub")  # emparejado por base stem
    _touch(root / "lonely.srt", b"orphan")

    resp = c.get("/api/cleanup/scan", params={"path": str(root)})
    assert resp.status_code == 200
    data = resp.json()
    orphans = [Path(x["path"]).name for x in data["categories"]["orphan_srt"]]
    assert "lonely.srt" in orphans
    assert "pair.en.srt" not in orphans


def test_scan_detects_old_dubbed(client):
    c, root = client
    now = time.time()
    _touch(root / "video.ES.srt", b"newer subs", mtime=now)
    _touch(root / "video_DOBLADO.mkv", b"older dub", mtime=now - 10_000)

    resp = c.get("/api/cleanup/scan", params={"path": str(root)})
    assert resp.status_code == 200
    old = [Path(x["path"]).name for x in resp.json()["categories"]["old_dubbed"]]
    assert "video_DOBLADO.mkv" in old


def test_scan_detects_temp_files(client):
    c, root = client
    _touch(root / "a.tmp")
    _touch(root / "b.part")
    _touch(root / "c.crdownload")
    _touch(root / "~stuff.doc")
    _touch(root / "backup.bak")
    _touch(root / "keep.mkv")

    resp = c.get("/api/cleanup/scan", params={"path": str(root)})
    temps = {Path(x["path"]).name for x in resp.json()["categories"]["temp_files"]}
    assert temps == {"a.tmp", "b.part", "c.crdownload", "~stuff.doc", "backup.bak"}


def test_scan_detects_empty_dirs(client):
    c, root = client
    (root / "empty1").mkdir()
    (root / "non_empty").mkdir()
    _touch(root / "non_empty" / "file.mkv")
    (root / "deep" / "empty2").mkdir(parents=True)

    resp = c.get("/api/cleanup/scan", params={"path": str(root)})
    empties = {Path(x["path"]).name for x in resp.json()["categories"]["empty_dirs"]}
    assert "empty1" in empties
    assert "empty2" in empties
    assert "non_empty" not in empties


def test_scan_shape_and_totals(client):
    c, root = client
    _touch(root / "x.tmp", b"abcde")
    _touch(root / "orph.srt", b"12")
    resp = c.get("/api/cleanup/scan", params={"path": str(root)})
    data = resp.json()
    assert set(data["categories"].keys()) == {
        "orphan_srt", "old_dubbed", "temp_files", "empty_dirs"
    }
    assert data["total_items"] >= 2
    assert data["total_bytes"] >= 7


def test_scan_rejects_traversal(client):
    c, root = client
    # Intento de ir fuera de root
    outside = root.parent / "not_under_lib"
    outside.mkdir(exist_ok=True)
    resp = c.get("/api/cleanup/scan", params={"path": str(outside)})
    assert resp.status_code == 403


def test_apply_dry_run_does_not_delete(client):
    c, root = client
    f = _touch(root / "garbage.tmp", b"abc")
    resp = c.post(
        "/api/cleanup/apply",
        json={"paths": [str(f)], "dry_run": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert str(f) in data["deleted"]
    assert f.exists()  # dry_run: no lo borra
    assert data["dry_run"] is True


def test_apply_real_deletes_and_rejects_plain_video(client):
    c, root = client
    f_tmp = _touch(root / "junk.bak", b"abcd")
    f_video = _touch(root / "clean.mkv", b"keep")
    # vídeo doblado sí se puede borrar
    f_dub = _touch(root / "v_DOBLADO.mkv", b"dub")

    resp = c.post(
        "/api/cleanup/apply",
        json={
            "paths": [str(f_tmp), str(f_video), str(f_dub)],
            "dry_run": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert str(f_tmp) in data["deleted"]
    assert str(f_dub) in data["deleted"]
    assert not f_tmp.exists()
    assert not f_dub.exists()
    # el vídeo plano debe permanecer y aparecer como error
    assert f_video.exists()
    assert any(str(f_video) == e["path"] for e in data["errors"])
    assert data["freed_bytes"] >= 4


def test_start_launches_background_job(client, monkeypatch, tmp_path):
    c, root = client
    _touch(root / "junk.tmp", b"abc")
    _touch(root / "lonely.srt", b"ab")

    # Point the registry to a fresh tmp history file so this test is hermetic
    from api import background_jobs as bg
    from api.background_jobs import JobRegistry
    fresh = JobRegistry(history_file=tmp_path / "bg.json")
    monkeypatch.setattr(bg, "registry", fresh)
    import api.cleanup as cleanup_mod
    monkeypatch.setattr(cleanup_mod, "_jobs_registry", fresh)

    resp = c.post(f"/api/cleanup/start?path={root}")
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    # Poll /job/{id} until it finishes
    import time as _t
    deadline = _t.time() + 3.0
    final = None
    while _t.time() < deadline:
        jr = c.get(f"/api/cleanup/job/{job_id}")
        assert jr.status_code == 200
        j = jr.json()
        if j["status"] in ("completed", "failed"):
            final = j
            break
        _t.sleep(0.02)

    assert final is not None, "job never completed"
    assert final["status"] == "completed"
    assert final["type"] == "cleanup_scan"
    assert final["result"] is not None
    assert "categories" in final["result"]
    assert final["result"]["total_items"] >= 2


def test_start_rejects_traversal(client):
    c, root = client
    outside = root.parent / "out_of_lib"
    outside.mkdir(exist_ok=True)
    try:
        r = c.post(f"/api/cleanup/start?path={outside}")
        assert r.status_code == 403
    finally:
        outside.rmdir()


def test_job_endpoint_404(client):
    c, _ = client
    r = c.get("/api/cleanup/job/does-not-exist")
    assert r.status_code == 404


def test_apply_traversal_denied(client):
    c, root = client
    outside = root.parent / "outside.txt"
    outside.write_bytes(b"nope")
    try:
        resp = c.post(
            "/api/cleanup/apply",
            json={"paths": [str(outside)], "dry_run": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == []
        assert len(data["errors"]) == 1
        assert outside.exists()
    finally:
        outside.unlink(missing_ok=True)
