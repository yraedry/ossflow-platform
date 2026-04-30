"""Tests del filesystem browser y NAS mount (T23.4).

Cubren los endpoints ``/api/fs/browse``, ``/api/browse`` y ``/api/mount``.
``/api/mount`` POST se testea solo en su parte de validación (el
``subprocess.run`` real lo dejamos fuera porque requiere ``mount.cifs`` y
permisos root).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.library import library_router
from ossflow_api.modules.library import filesystem as fs_mod
from ossflow_api.modules.library import mount as mount_mod
from ossflow_api.modules.library.cache import LibraryCache
from ossflow_api.modules.library.dependencies import get_library_service
from ossflow_api.modules.library.service import LibraryService


@pytest.fixture
def env(tmp_path, monkeypatch):
    media_root = tmp_path / "media"
    media_root.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    monkeypatch.setenv("MEDIA_ROOT", str(media_root))
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))

    cache = LibraryCache(config_dir / "library.json")
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
        "config": config_dir,
    }


# ---------------------------------------------------------------------------
# /api/fs/browse
# ---------------------------------------------------------------------------


def test_fs_browse_lists_root_when_no_path(env):
    (env["media"] / "Folder A").mkdir()
    (env["media"] / "Folder B").mkdir()
    (env["media"] / ".hidden").mkdir()

    resp = env["client"].get("/api/fs/browse")
    assert resp.status_code == 200
    body = resp.json()
    names = [e["name"] for e in body["entries"]]
    assert "Folder A" in names
    assert "Folder B" in names
    assert ".hidden" not in names
    assert body["parent"] is None


def test_fs_browse_lists_subdir_with_parent(env):
    sub = env["media"] / "sub"
    sub.mkdir()
    (sub / "child").mkdir()

    resp = env["client"].get("/api/fs/browse", params={"path": str(sub)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["parent"] == str(env["media"])
    assert any(e["name"] == "child" for e in body["entries"])


def test_fs_browse_rejects_traversal(env, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    resp = env["client"].get("/api/fs/browse", params={"path": str(outside)})
    assert resp.status_code == 400


def test_fs_browse_404_when_dir_missing(env):
    resp = env["client"].get(
        "/api/fs/browse", params={"path": str(env["media"] / "ghost")}
    )
    assert resp.status_code == 404


def test_fs_browse_503_when_media_root_missing(tmp_path, monkeypatch):
    fake_root = tmp_path / "ghost_media"
    monkeypatch.setenv("MEDIA_ROOT", str(fake_root))
    cache = LibraryCache(tmp_path / "library.json")
    svc = LibraryService(cache=cache, library_path_loader=lambda: None)
    app = FastAPI()
    app.include_router(library_router)
    app.dependency_overrides[get_library_service] = lambda: svc
    client = TestClient(app)

    resp = client.get("/api/fs/browse")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /api/browse
# ---------------------------------------------------------------------------


def test_browse_defaults_to_library_path(env):
    course = env["library"] / "Course"
    course.mkdir()
    video = course / "ep1.mp4"
    video.write_bytes(b"\x00" * 100)

    resp = env["client"].get("/api/browse")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current"] == str(env["library"])
    names = [d["name"] for d in body["directories"]]
    assert "Course" in names


def test_browse_lists_video_files(env):
    sub = env["library"] / "season"
    sub.mkdir()
    (sub / "video.mp4").write_bytes(b"\x00" * 50)
    (sub / "notes.txt").write_text("ignored")

    resp = env["client"].get("/api/browse", params={"path": str(sub)})
    assert resp.status_code == 200
    body = resp.json()
    file_names = [f["name"] for f in body["files"]]
    assert "video.mp4" in file_names
    assert "notes.txt" not in file_names


def test_browse_returns_404_when_path_does_not_exist(env, tmp_path, monkeypatch):
    # Quitar MEDIA_ROOT a algo inexistente y pedir un path inexistente.
    bogus = tmp_path / "nope_media"
    monkeypatch.setenv("MEDIA_ROOT", str(bogus))
    cache = LibraryCache(tmp_path / "library.json")
    svc = LibraryService(cache=cache, library_path_loader=lambda: None)
    app = FastAPI()
    app.include_router(library_router)
    app.dependency_overrides[get_library_service] = lambda: svc
    client = TestClient(app)

    resp = client.get("/api/browse", params={"path": "/totally/nonexistent"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/mount
# ---------------------------------------------------------------------------


def test_mount_post_requires_share(env):
    resp = env["client"].post("/api/mount", json={})
    assert resp.status_code == 422


def test_mount_post_translates_share_format(env, monkeypatch):
    """Verifica que se llama subprocess con el share normalizado."""
    captured = {}

    class _OK:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _OK()

    monkeypatch.setattr(mount_mod.subprocess, "run", fake_run)

    resp = env["client"].post(
        "/api/mount",
        json={"share": "10.10.0.1\\share\\path", "username": "u", "password": "p"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mounted"] is True
    assert body["share"] == "//10.10.0.1/share/path"
    # mount.json persistido
    cfg = (env["config"] / "mount.json").read_text()
    assert "10.10.0.1" in cfg


def test_mount_post_reports_failure(env, monkeypatch):
    class _Fail:
        returncode = 1
        stderr = "permission denied"

    monkeypatch.setattr(mount_mod.subprocess, "run", lambda *a, **kw: _Fail())

    resp = env["client"].post("/api/mount", json={"share": "//10/share"})
    assert resp.status_code == 500
    assert "No se pudo montar" in resp.json()["error"]


def test_mount_get_status_reports_unmounted(env, monkeypatch):
    class _NotMount:
        returncode = 1

    monkeypatch.setattr(mount_mod.subprocess, "run", lambda *a, **kw: _NotMount())

    resp = env["client"].get("/api/mount")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mounted"] is False
    assert body["path"] == str(env["media"])
