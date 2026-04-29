"""Verify that scan_library persists results to library.json with poster fields."""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    library = tmp_path / "library"
    library.mkdir()

    # Create a minimal instructional with poster
    instr = library / "Instr1"
    instr.mkdir()
    (instr / "Instr1_S01E01.mp4").write_bytes(b"x")
    (instr / "poster.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    # Second instructional without poster
    instr2 = library / "Instr2"
    instr2.mkdir()
    (instr2 / "video.mp4").write_bytes(b"x")

    monkeypatch.setenv("CONFIG_DIR", str(config_dir))

    import api.settings as sm
    importlib.reload(sm)
    sm.save_settings({**sm.load_settings(), "library_path": str(library)})

    import api.app as app_mod
    importlib.reload(app_mod)

    tc = TestClient(app_mod.app)
    tc.config_dir = config_dir  # type: ignore[attr-defined]
    tc.library = library  # type: ignore[attr-defined]
    return tc


def test_scan_writes_library_cache_with_poster_flags(client):
    r = client.post("/api/scan", json={"path": str(client.library)})  # type: ignore[attr-defined]
    assert r.status_code == 200

    cache = client.config_dir / "library.json"  # type: ignore[attr-defined]
    assert cache.exists(), "scan must persist library.json"

    data = json.loads(cache.read_text())
    by_name = {i["name"]: i for i in data["instructionals"]}
    assert by_name["Instr1"]["has_poster"] is True
    assert by_name["Instr1"]["poster_filename"] == "poster.jpg"
    assert by_name["Instr2"]["has_poster"] is False
    assert by_name["Instr2"]["poster_filename"] is None


def test_library_endpoint_serves_cache_after_scan(client):
    client.post("/api/scan", json={"path": str(client.library)})  # type: ignore[attr-defined]
    r = client.get("/api/library")
    assert r.status_code == 200
    names = {i["name"] for i in r.json()["instructionals"]}
    assert "Instr1" in names
