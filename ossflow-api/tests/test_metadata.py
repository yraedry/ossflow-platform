"""Tests for /api/library/{name}/metadata sidecar endpoints."""

from __future__ import annotations

import json

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

    import importlib

    import api.settings as settings_mod
    importlib.reload(settings_mod)
    settings_mod.save_settings({**settings_mod.load_settings(), "library_path": str(library_dir)})

    import api.metadata as metadata_mod
    importlib.reload(metadata_mod)

    app = FastAPI()
    app.include_router(metadata_mod.router)

    tc = TestClient(app)
    tc.library_dir = library_dir  # type: ignore[attr-defined]
    return tc


def _mkfolder(client, name):
    folder = client.library_dir / name  # type: ignore[attr-defined]
    folder.mkdir()
    return folder


def test_get_metadata_defaults_when_missing(client):
    _mkfolder(client, "Foo")
    r = client.get("/api/library/Foo/metadata")
    assert r.status_code == 200
    assert r.json() == {
        "instructor": "",
        "topic": "",
        "tags": [],
        "synopsis": "",
        "year": None,
    }


def test_put_then_get_roundtrip(client):
    folder = _mkfolder(client, "Bar")
    payload = {
        "instructor": "John Danaher",
        "topic": "Arm Drags",
        "tags": ["no-gi", "guard"],
        "synopsis": "Fundamentals of arm drags.",
        "year": 2024,
    }
    r = client.put("/api/library/Bar/metadata", json=payload)
    assert r.status_code == 200
    assert r.json() == payload

    sidecar = folder / ".bjj-meta.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data == payload

    # File indented with 2 spaces
    assert "  \"instructor\"" in sidecar.read_text(encoding="utf-8")

    r2 = client.get("/api/library/Bar/metadata")
    assert r2.status_code == 200
    assert r2.json() == payload


def test_put_validates_types(client):
    _mkfolder(client, "Baz")

    r = client.put("/api/library/Baz/metadata", json={"instructor": 42})
    assert r.status_code == 422

    r = client.put("/api/library/Baz/metadata", json={"tags": "not-a-list"})
    assert r.status_code == 422

    r = client.put("/api/library/Baz/metadata", json={"tags": [1, 2, 3]})
    assert r.status_code == 422

    r = client.put("/api/library/Baz/metadata", json={"year": "abc"})
    assert r.status_code == 422

    r = client.put("/api/library/Baz/metadata", json={"synopsis": ["nope"]})
    assert r.status_code == 422


def test_put_accepts_null_year_and_defaults(client):
    _mkfolder(client, "Qux")
    r = client.put("/api/library/Qux/metadata", json={})
    assert r.status_code == 200
    assert r.json()["year"] is None
    assert r.json()["tags"] == []


def test_path_traversal_denied(client):
    r = client.get("/api/library/..%2F..%2Fetc/metadata")
    assert r.status_code in (403, 404)

    r = client.put("/api/library/..%2F..%2Fetc/metadata", json={})
    assert r.status_code in (403, 404)


def test_404_when_folder_missing(client):
    r = client.get("/api/library/NoSuchFolder/metadata")
    assert r.status_code == 404
