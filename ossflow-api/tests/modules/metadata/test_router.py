"""Tests del router del módulo metadata."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.metadata import metadata_router
from ossflow_api.modules.metadata.dependencies import get_metadata_service
from ossflow_api.modules.metadata.service import MetadataService

DEFAULT_FULL = {
    "instructor": "",
    "topic": "",
    "tags": [],
    "synopsis": "",
    "year": None,
    "voice_profile": "",
}


@pytest.fixture
def client(tmp_path):
    library_dir = tmp_path / "library"
    library_dir.mkdir()

    app = FastAPI()
    app.include_router(metadata_router)
    app.dependency_overrides[get_metadata_service] = lambda: MetadataService(str(library_dir))

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
    assert r.json() == DEFAULT_FULL


def test_put_then_get_roundtrip(client):
    folder = _mkfolder(client, "Bar")
    payload = {
        "instructor": "John Danaher",
        "topic": "Arm Drags",
        "tags": ["no-gi", "guard"],
        "synopsis": "Fundamentals of arm drags.",
        "year": 2024,
        "voice_profile": "",
    }
    r = client.put("/api/library/Bar/metadata", json=payload)
    assert r.status_code == 200
    assert r.json() == payload

    sidecar = folder / ".bjj-meta.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data == payload
    assert "  \"instructor\"" in sidecar.read_text(encoding="utf-8")

    r2 = client.get("/api/library/Bar/metadata")
    assert r2.status_code == 200
    assert r2.json() == payload


def test_put_validates_types(client):
    _mkfolder(client, "Baz")

    assert client.put("/api/library/Baz/metadata", json={"instructor": 42}).status_code == 422
    assert client.put("/api/library/Baz/metadata", json={"tags": "not-a-list"}).status_code == 422
    assert client.put("/api/library/Baz/metadata", json={"tags": [1, 2, 3]}).status_code == 422
    assert client.put("/api/library/Baz/metadata", json={"year": "abc"}).status_code == 422
    assert client.put("/api/library/Baz/metadata", json={"synopsis": ["nope"]}).status_code == 422
    assert client.put("/api/library/Baz/metadata", json={"voice_profile": 1}).status_code == 422


def test_put_accepts_null_year_and_defaults(client):
    _mkfolder(client, "Qux")
    r = client.put("/api/library/Qux/metadata", json={})
    assert r.status_code == 200
    assert r.json()["year"] is None
    assert r.json()["tags"] == []
    assert r.json()["voice_profile"] == ""


def test_path_traversal_denied(client):
    r = client.get("/api/library/..%2F..%2Fetc/metadata")
    assert r.status_code in (403, 404)
    r = client.put("/api/library/..%2F..%2Fetc/metadata", json={})
    assert r.status_code in (403, 404)


def test_404_when_folder_missing(client):
    r = client.get("/api/library/NoSuchFolder/metadata")
    assert r.status_code == 404


def test_get_recovers_from_corrupt_sidecar(client):
    folder = _mkfolder(client, "Corrupt")
    (folder / ".bjj-meta.json").write_text("not json", encoding="utf-8")
    r = client.get("/api/library/Corrupt/metadata")
    assert r.status_code == 200
    assert r.json() == DEFAULT_FULL
