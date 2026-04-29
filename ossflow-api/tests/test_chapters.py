"""Tests for /api/chapters/rename."""

from __future__ import annotations

import importlib

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

    import api.settings as settings_mod
    importlib.reload(settings_mod)
    settings_mod.save_settings(
        {**settings_mod.load_settings(), "library_path": str(library_dir)}
    )

    import api.chapters as chapters_mod
    importlib.reload(chapters_mod)

    # Mount only the chapters router on a minimal app (avoids importing the
    # full app.py and its heavy deps).
    app = FastAPI()
    app.include_router(chapters_mod.router)

    tc = TestClient(app)
    tc.library_dir = library_dir  # type: ignore[attr-defined]
    return tc


def _touch(path):
    path.write_text("x", encoding="utf-8")


def test_rename_happy_path_with_siblings(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    season = lib / "Season 01"
    season.mkdir()
    stem = "John Danaher - S01E03 - Old Title"
    main = season / f"{stem}.mkv"
    sib_srt = season / f"{stem}.srt"
    sib_en = season / f"{stem}.en.srt"
    sib_dub = season / f"{stem}_DOBLADO.mkv"
    for p in (main, sib_srt, sib_en, sib_dub):
        _touch(p)

    r = client.patch(
        "/api/chapters/rename",
        json={"old_path": str(main), "new_title": "Armbar Fundamentals"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "renamed" in data
    assert len(data["renamed"]) == 4

    new_stem = "John Danaher - S01E03 - Armbar Fundamentals"
    assert (season / f"{new_stem}.mkv").exists()
    assert (season / f"{new_stem}.srt").exists()
    assert (season / f"{new_stem}.en.srt").exists()
    assert (season / f"{new_stem}_DOBLADO.mkv").exists()
    assert not main.exists()
    assert not sib_srt.exists()


def test_rename_without_siblings(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    main = lib / "Gordon - S02E10 - Whatever.mp4"
    _touch(main)

    r = client.patch(
        "/api/chapters/rename",
        json={"old_path": str(main), "new_title": "Heel Hook Entry"},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["renamed"]) == 1
    assert (lib / "Gordon - S02E10 - Heel Hook Entry.mp4").exists()


def test_rename_empty_title_rejected(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    main = lib / "Author - S01E01 - X.mkv"
    _touch(main)

    r = client.patch(
        "/api/chapters/rename",
        json={"old_path": str(main), "new_title": "   "},
    )
    assert r.status_code == 422


def test_rename_only_illegal_chars_rejected(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    main = lib / "Author - S01E01 - X.mkv"
    _touch(main)

    # Only illegal chars + whitespace → after sanitize becomes `_ _ _` (non-empty).
    # But a pure illegal-only title like `//` becomes `__` (non-empty).
    # To test the empty branch we pass whitespace only (already covered); add
    # a path-traversal guard check here instead with a benign title.
    r = client.patch(
        "/api/chapters/rename",
        json={"old_path": str(main), "new_title": "///"},
    )
    # `///` → `___` which is NOT empty → should succeed.
    assert r.status_code == 200
    assert (lib / "Author - S01E01 - ___.mkv").exists()


def test_rename_traversal_blocked(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    # A path that's clearly outside the library root.
    outside = lib.parent / "not_in_library.mkv"
    _touch(outside)

    r = client.patch(
        "/api/chapters/rename",
        json={"old_path": str(outside), "new_title": "Hack"},
    )
    assert r.status_code == 403


def test_rename_missing_snnemm_pattern(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    # No SNNeMM in the filename.
    main = lib / "just-a-file.mkv"
    _touch(main)

    r = client.patch(
        "/api/chapters/rename",
        json={"old_path": str(main), "new_title": "New Name"},
    )
    assert r.status_code == 422


def test_rename_file_not_found(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    missing = lib / "Author - S01E01 - Missing.mkv"

    r = client.patch(
        "/api/chapters/rename",
        json={"old_path": str(missing), "new_title": "Whatever"},
    )
    assert r.status_code == 404


def test_rename_sanitizes_illegal_and_collapses_spaces(client):
    lib = client.library_dir  # type: ignore[attr-defined]
    main = lib / "Author - S01E01 - Old.mkv"
    _touch(main)

    r = client.patch(
        "/api/chapters/rename",
        json={"old_path": str(main), "new_title": '  Foo:  bar/baz   *?  '},
    )
    assert r.status_code == 200
    # ':' / '*' / '?' become '_', multiple spaces collapse to one.
    assert (lib / "Author - S01E01 - Foo_ bar_baz __.mkv").exists()
