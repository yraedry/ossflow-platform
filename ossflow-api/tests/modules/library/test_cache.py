"""Tests del LibraryCache (renombrado de ScanCache en T23.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ossflow_api.modules.library.cache import (
    POSTER_NAMES,
    LibraryCache,
    enrich_with_poster,
    find_poster,
    find_poster_cached,
    patch_poster_in_cache,
)


# ---------------------------------------------------------------------------
# LibraryCache
# ---------------------------------------------------------------------------


def test_cache_does_not_exist_initially(tmp_path):
    cache = LibraryCache(tmp_path / "library.json")
    assert cache.exists() is False
    assert cache.load() is None


def test_save_then_load_roundtrip(tmp_path):
    cache = LibraryCache(tmp_path / "library.json")
    items = [{"name": "Course 1", "path": "/x", "videos": []}]
    cache.save(items)
    assert cache.exists() is True
    loaded = cache.load()
    assert loaded == {"instructionals": items}


def test_save_creates_parent_dir(tmp_path):
    cache = LibraryCache(tmp_path / "deep" / "nested" / "library.json")
    cache.save([{"name": "a"}])
    assert cache.exists()


def test_load_returns_none_on_corrupt_json(tmp_path):
    path = tmp_path / "library.json"
    path.write_text("not valid json", encoding="utf-8")
    cache = LibraryCache(path)
    assert cache.load() is None


def test_save_uses_atomic_replace(tmp_path):
    """``save`` escribe a ``.tmp`` antes de ``os.replace``. Tras un guardado
    exitoso no debe quedar ningún ``.tmp`` orphan."""
    path = tmp_path / "library.json"
    cache = LibraryCache(path)
    cache.save([{"name": "x"}])
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


# ---------------------------------------------------------------------------
# find_poster
# ---------------------------------------------------------------------------


def test_find_poster_returns_none_when_folder_missing(tmp_path):
    assert find_poster(tmp_path / "missing") is None


def test_find_poster_returns_none_when_no_image(tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"x")
    assert find_poster(tmp_path) is None


def test_find_poster_prefers_canonical_name(tmp_path):
    (tmp_path / "random.jpg").write_bytes(b"x")
    (tmp_path / "poster.jpg").write_bytes(b"x")
    found = find_poster(tmp_path)
    assert found is not None
    assert found.name == "poster.jpg"


def test_find_poster_falls_back_to_any_image(tmp_path):
    (tmp_path / "first.png").write_bytes(b"x")
    (tmp_path / "second.jpg").write_bytes(b"x")
    found = find_poster(tmp_path)
    assert found is not None
    # Sorted alphabetically
    assert found.name == "first.png"


def test_find_poster_canonical_names_match_constant():
    """Sanity: la lista de nombres canónicos no se ha vaciado."""
    assert "poster.jpg" in POSTER_NAMES
    assert "cover.jpg" in POSTER_NAMES
    assert "folder.jpg" in POSTER_NAMES


# ---------------------------------------------------------------------------
# find_poster_cached
# ---------------------------------------------------------------------------


def test_find_poster_cached_uses_hint_first(tmp_path):
    """Si se pasa ``poster_filename`` y existe, no hace iterdir."""
    poster = tmp_path / "custom.jpg"
    poster.write_bytes(b"x")
    found = find_poster_cached(tmp_path, "custom.jpg")
    assert found == poster


def test_find_poster_cached_falls_back_when_hint_missing(tmp_path):
    (tmp_path / "poster.png").write_bytes(b"x")
    # Hint apunta a fichero inexistente → fallback a find_poster.
    found = find_poster_cached(tmp_path, "ghost.jpg")
    assert found is not None
    assert found.name == "poster.png"


def test_find_poster_cached_no_hint_falls_back(tmp_path):
    (tmp_path / "poster.png").write_bytes(b"x")
    found = find_poster_cached(tmp_path, None)
    assert found is not None


# ---------------------------------------------------------------------------
# enrich_with_poster
# ---------------------------------------------------------------------------


def test_enrich_adds_poster_fields(tmp_path):
    folder = tmp_path / "course"
    folder.mkdir()
    (folder / "poster.jpg").write_bytes(b"x")
    items = [{"name": "Course", "path": str(folder)}]

    enriched = enrich_with_poster(items)

    assert enriched[0]["has_poster"] is True
    assert enriched[0]["poster_filename"] == "poster.jpg"
    assert isinstance(enriched[0]["poster_mtime"], int)


def test_enrich_handles_missing_poster(tmp_path):
    folder = tmp_path / "course"
    folder.mkdir()
    items = [{"name": "Course", "path": str(folder)}]

    enriched = enrich_with_poster(items)

    assert enriched[0]["has_poster"] is False
    assert enriched[0]["poster_filename"] is None
    assert enriched[0]["poster_mtime"] is None


# ---------------------------------------------------------------------------
# patch_poster_in_cache
# ---------------------------------------------------------------------------


def test_patch_updates_existing_item(tmp_path):
    cache_path = tmp_path / "library.json"
    folder = tmp_path / "course"
    folder.mkdir()
    poster = folder / "newposter.jpg"
    poster.write_bytes(b"x")

    cache = LibraryCache(cache_path)
    cache.save([{
        "name": "Course",
        "path": str(folder),
        "has_poster": False,
        "poster_filename": None,
    }])

    changed = patch_poster_in_cache(cache, "Course", "newposter.jpg")
    assert changed is True

    loaded = cache.load()
    item = loaded["instructionals"][0]
    assert item["has_poster"] is True
    assert item["poster_filename"] == "newposter.jpg"
    assert isinstance(item["poster_mtime"], int)


def test_patch_returns_false_when_item_not_found(tmp_path):
    cache = LibraryCache(tmp_path / "library.json")
    cache.save([{"name": "Other"}])
    changed = patch_poster_in_cache(cache, "Course", "x.jpg")
    assert changed is False


def test_patch_returns_false_when_cache_empty(tmp_path):
    cache = LibraryCache(tmp_path / "library.json")
    changed = patch_poster_in_cache(cache, "Course", "x.jpg")
    assert changed is False


def test_patch_clears_poster_fields_when_filename_none(tmp_path):
    cache = LibraryCache(tmp_path / "library.json")
    cache.save([{
        "name": "Course",
        "path": "/x",
        "has_poster": True,
        "poster_filename": "old.jpg",
    }])
    changed = patch_poster_in_cache(cache, "Course", None)
    assert changed is True
    item = cache.load()["instructionals"][0]
    assert item["has_poster"] is False
    assert item["poster_filename"] is None


# ---------------------------------------------------------------------------
# Compat shim — verifica que api/scan_cache.py sigue funcionando
# ---------------------------------------------------------------------------


def test_legacy_shim_still_works(tmp_path):
    """api/scan_cache.py debe seguir reexportando las APIs antiguas."""
    from api.scan_cache import (
        ScanCache,
        find_poster as fp_legacy,
        enrich_with_poster as ewp_legacy,
        POSTER_NAMES as pn_legacy,
    )
    # ScanCache es alias de LibraryCache.
    assert ScanCache is LibraryCache
    cache = ScanCache(tmp_path / "library.json")
    cache.save([{"name": "x"}])
    assert cache.exists()
    # find_poster y enrich_with_poster son los mismos objetos.
    assert fp_legacy is find_poster
    assert ewp_legacy is enrich_with_poster
    assert pn_legacy == POSTER_NAMES
