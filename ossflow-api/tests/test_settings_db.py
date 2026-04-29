"""Regression: settings endpoints now backed by unified SQLite DB."""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def env(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    db_path = tmp_path / "bjj.db"
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))

    # Reset engine cache so new DB path takes effect
    from bjj_service_kit.db import engine as eng_mod
    from bjj_service_kit.db import session as sess_mod
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    import api.settings as settings_mod
    importlib.reload(settings_mod)

    app = FastAPI()
    app.include_router(settings_mod.router)
    return {"client": TestClient(app), "config_dir": config_dir, "db_path": db_path,
            "settings_mod": settings_mod}


def test_get_returns_defaults_when_empty(env):
    r = env["client"].get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    assert data["library_path"] == ""
    assert data["telegram_api_id"] is None


def test_put_then_get_persists(env):
    payload = {"library_path": "/media/lib", "voice_profile_default": "voice_a"}
    r = env["client"].put("/api/settings", json=payload)
    assert r.status_code == 200
    r2 = env["client"].get("/api/settings")
    assert r2.json()["library_path"] == "/media/lib"
    assert r2.json()["voice_profile_default"] == "voice_a"


def test_put_validates_library_path_type(env):
    r = env["client"].put("/api/settings", json={"library_path": 123})
    assert r.status_code == 422


def test_put_validates_telegram_hash(env):
    r = env["client"].put("/api/settings", json={"telegram_api_hash": "nope"})
    assert r.status_code == 422


def test_put_accepts_valid_telegram_hash(env):
    h = "a" * 32
    r = env["client"].put("/api/settings", json={"telegram_api_hash": h, "telegram_api_id": 42})
    assert r.status_code == 200
    assert r.json()["telegram_api_hash"] == h
    assert r.json()["telegram_api_id"] == 42


def test_legacy_json_is_imported_and_backed_up(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    legacy = config_dir / "settings.json"
    legacy.write_text(json.dumps({"library_path": "/from/legacy"}), encoding="utf-8")

    db_path = tmp_path / "bjj.db"
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))

    from bjj_service_kit.db import engine as eng_mod, session as sess_mod
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    import api.settings as settings_mod
    importlib.reload(settings_mod)

    app = FastAPI()
    app.include_router(settings_mod.router)
    client = TestClient(app)

    r = client.get("/api/settings")
    assert r.json()["library_path"] == "/from/legacy"
    assert not legacy.exists()
    assert (config_dir / "settings.json.bak").exists()


def test_custom_prompts_and_author_aliases_roundtrip(env):
    payload = {
        "custom_prompts": {"chapters": "prompt A"},
        "author_aliases": {"danaher": "John Danaher", "": "  "},
    }
    r = env["client"].put("/api/settings", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["custom_prompts"] == {"chapters": "prompt A"}
    assert data["author_aliases"] == {"danaher": "John Danaher"}
