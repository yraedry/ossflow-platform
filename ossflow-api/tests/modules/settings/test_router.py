"""Tests del módulo settings (BD SQLite + endpoints HTTP)."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.settings import settings_router
from ossflow_api.modules.settings.dependencies import (
    get_settings_service,
    reset_for_tests,
)
from ossflow_api.modules.settings.service import SettingsService


@pytest.fixture
def env(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    db_path = tmp_path / "bjj.db"
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))

    # Reset engine cache para que el nuevo DB path tenga efecto.
    from ossflow_service_kit.db import engine as eng_mod
    from ossflow_service_kit.db import session as sess_mod
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    # ``CONFIG_DIR`` es leído al import time del módulo schemas, así que
    # necesitamos recargar el módulo para que recoja la nueva env var.
    import importlib
    from ossflow_api.modules.settings import schemas as schemas_mod
    importlib.reload(schemas_mod)
    # Y service hereda de schemas — recargamos también para que use el nuevo
    # ``LEGACY_SETTINGS_FILE``.
    from ossflow_api.modules.settings import service as service_mod
    importlib.reload(service_mod)

    reset_for_tests()
    svc = service_mod.SettingsService()

    app = FastAPI()
    app.include_router(settings_router)
    app.dependency_overrides[get_settings_service] = lambda: svc

    yield {
        "client": TestClient(app),
        "config_dir": config_dir,
        "db_path": db_path,
        "service": svc,
    }

    # Teardown: deja el engine y los singletons limpios para que los tests
    # siguientes (potencialmente con BJJ_DB_PATH del entorno real) no usen
    # nuestra BD efímera.
    eng_mod.reset_engine()
    sess_mod.reset_factory()
    importlib.reload(schemas_mod)
    importlib.reload(service_mod)
    reset_for_tests()


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

    from ossflow_service_kit.db import engine as eng_mod
    from ossflow_service_kit.db import session as sess_mod
    eng_mod.reset_engine()
    sess_mod.reset_factory()

    import importlib
    from ossflow_api.modules.settings import schemas as schemas_mod
    importlib.reload(schemas_mod)
    from ossflow_api.modules.settings import service as service_mod
    importlib.reload(service_mod)

    reset_for_tests()
    svc = service_mod.SettingsService()

    app = FastAPI()
    app.include_router(settings_router)
    app.dependency_overrides[get_settings_service] = lambda: svc
    client = TestClient(app)

    try:
        r = client.get("/api/settings")
        assert r.json()["library_path"] == "/from/legacy"
        assert not legacy.exists()
        assert (config_dir / "settings.json.bak").exists()
    finally:
        # Teardown: igual que el fixture env, dejamos engine/singletons limpios.
        eng_mod.reset_engine()
        sess_mod.reset_factory()
        importlib.reload(schemas_mod)
        importlib.reload(service_mod)
        reset_for_tests()


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
