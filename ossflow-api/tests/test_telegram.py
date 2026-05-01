"""Tests de credenciales de Telegram en el módulo settings.

Los tests del proxy ``/api/telegram/*`` viven ahora en
``tests/modules/telegram/test_router.py`` (migración Plan 2 task 18).
Aquí se mantienen sólo los tests de settings que validan
``telegram_api_id`` / ``telegram_api_hash`` porque están entrelazados
con la persistencia de settings y no con el proxy en sí.
"""

from __future__ import annotations

import importlib

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

    from ossflow_service_kit.db import engine as _eng, session as _sess
    _eng.reset_engine()
    _sess.reset_factory()

    import api.settings as settings_mod
    importlib.reload(settings_mod)

    app = FastAPI()
    app.include_router(settings_mod.router)

    return {"client": TestClient(app), "settings_mod": settings_mod}


# ---------------------------------------------------------------------------
# Settings: telegram credentials
# ---------------------------------------------------------------------------


def test_settings_accepts_telegram_credentials(env):
    client = env["client"]
    r = client.put(
        "/api/settings",
        json={
            "telegram_api_id": 123456,
            "telegram_api_hash": "a" * 32,
        },
    )
    assert r.status_code == 200
    body = r.json()
    # PUT echoes the unmasked value back so the frontend's optimistic update
    # has the real hash without an extra round-trip.
    assert body["telegram_api_id"] == 123456
    assert body["telegram_api_hash"] == "a" * 32

    # Public GET masks the hash to avoid leaking it to the browser.
    r = client.get("/api/settings")
    assert r.status_code == 200
    assert r.json()["telegram_api_id"] == 123456
    assert r.json()["telegram_api_hash"] == "***"

    # The internal endpoint (used by telegram-fetcher) returns the real hash.
    r = client.get("/api/settings/internal")
    assert r.status_code == 200
    assert r.json()["telegram_api_id"] == 123456
    assert r.json()["telegram_api_hash"] == "a" * 32


def test_settings_defaults_telegram_none(env):
    r = env["client"].get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["telegram_api_id"] is None
    assert body["telegram_api_hash"] is None


def test_settings_rejects_bad_api_id(env):
    r = env["client"].put(
        "/api/settings", json={"telegram_api_id": "not-an-int"}
    )
    assert r.status_code == 422


def test_settings_rejects_bad_api_hash(env):
    r = env["client"].put(
        "/api/settings", json={"telegram_api_hash": "short"}
    )
    assert r.status_code == 422

    r = env["client"].put(
        "/api/settings", json={"telegram_api_hash": "z" * 32}  # not hex
    )
    assert r.status_code == 422


def test_settings_allows_null_telegram(env):
    client = env["client"]
    client.put(
        "/api/settings",
        json={"telegram_api_id": 42, "telegram_api_hash": "a" * 32},
    )
    r = client.put(
        "/api/settings",
        json={"telegram_api_id": None, "telegram_api_hash": None},
    )
    assert r.status_code == 200
    assert r.json()["telegram_api_id"] is None
    assert r.json()["telegram_api_hash"] is None
