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


# ---------------------------------------------------------------------------
# T22.5 — limpieza TTS (eliminación motores no-S2-Pro + s2_quantization)
# ---------------------------------------------------------------------------


def test_s2_quantization_default_is_q6_k(env):
    """El default del schema tras T22.5 es q6_k."""
    r = env["client"].get("/api/settings")
    assert r.status_code == 200
    assert r.json()["s2_quantization"] == "q6_k"


def test_put_s2_quantization_accepts_any_safe_identifier(env):
    """Whitelist eliminada: cualquier identificador seguro vale.

    La UI descubre los GGUF disponibles vía
    ``GET /api/dubbing/s2pro/models`` leyendo el bind-mount del dubbing,
    así que el backend solo valida que el valor sea utilizable como
    nombre de fichero (sin path traversal).
    """
    for value in ("q4_k_m", "q6_k", "q8_0", "q5_k_m", "f16"):
        r = env["client"].put("/api/settings", json={"s2_quantization": value})
        assert r.status_code == 200
        assert r.json()["s2_quantization"] == value


def test_put_s2_quantization_normalizes_case(env):
    r = env["client"].put("/api/settings", json={"s2_quantization": "Q4_K_M"})
    assert r.status_code == 200
    assert r.json()["s2_quantization"] == "q4_k_m"


def test_put_s2_quantization_rejects_path_traversal(env):
    for bad in ("", "  ", "../etc/passwd", "q4/k", "q4-k", "q4 k"):
        r = env["client"].put("/api/settings", json={"s2_quantization": bad})
        assert r.status_code == 422, f"unexpected accept for {bad!r}"


def test_put_s2_quantization_rejects_non_string(env):
    r = env["client"].put("/api/settings", json={"s2_quantization": 123})
    assert r.status_code == 422


def test_legacy_tts_keys_removed_from_schema(env):
    """Los settings de motores eliminados ya no existen en DEFAULTS."""
    r = env["client"].get("/api/settings")
    body = r.json()
    for key in (
        "tts_engine",
        "elevenlabs_voice_id",
        "elevenlabs_model_id",
        "piper_model_path",
        "kokoro_voice",
    ):
        assert key not in body, f"campo legacy '{key}' aún en schema"


def test_legacy_tts_settings_migration_deletes_obsolete_rows(tmp_path, monkeypatch):
    """Migración silenciosa: filas TTS legacy se borran al primer init."""
    from ossflow_service_kit.db import engine as eng_mod
    from ossflow_service_kit.db import session as sess_mod

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    db_path = tmp_path / "bjj.db"
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))

    eng_mod.reset_engine()
    sess_mod.reset_factory()

    # Pre-seed la BD con filas legacy (simula un usuario actualizando).
    from ossflow_service_kit.db import init_db, session_scope
    from ossflow_service_kit.db.models import Setting
    init_db()
    with session_scope() as s:
        s.add(Setting(key="tts_engine", value='"elevenlabs"'))
        s.add(Setting(key="elevenlabs_voice_id", value='"abc123"'))
        s.add(Setting(key="piper_model_path", value='"/tmp/piper.onnx"'))
        s.add(Setting(key="kokoro_voice", value='"em_santa"'))

    import importlib
    from ossflow_api.modules.settings import schemas as schemas_mod
    importlib.reload(schemas_mod)
    from ossflow_api.modules.settings import service as service_mod
    importlib.reload(service_mod)
    reset_for_tests()

    svc = service_mod.SettingsService()

    try:
        # Forzar la inicialización (que ejecuta la migración).
        svc.ensure_initialized()
        # Las filas legacy ya no deben estar.
        with session_scope() as s:
            for key in ("tts_engine", "elevenlabs_voice_id", "piper_model_path", "kokoro_voice"):
                assert s.get(Setting, key) is None, f"fila '{key}' no fue migrada"
        # El load() devuelve solo defaults vigentes (sin los keys legacy).
        loaded = svc.load()
        assert "tts_engine" not in loaded
        assert "elevenlabs_voice_id" not in loaded
    finally:
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
