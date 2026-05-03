"""Service-level FastAPI smoke tests for dubbing-generator."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Make the dubbing-generator dir importable as a package root for `app`.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def test_app_state_carries_s2pro_manager_after_startup(monkeypatch):
    """Startup hook puts the manager on app.state, not in a global."""
    monkeypatch.delenv("DUBBING_TTS_ENGINE", raising=False)
    import app as dub_app
    dub_app._start_s2pro_server()
    assert hasattr(dub_app.app.state, "s2pro_manager")


def test_s2pro_status_route_present():
    import app as dub_app
    routes = {r.path for r in dub_app.app.router.routes}
    assert "/s2pro/status" in routes


def test_s2pro_models_route_present():
    import app as dub_app
    routes = {r.path for r in dub_app.app.router.routes}
    assert "/s2pro/models" in routes


def test_s2pro_models_lists_gguf_files(monkeypatch, tmp_path):
    """``/s2pro/models`` escanea el directorio y devuelve metadata útil."""
    # Crea un set realista: 2 GGUF, tokenizer presente, fichero ruido.
    (tmp_path / "s2-pro-q4_k_m.gguf").write_bytes(b"\x00" * 1024)
    (tmp_path / "s2-pro-q8_0.gguf").write_bytes(b"\x00" * 2048)
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("ignored", encoding="utf-8")

    from dubbing_generator.api import router as router_mod
    monkeypatch.setattr(router_mod, "S2PRO_MODELS_DIR", tmp_path)

    import app as dub_app
    from fastapi.testclient import TestClient

    with TestClient(dub_app.app) as client:
        r = client.get("/s2pro/models")

    assert r.status_code == 200
    body = r.json()
    assert body["dir_exists"] is True
    assert body["tokenizer_present"] is True
    quants = sorted(m["quant"] for m in body["models"])
    assert quants == ["q4_k_m", "q8_0"]
    sizes = {m["quant"]: m["size_bytes"] for m in body["models"]}
    assert sizes["q4_k_m"] == 1024
    assert sizes["q8_0"] == 2048


def test_s2pro_models_reports_missing_tokenizer(monkeypatch, tmp_path):
    (tmp_path / "s2-pro-q4_k_m.gguf").write_bytes(b"x")
    from dubbing_generator.api import router as router_mod
    monkeypatch.setattr(router_mod, "S2PRO_MODELS_DIR", tmp_path)

    import app as dub_app
    from fastapi.testclient import TestClient

    with TestClient(dub_app.app) as client:
        r = client.get("/s2pro/models")

    assert r.status_code == 200
    assert r.json()["tokenizer_present"] is False


def test_s2pro_models_handles_missing_dir(monkeypatch, tmp_path):
    missing = tmp_path / "does-not-exist"
    from dubbing_generator.api import router as router_mod
    monkeypatch.setattr(router_mod, "S2PRO_MODELS_DIR", missing)

    import app as dub_app
    from fastapi.testclient import TestClient

    with TestClient(dub_app.app) as client:
        r = client.get("/s2pro/models")

    assert r.status_code == 200
    body = r.json()
    assert body["dir_exists"] is False
    assert body["models"] == []
    assert body["tokenizer_present"] is False


def test_run_dubbing_generator_accepts_s2pro_engine():
    import app as dub_app
    src = inspect.getsource(dub_app._run_dubbing_generator)
    assert '"s2pro"' in src
    # S2-Pro is the default engine since the migration completed (memory:
    # project_s2pro_integration). Env-fallback resolves to s2pro when
    # DUBBING_TTS_ENGINE is unset.
    assert 'or "s2pro"' in src
