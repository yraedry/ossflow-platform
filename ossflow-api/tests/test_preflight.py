"""Tests for api.preflight."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from collections import namedtuple
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import preflight


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_preflight_cache() -> None:
    """Cache del preflight es estado global de módulo → limpiar entre tests."""
    preflight._invalidate_cache()
    yield
    preflight._invalidate_cache()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app = FastAPI()
    app.include_router(preflight.router)
    return TestClient(app)


@pytest.fixture
def happy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch filesystem + network helpers so every check succeeds."""
    # Real dir => check_path returns ok
    target = tmp_path / "instruccional"
    target.mkdir()
    (target / "sample.mkv").write_text("x")

    # 10 GB free
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=10 * 1024**3),
    )

    # All executables on PATH
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    # nvidia-smi ok
    class _Result:
        returncode = 0
        stdout = b"GPU OK"
        stderr = b""

    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *a, **kw: _Result(),
    )

    # Backend health => 200
    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok"}, request=req)

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _MockClient)
    return target


# ---------------------------------------------------------------------------
# Unit tests on individual checks
# ---------------------------------------------------------------------------


def test_check_path_missing(tmp_path: Path) -> None:
    r = preflight.check_path(str(tmp_path / "does-not-exist"))
    assert r.ok is False
    assert "no existe" in r.message


def test_check_path_ok(tmp_path: Path) -> None:
    r = preflight.check_path(str(tmp_path))
    assert r.ok is True


def test_check_path_empty() -> None:
    r = preflight.check_path("")
    assert r.ok is False


def test_check_disk_space_insufficient(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=1 * 1024**3),
    )
    r = preflight.check_disk_space(str(tmp_path))
    assert r.ok is False
    assert "insuficiente" in r.message


def test_check_disk_space_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )
    r = preflight.check_disk_space(str(tmp_path))
    assert r.ok is True


def test_check_executable_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda _n: None)
    r = preflight.check_executable("ffmpeg")
    assert r.ok is False


def test_check_executable_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    r = preflight.check_executable("ffmpeg")
    assert r.ok is True


@pytest.mark.asyncio
async def test_check_nvidia_smi_missing_falls_back_to_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """processor-api no tiene nvidia-smi local. Si ningún backend responde
    con GPUs, el check falla con un mensaje explicativo (no 'no está en PATH')."""
    monkeypatch.setattr(preflight.shutil, "which", lambda _n: None)

    class _FailClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, _url):
            raise preflight.httpx.ConnectError("nope")

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _FailClient)
    r = await preflight.check_nvidia_smi()
    assert r.ok is False
    assert "Ningún backend" in r.message


@pytest.mark.asyncio
async def test_check_nvidia_smi_local_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    class _Result:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: _Result())
    r = await preflight.check_nvidia_smi()
    assert r.ok is True
    assert "local" in r.message


@pytest.mark.asyncio
async def test_check_nvidia_smi_local_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    class _Result:
        returncode = 1
        stdout = b""
        stderr = b"err"

    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: _Result())
    r = await preflight.check_nvidia_smi()
    assert r.ok is False


@pytest.mark.asyncio
async def test_check_nvidia_smi_remote_reports_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cuando processor-api no tiene nvidia-smi pero un backend GPU sí,
    el fallback HTTP debe reportarlo correctamente."""
    monkeypatch.setattr(preflight.shutil, "which", lambda _n: None)

    class _OkClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, _url):
            class _R:
                status_code = 200
                def json(self):
                    return {"gpus": [{"name": "NVIDIA RTX 3090"}]}
            return _R()

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _OkClient)
    r = await preflight.check_nvidia_smi()
    assert r.ok is True
    assert "RTX 3090" in r.message


@pytest.mark.asyncio
async def test_check_backend_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok"}, request=req)

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _MockClient)
    r = await preflight.check_backend("splitter", "http://x:8001")
    assert r.ok is True


@pytest.mark.asyncio
async def test_check_backend_down(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _MockClient)
    r = await preflight.check_backend("splitter", "http://x:8001")
    assert r.ok is False


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def test_preflight_endpoint_all_ok(client: TestClient, happy_env: Path) -> None:
    resp = client.get("/api/pipeline/preflight", params={"path": str(happy_env)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["all_ok"] is True
    names = {c["name"] for c in body["checks"]}
    assert {"path", "disk_space", "ffmpeg", "mkvtoolnix", "nvidia-smi",
            "splitter", "subs", "dubbing"} <= names
    for c in body["checks"]:
        assert c["ok"] is True


def test_preflight_endpoint_failures(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # ffmpeg missing, disk ok, nvidia missing, backends down
    def _which(name: str):
        return None  # nothing in PATH

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(preflight.shutil, "which", _which)
    monkeypatch.setattr(
        preflight.shutil, "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )

    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _MockClient)

    resp = client.get(
        "/api/pipeline/preflight",
        params={"path": str(tmp_path / "missing")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["all_ok"] is False
    by_name = {c["name"]: c for c in body["checks"]}
    assert by_name["path"]["ok"] is False
    assert by_name["ffmpeg"]["ok"] is False
    assert by_name["nvidia-smi"]["ok"] is False
    assert by_name["splitter"]["ok"] is False


# ---------------------------------------------------------------------------
# Cache + lock + paralelismo (fix 2 y 4 del diagnóstico 2026-04-13)
# ---------------------------------------------------------------------------


def _install_counting_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Instala mocks que cuentan cuántas rondas de checks se hacen.

    Devuelve un dict con contadores compartidos para que los tests puedan
    assertar sobre ellos.
    """
    target = tmp_path / "instruccional"
    target.mkdir()

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight.shutil, "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")

    class _Result:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: _Result())

    counters = {"http_calls": 0, "target": str(target)}

    class _CountingClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            counters["http_calls"] += 1
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok", "gpus": []}, request=req)

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _CountingClient)
    return counters


def test_cache_hits_within_ttl(client: TestClient, monkeypatch, tmp_path) -> None:
    """2 peticiones dentro del TTL → una sola ronda de checks."""
    counters = _install_counting_env(monkeypatch, tmp_path)
    path = counters["target"]

    r1 = client.get("/api/pipeline/preflight", params={"path": path})
    calls_after_first = counters["http_calls"]
    assert r1.status_code == 200
    assert calls_after_first > 0

    r2 = client.get("/api/pipeline/preflight", params={"path": path})
    assert r2.status_code == 200
    # No nuevas llamadas HTTP — servido desde cache
    assert counters["http_calls"] == calls_after_first
    assert r1.json() == r2.json()


def test_cache_key_per_path(client: TestClient, monkeypatch, tmp_path) -> None:
    """Paths distintos NO comparten entrada de cache."""
    counters = _install_counting_env(monkeypatch, tmp_path)
    path_a = counters["target"]
    path_b = str(tmp_path / "otro")
    Path(path_b).mkdir()

    client.get("/api/pipeline/preflight", params={"path": path_a})
    calls_a = counters["http_calls"]
    assert calls_a > 0

    # path distinto → debe relanzar los checks (más llamadas HTTP)
    client.get("/api/pipeline/preflight", params={"path": path_b})
    assert counters["http_calls"] > calls_a


@pytest.mark.asyncio
async def test_lock_prevents_thundering_herd(monkeypatch, tmp_path) -> None:
    """2 peticiones simultáneas para el mismo path → una única ronda de checks."""
    target = tmp_path / "inst"
    target.mkdir()

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight.shutil, "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")

    class _Result:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: _Result())

    counters = {"http_calls": 0}

    class _SlowClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            counters["http_calls"] += 1
            # Dormir un poco para que las corrutinas se solapen
            await asyncio.sleep(0.05)
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok", "gpus": []}, request=req)

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _SlowClient)

    # Lanzar 2 simultáneamente
    import asyncio as _a
    results = await _a.gather(
        preflight.get_preflight_cached(str(target)),
        preflight.get_preflight_cached(str(target)),
    )
    assert results[0] == results[1]
    # Sólo UNA ronda de HTTP calls — el lock impidió que ambas ejecutaran checks
    # Cada ronda hace 6 calls (3 /gpu + 3 /health). Si el lock fallase, serían 12.
    assert counters["http_calls"] <= 6


@pytest.mark.asyncio
async def test_run_all_checks_runs_in_parallel(monkeypatch, tmp_path) -> None:
    """El tiempo total debe ser < 2× el slowest single check (evidencia gather)."""
    target = tmp_path / "inst"
    target.mkdir()

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight.shutil, "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")

    class _Result:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: _Result())

    SLOW = 0.1  # 100 ms por HTTP call

    class _UniformSlowClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            await asyncio.sleep(SLOW)
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok", "gpus": []}, request=req)

    monkeypatch.setattr(preflight.httpx, "AsyncClient", _UniformSlowClient)

    import time as _t
    t0 = _t.monotonic()
    await preflight.run_all_checks(str(target))
    elapsed = _t.monotonic() - t0
    # Hay 3 /gpu + 3 /health = 6 HTTP calls. Secuencial: 6*SLOW = 600ms.
    # En paralelo: ~SLOW (+overhead). Aceptamos < 2*SLOW como evidencia.
    assert elapsed < 2 * SLOW, (
        f"run_all_checks tardó {elapsed:.3f}s, esperado < {2*SLOW}s "
        f"(SLOW={SLOW}). Los checks no se están ejecutando en paralelo."
    )
