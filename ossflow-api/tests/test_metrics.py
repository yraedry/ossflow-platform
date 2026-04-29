"""Tests for api.metrics endpoint."""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import metrics as metrics_mod


class _FakePsutil:
    @staticmethod
    def cpu_percent(interval=None):
        return 42.5

    @staticmethod
    def virtual_memory():
        total = 16 * 1024 ** 3
        available = 10 * 1024 ** 3
        return SimpleNamespace(total=total, available=available, percent=37.5)

    @staticmethod
    def disk_usage(path):
        total = 1000 * 1024 ** 3
        used = 400 * 1024 ** 3
        free = 600 * 1024 ** 3
        return SimpleNamespace(total=total, used=used, free=free, percent=40.0)


@pytest.fixture(autouse=True)
def _reset_metrics_cache():
    metrics_mod._reset_cache_for_tests()
    yield
    metrics_mod._reset_cache_for_tests()


@pytest.fixture
def client(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    monkeypatch.setattr(metrics_mod, "load_settings", lambda: {"library_path": ""})

    app = FastAPI()
    app.include_router(metrics_mod.router)
    return TestClient(app)


def _fake_run_ok(*args, **kwargs):
    stdout = (
        "NVIDIA GeForce RTX 3090, 55, 4096, 24576, 62\n"
        "NVIDIA GeForce RTX 3060, 10, 1024, 12288, 48\n"
    )
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_metrics_with_gpu(client, monkeypatch):
    monkeypatch.setattr(metrics_mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(metrics_mod.subprocess, "run", _fake_run_ok)

    resp = client.get("/api/metrics/")
    assert resp.status_code == 200
    data = resp.json()

    assert data["cpu_percent"] == 42.5
    assert data["ram"]["total_gb"] == 16.0
    assert data["ram"]["used_gb"] == 6.0
    assert data["ram"]["percent"] == 37.5

    assert data["disk"]["total_gb"] == 1000.0
    assert data["disk"]["used_gb"] == 400.0
    assert data["disk"]["free_gb"] == 600.0

    assert len(data["gpus"]) == 2
    g0 = data["gpus"][0]
    assert g0["name"] == "NVIDIA GeForce RTX 3090"
    assert g0["util_percent"] == 55.0
    assert g0["mem_used_mb"] == 4096.0
    assert g0["mem_total_mb"] == 24576.0
    assert g0["temp_c"] == 62.0

    # Shape backwards-compat: claves obligatorias del payload.
    for key in ("cpu_percent", "cpu_temp_c", "ram", "disk", "disks", "gpus", "ram_note"):
        assert key in data


def test_metrics_without_gpu(client, monkeypatch):
    monkeypatch.setattr(metrics_mod.shutil, "which", lambda x: None)

    def _boom(*a, **kw):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run should not be invoked")

    monkeypatch.setattr(metrics_mod.subprocess, "run", _boom)

    # Sin GPU local -> intenta backends. Mockeamos el cliente httpx para que
    # respondan sin GPUs, así el fallback devuelve [].
    async def _fake_get(url, timeout=None):
        return SimpleNamespace(status_code=200, json=lambda: {"gpus": []})

    fake_client = SimpleNamespace(get=_fake_get)
    monkeypatch.setattr(metrics_mod, "_get_http_client", lambda: fake_client)

    resp = client.get("/api/metrics/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["gpus"] == []
    assert data["cpu_percent"] == 42.5
    assert "total_gb" in data["ram"]


def test_metrics_gpu_subprocess_failure(client, monkeypatch):
    monkeypatch.setattr(metrics_mod.shutil, "which", lambda x: "/usr/bin/nvidia-smi")

    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2)

    monkeypatch.setattr(metrics_mod.subprocess, "run", _raise)

    # Backends también fallan -> gpus == [].
    async def _fake_get(url, timeout=None):
        raise httpx_timeout_like()

    class httpx_timeout_like(Exception):
        pass

    async def _boom_get(url, timeout=None):
        raise RuntimeError("backend down")

    fake_client = SimpleNamespace(get=_boom_get)
    monkeypatch.setattr(metrics_mod, "_get_http_client", lambda: fake_client)

    resp = client.get("/api/metrics/")
    assert resp.status_code == 200
    assert resp.json()["gpus"] == []


# --- P1-F2: TTL cache + asyncio.gather -----------------------------------


@pytest.mark.asyncio
async def test_ttl_cache_deduplicates_consecutive_calls(monkeypatch):
    """Dos llamadas consecutivas en <5s deben disparar UN solo fan-out HTTP."""
    import sys
    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    monkeypatch.setattr(metrics_mod, "load_settings", lambda: {"library_path": ""})
    monkeypatch.setattr(metrics_mod.shutil, "which", lambda x: None)

    call_count = {"n": 0}

    async def _fake_get(url, timeout=None):
        call_count["n"] += 1
        return SimpleNamespace(status_code=200, json=lambda: {"gpus": []})

    fake_client = SimpleNamespace(get=_fake_get)
    monkeypatch.setattr(metrics_mod, "_get_http_client", lambda: fake_client)

    # Primera llamada: 3 backends -> 3 GETs.
    await metrics_mod.get_metrics()
    first = call_count["n"]
    assert first == len(metrics_mod._GPU_BACKEND_URLS)

    # Segunda llamada inmediata: cache hit, no más GETs.
    await metrics_mod.get_metrics()
    assert call_count["n"] == first, "cache TTL debería evitar el segundo fan-out"


@pytest.mark.asyncio
async def test_cache_lock_prevents_race(monkeypatch):
    """N requests concurrentes tras cache expirada solo refrescan UNA vez."""
    import sys
    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    monkeypatch.setattr(metrics_mod, "load_settings", lambda: {"library_path": ""})
    monkeypatch.setattr(metrics_mod.shutil, "which", lambda x: None)

    fanout_count = {"n": 0}

    async def _fake_get(url, timeout=None):
        # Simula latencia para maximizar la ventana de race.
        await asyncio.sleep(0.02)
        fanout_count["n"] += 1
        return SimpleNamespace(status_code=200, json=lambda: {"gpus": []})

    fake_client = SimpleNamespace(get=_fake_get)
    monkeypatch.setattr(metrics_mod, "_get_http_client", lambda: fake_client)

    # 10 requests concurrentes, cache vacía.
    results = await asyncio.gather(*[metrics_mod.get_metrics() for _ in range(10)])

    assert len(results) == 10
    # Exactamente UN refresh (= N backends GETs, no 10*N).
    assert fanout_count["n"] == len(metrics_mod._GPU_BACKEND_URLS), (
        f"expected single refresh ({len(metrics_mod._GPU_BACKEND_URLS)} GETs), "
        f"got {fanout_count['n']}"
    )
    # Todas las respuestas comparten el mismo dict cacheado.
    for r in results[1:]:
        assert r is results[0]


@pytest.mark.asyncio
async def test_gpus_uses_first_nonempty_backend(monkeypatch):
    """asyncio.gather: si el primer backend devuelve gpus, se usa ese."""
    import sys
    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    monkeypatch.setattr(metrics_mod, "load_settings", lambda: {"library_path": ""})
    monkeypatch.setattr(metrics_mod.shutil, "which", lambda x: None)

    sample_gpu = {
        "name": "RTX 4090",
        "util_percent": 80.0,
        "mem_used_mb": 10000.0,
        "mem_total_mb": 24000.0,
        "temp_c": 70.0,
    }

    async def _fake_get(url, timeout=None):
        if "8001" in url:
            return SimpleNamespace(status_code=200, json=lambda: {"gpus": [sample_gpu]})
        return SimpleNamespace(status_code=200, json=lambda: {"gpus": []})

    fake_client = SimpleNamespace(get=_fake_get)
    monkeypatch.setattr(metrics_mod, "_get_http_client", lambda: fake_client)

    data = await metrics_mod.get_metrics()
    assert data["gpus"] == [sample_gpu]
