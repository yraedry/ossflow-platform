"""Tests del MetricsService."""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.metrics import metrics_router
from ossflow_api.modules.metrics import service as metrics_service
from ossflow_api.modules.metrics.dependencies import get_metrics_service
from ossflow_api.modules.metrics.service import MetricsService


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


@pytest.fixture
def fake_psutil(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    yield


@pytest.fixture
def make_client():
    """Devuelve un (TestClient, MetricsService) con backends mockeables."""

    def _factory(*, gpu_backends=None) -> tuple[TestClient, MetricsService]:
        svc = MetricsService(
            gpu_backends=gpu_backends or list(metrics_service._DEFAULT_GPU_BACKENDS),
            load_settings=lambda: {"library_path": ""},
        )
        app = FastAPI()
        app.include_router(metrics_router)
        app.dependency_overrides[get_metrics_service] = lambda: svc
        return TestClient(app), svc

    return _factory


def _fake_run_ok(*args, **kwargs):
    stdout = (
        "NVIDIA GeForce RTX 3090, 55, 4096, 24576, 62\n"
        "NVIDIA GeForce RTX 3060, 10, 1024, 12288, 48\n"
    )
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_metrics_with_gpu(fake_psutil, make_client, monkeypatch):
    monkeypatch.setattr(metrics_service.shutil, "which", lambda x: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(metrics_service.subprocess, "run", _fake_run_ok)

    client, _ = make_client()
    resp = client.get("/api/metrics/")

    assert resp.status_code == 200
    data = resp.json()
    assert data["cpu_percent"] == 42.5
    assert data["ram"]["total_gb"] == 16.0
    assert data["ram"]["used_gb"] == 6.0
    assert len(data["gpus"]) == 2
    assert data["gpus"][0]["name"] == "NVIDIA GeForce RTX 3090"
    for key in ("cpu_percent", "cpu_temp_c", "ram", "disk", "disks", "gpus", "ram_note"):
        assert key in data


def test_metrics_without_gpu(fake_psutil, make_client, monkeypatch):
    monkeypatch.setattr(metrics_service.shutil, "which", lambda x: None)

    def _boom(*a, **kw):
        raise AssertionError("subprocess.run should not be invoked")

    monkeypatch.setattr(metrics_service.subprocess, "run", _boom)

    client, svc = make_client()

    async def _fake_get(url, timeout=None):
        return SimpleNamespace(status_code=200, json=lambda: {"gpus": []})

    svc._http_client = SimpleNamespace(get=_fake_get)  # type: ignore[assignment]

    resp = client.get("/api/metrics/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["gpus"] == []
    assert data["cpu_percent"] == 42.5


def test_metrics_gpu_subprocess_failure(fake_psutil, make_client, monkeypatch):
    monkeypatch.setattr(metrics_service.shutil, "which", lambda x: "/usr/bin/nvidia-smi")

    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2)

    monkeypatch.setattr(metrics_service.subprocess, "run", _raise)

    client, svc = make_client()

    async def _boom_get(url, timeout=None):
        raise RuntimeError("backend down")

    svc._http_client = SimpleNamespace(get=_boom_get)  # type: ignore[assignment]

    resp = client.get("/api/metrics/")
    assert resp.status_code == 200
    assert resp.json()["gpus"] == []


@pytest.mark.asyncio
async def test_ttl_cache_deduplicates_consecutive_calls(fake_psutil, monkeypatch):
    """Dos llamadas en <5s deben disparar UN solo fan-out."""
    monkeypatch.setattr(metrics_service.shutil, "which", lambda x: None)

    call_count = {"n": 0}

    async def _fake_get(url, timeout=None):
        call_count["n"] += 1
        return SimpleNamespace(status_code=200, json=lambda: {"gpus": []})

    svc = MetricsService(load_settings=lambda: {"library_path": ""})
    svc._http_client = SimpleNamespace(get=_fake_get)  # type: ignore[assignment]

    await svc.snapshot()
    first = call_count["n"]
    assert first == len(svc.gpu_backends)

    await svc.snapshot()
    assert call_count["n"] == first


@pytest.mark.asyncio
async def test_cache_lock_prevents_race(fake_psutil, monkeypatch):
    monkeypatch.setattr(metrics_service.shutil, "which", lambda x: None)

    fanout_count = {"n": 0}

    async def _fake_get(url, timeout=None):
        await asyncio.sleep(0.02)
        fanout_count["n"] += 1
        return SimpleNamespace(status_code=200, json=lambda: {"gpus": []})

    svc = MetricsService(load_settings=lambda: {"library_path": ""})
    svc._http_client = SimpleNamespace(get=_fake_get)  # type: ignore[assignment]

    results = await asyncio.gather(*[svc.snapshot() for _ in range(10)])

    assert len(results) == 10
    assert fanout_count["n"] == len(svc.gpu_backends)
    for r in results[1:]:
        assert r is results[0]


@pytest.mark.asyncio
async def test_gpus_uses_first_nonempty_backend(fake_psutil, monkeypatch):
    monkeypatch.setattr(metrics_service.shutil, "which", lambda x: None)

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

    svc = MetricsService(load_settings=lambda: {"library_path": ""})
    svc._http_client = SimpleNamespace(get=_fake_get)  # type: ignore[assignment]

    data = await svc.snapshot()
    assert data["gpus"] == [sample_gpu]
