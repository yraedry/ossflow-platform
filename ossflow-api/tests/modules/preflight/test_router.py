"""Tests del módulo preflight."""

from __future__ import annotations

import asyncio
from collections import namedtuple
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.preflight import preflight_router
from ossflow_api.modules.preflight import service as preflight_service
from ossflow_api.modules.preflight.dependencies import (
    get_preflight_service,
    reset_for_tests,
)
from ossflow_api.modules.preflight.service import PreflightService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_preflight_singleton():
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(preflight_router)
    svc = PreflightService()
    app.dependency_overrides[get_preflight_service] = lambda: svc
    return TestClient(app)


@pytest.fixture
def happy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    target = tmp_path / "instruccional"
    target.mkdir()
    (target / "sample.mkv").write_text("x")

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight_service.shutil,
        "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=10 * 1024**3),
    )
    monkeypatch.setattr(
        preflight_service.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    class _Result:
        returncode = 0
        stdout = b"GPU OK"
        stderr = b""

    monkeypatch.setattr(
        preflight_service.subprocess,
        "run",
        lambda *a, **kw: _Result(),
    )

    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok"}, request=req)

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _MockClient)
    return target


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_check_path_missing(tmp_path: Path) -> None:
    r = preflight_service.check_path(str(tmp_path / "does-not-exist"))
    assert r.ok is False
    assert "no existe" in r.message


def test_check_path_ok(tmp_path: Path) -> None:
    r = preflight_service.check_path(str(tmp_path))
    assert r.ok is True


def test_check_path_empty() -> None:
    r = preflight_service.check_path("")
    assert r.ok is False


def test_check_disk_space_insufficient(monkeypatch, tmp_path: Path) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight_service.shutil,
        "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=1 * 1024**3),
    )
    r = preflight_service.check_disk_space(str(tmp_path))
    assert r.ok is False
    assert "insuficiente" in r.message


def test_check_disk_space_ok(monkeypatch, tmp_path: Path) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight_service.shutil,
        "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )
    r = preflight_service.check_disk_space(str(tmp_path))
    assert r.ok is True


def test_check_executable_missing(monkeypatch) -> None:
    monkeypatch.setattr(preflight_service.shutil, "which", lambda _n: None)
    r = preflight_service.check_executable("ffmpeg")
    assert r.ok is False


def test_check_executable_found(monkeypatch) -> None:
    monkeypatch.setattr(preflight_service.shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    r = preflight_service.check_executable("ffmpeg")
    assert r.ok is True


@pytest.mark.asyncio
async def test_check_nvidia_smi_missing_falls_back_to_remote(monkeypatch) -> None:
    monkeypatch.setattr(preflight_service.shutil, "which", lambda _n: None)

    class _FailClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, _url):
            raise preflight_service.httpx.ConnectError("nope")

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _FailClient)
    r = await PreflightService.check_nvidia_smi()
    assert r.ok is False
    assert "Ningún backend" in r.message


@pytest.mark.asyncio
async def test_check_nvidia_smi_local_ok(monkeypatch) -> None:
    monkeypatch.setattr(preflight_service.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    class _Result:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(preflight_service.subprocess, "run", lambda *a, **kw: _Result())
    r = await PreflightService.check_nvidia_smi()
    assert r.ok is True
    assert "local" in r.message


@pytest.mark.asyncio
async def test_check_nvidia_smi_local_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(preflight_service.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    class _Result:
        returncode = 1
        stdout = b""
        stderr = b"err"

    monkeypatch.setattr(preflight_service.subprocess, "run", lambda *a, **kw: _Result())
    r = await PreflightService.check_nvidia_smi()
    assert r.ok is False


@pytest.mark.asyncio
async def test_check_nvidia_smi_remote_reports_gpu(monkeypatch) -> None:
    monkeypatch.setattr(preflight_service.shutil, "which", lambda _n: None)

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

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _OkClient)
    r = await PreflightService.check_nvidia_smi()
    assert r.ok is True
    assert "RTX 3090" in r.message


@pytest.mark.asyncio
async def test_check_backend_ok(monkeypatch) -> None:
    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok"}, request=req)

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _MockClient)
    r = await PreflightService.check_backend("splitter", "http://x:8001")
    assert r.ok is True


@pytest.mark.asyncio
async def test_check_backend_down(monkeypatch) -> None:
    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _MockClient)
    r = await PreflightService.check_backend("splitter", "http://x:8001")
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
    monkeypatch,
    tmp_path: Path,
) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(preflight_service.shutil, "which", lambda _n: None)
    monkeypatch.setattr(
        preflight_service.shutil, "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )

    class _MockClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _MockClient)

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
# Cache + lock + paralelismo
# ---------------------------------------------------------------------------


def _install_counting_env(monkeypatch, tmp_path: Path) -> dict:
    target = tmp_path / "instruccional"
    target.mkdir()

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight_service.shutil, "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )
    monkeypatch.setattr(preflight_service.shutil, "which", lambda name: f"/usr/bin/{name}")

    class _Result:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(preflight_service.subprocess, "run", lambda *a, **kw: _Result())

    counters = {"http_calls": 0, "target": str(target)}

    class _CountingClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            counters["http_calls"] += 1
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok", "gpus": []}, request=req)

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _CountingClient)
    return counters


def test_cache_hits_within_ttl(client: TestClient, monkeypatch, tmp_path) -> None:
    counters = _install_counting_env(monkeypatch, tmp_path)
    path = counters["target"]

    r1 = client.get("/api/pipeline/preflight", params={"path": path})
    calls_after_first = counters["http_calls"]
    assert r1.status_code == 200
    assert calls_after_first > 0

    r2 = client.get("/api/pipeline/preflight", params={"path": path})
    assert r2.status_code == 200
    assert counters["http_calls"] == calls_after_first
    assert r1.json() == r2.json()


def test_cache_key_per_path(client: TestClient, monkeypatch, tmp_path) -> None:
    counters = _install_counting_env(monkeypatch, tmp_path)
    path_a = counters["target"]
    path_b = str(tmp_path / "otro")
    Path(path_b).mkdir()

    client.get("/api/pipeline/preflight", params={"path": path_a})
    calls_a = counters["http_calls"]
    assert calls_a > 0

    client.get("/api/pipeline/preflight", params={"path": path_b})
    assert counters["http_calls"] > calls_a


@pytest.mark.asyncio
async def test_lock_prevents_thundering_herd(monkeypatch, tmp_path) -> None:
    target = tmp_path / "inst"
    target.mkdir()

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight_service.shutil, "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )
    monkeypatch.setattr(preflight_service.shutil, "which", lambda name: f"/usr/bin/{name}")

    class _Result:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(preflight_service.subprocess, "run", lambda *a, **kw: _Result())

    counters = {"http_calls": 0}

    class _SlowClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            counters["http_calls"] += 1
            await asyncio.sleep(0.05)
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok", "gpus": []}, request=req)

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _SlowClient)

    svc = PreflightService()
    results = await asyncio.gather(
        svc.get_preflight_cached(str(target)),
        svc.get_preflight_cached(str(target)),
    )
    assert results[0] == results[1]
    assert counters["http_calls"] <= 6


@pytest.mark.asyncio
async def test_run_all_checks_runs_in_parallel(monkeypatch, tmp_path) -> None:
    target = tmp_path / "inst"
    target.mkdir()

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        preflight_service.shutil, "disk_usage",
        lambda _p: Usage(total=100 * 1024**3, used=0, free=50 * 1024**3),
    )
    monkeypatch.setattr(preflight_service.shutil, "which", lambda name: f"/usr/bin/{name}")

    class _Result:
        returncode = 0
        stdout = b""
        stderr = b""

    monkeypatch.setattr(preflight_service.subprocess, "run", lambda *a, **kw: _Result())

    SLOW = 0.1

    class _UniformSlowClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            await asyncio.sleep(SLOW)
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok", "gpus": []}, request=req)

    monkeypatch.setattr(preflight_service.httpx, "AsyncClient", _UniformSlowClient)

    import time as _t
    t0 = _t.monotonic()
    await PreflightService.run_all_checks(str(target))
    elapsed = _t.monotonic() - t0
    assert elapsed < 2 * SLOW, (
        f"run_all_checks tardó {elapsed:.3f}s, esperado < {2*SLOW}s "
        f"(SLOW={SLOW}). Los checks no se están ejecutando en paralelo."
    )
