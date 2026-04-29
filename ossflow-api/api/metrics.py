"""System metrics endpoint: CPU, RAM, disk and GPU stats.

Exposes ``GET /api/metrics/`` returning a snapshot dict suitable for a
live dashboard widget. GPU data is collected via ``nvidia-smi`` when
available; if the binary is missing or fails the endpoint still returns
CPU/RAM/disk with ``gpus: []``.

P1-F2: async + asyncio.gather + TTL cache. The endpoint is ``async def``
so the event loop is not blocked while polling the three GPU backends in
parallel. Results are cached in-memory for 5 seconds behind an asyncio
lock with double-check so concurrent requests trigger only one refresh.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter

from api.settings import load_settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

# Backends GPU — consultados en paralelo con asyncio.gather. Todos montan
# la misma GPU física (NVIDIA_VISIBLE_DEVICES=all), así que nos vale el
# primer backend que responda con `gpus` no vacío.
_GPU_BACKEND_URLS = [
    os.environ.get("SPLITTER_URL", "http://chapter-splitter:8001"),
    os.environ.get("SUBS_URL", "http://subtitle-generator:8002"),
    os.environ.get("DUBBING_URL", "http://dubbing-generator:8003"),
]

# --- TTL cache -------------------------------------------------------------
# 5 s es el intervalo de polling del dashboard; así N clientes concurrentes
# solo provocan un fan-out a los backends cada 5 s.
_CACHE_TTL_SECONDS = 5.0
_cache: dict[str, Any] = {"data": None, "expires_at": 0.0}
_cache_lock = asyncio.Lock()

# Cliente httpx compartido — reutiliza conexiones (keep-alive) y evita el
# coste de un handshake TCP por request.
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(1.5))
    return _http_client


def _bytes_to_gb(n: int | float) -> float:
    return round(float(n) / (1024 ** 3), 2)


def _cpu_temp_c() -> float | None:
    """Best-effort CPU temp. Devuelve None si el kernel no la expone
    (caso habitual en Docker Desktop/WSL2 — la VM no ve los thermal_zones)."""
    import psutil
    try:
        sensors = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None
    if not sensors:
        return None
    for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
        if key in sensors and sensors[key]:
            return float(sensors[key][0].current)
    for entries in sensors.values():
        if entries:
            return float(entries[0].current)
    return None


def _collect_cpu_ram() -> tuple[float, dict[str, float], float | None]:
    import psutil  # imported lazily so tests can monkeypatch

    cpu = float(psutil.cpu_percent(interval=None))
    vm = psutil.virtual_memory()
    ram = {
        "used_gb": _bytes_to_gb(vm.total - vm.available),
        "total_gb": _bytes_to_gb(vm.total),
        "percent": float(vm.percent),
    }
    return cpu, ram, _cpu_temp_c()


def _disk_entry(label: str, path: str) -> dict[str, Any] | None:
    import psutil
    try:
        usage = psutil.disk_usage(path)
    except Exception as exc:
        log.debug("disk_usage failed for %s: %s", path, exc)
        return None
    return {
        "label": label,
        "path": path,
        "used_gb": _bytes_to_gb(usage.used),
        "free_gb": _bytes_to_gb(usage.free),
        "total_gb": _bytes_to_gb(usage.total),
        "percent": float(usage.percent),
    }


def _collect_disks() -> list[dict[str, Any]]:
    """Return a list of monitored volumes: container root + library mount."""
    entries: list[dict[str, Any]] = []
    local = _disk_entry("Local", os.path.abspath(os.sep))
    if local:
        entries.append(local)

    try:
        settings = load_settings()
        lib = (settings or {}).get("library_path") or ""
    except Exception:
        lib = ""
    lib_mount = "/media" if Path("/media").exists() else lib
    if lib_mount and lib_mount != local["path"] if local else lib_mount:
        lib_entry = _disk_entry("Biblioteca", lib_mount)
        if lib_entry:
            lib_entry["host_path"] = lib or None
            entries.append(lib_entry)
    return entries


def _collect_gpus_local() -> list[dict[str, Any]]:
    """Local nvidia-smi — usually empty in processor-api (no GPU device)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in (result.stdout or "").splitlines():
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 5:
            continue
        name, util, mem_used, mem_total, temp = parts[:5]
        try:
            gpus.append({
                "name": name,
                "util_percent": float(util),
                "mem_used_mb": float(mem_used),
                "mem_total_mb": float(mem_total),
                "temp_c": float(temp),
            })
        except ValueError:
            continue
    return gpus


async def _collect_gpus() -> list[dict[str, Any]]:
    """Try local nvidia-smi first; si no hay, pregunta a los backends GPU en paralelo.

    Usa ``asyncio.gather(..., return_exceptions=True)`` para no bloquearse si
    un backend está caído. Devuelve la primera lista no vacía de ``gpus``.
    """
    local = _collect_gpus_local()
    if local:
        return local

    client = _get_http_client()
    coros = [client.get(f"{base}/gpu", timeout=1.5) for base in _GPU_BACKEND_URLS]
    results = await asyncio.gather(*coros, return_exceptions=True)

    for base, resp in zip(_GPU_BACKEND_URLS, results):
        if isinstance(resp, BaseException):
            log.debug("GPU backend %s unreachable: %s", base, resp)
            continue
        try:
            if resp.status_code != 200:
                continue
            data = resp.json() or {}
            gpus = data.get("gpus") or []
            if gpus:
                return gpus
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("GPU backend %s bad response: %s", base, exc)
            continue
    return []


async def _build_snapshot() -> dict[str, Any]:
    """Assemble the full metrics payload (uncached)."""
    cpu_percent, ram, cpu_temp = _collect_cpu_ram()
    disks = _collect_disks()
    gpus = await _collect_gpus()
    return {
        "cpu_percent": cpu_percent,
        "cpu_temp_c": cpu_temp,
        "ram": ram,
        # Mantenemos ``disk`` (primer volumen) por compat con el frontend viejo.
        "disk": disks[0] if disks else None,
        "disks": disks,
        "gpus": gpus,
        "ram_note": "container_visible",
    }


@router.get("/")
async def get_metrics() -> dict[str, Any]:
    """Return a snapshot of CPU/RAM/disk/GPU metrics (cached 5s).

    Note: ``ram`` refleja la memoria visible dentro del contenedor. En Docker
    Desktop/WSL2 eso suele ser ~50% del host por defecto; se ajusta en
    `%UserProfile%\\.wslconfig` con `memory=32GB`. ``disks`` lista volúmenes
    monitorizados (local + mount de biblioteca) para no confundir NAS con SSD.
    """
    now = time.monotonic()
    # Fast path — cache válida, sin coger el lock.
    if _cache["data"] is not None and now < _cache["expires_at"]:
        return _cache["data"]

    async with _cache_lock:
        # Double-check: otro coroutine pudo haber refrescado mientras
        # esperábamos el lock.
        now = time.monotonic()
        if _cache["data"] is not None and now < _cache["expires_at"]:
            return _cache["data"]

        snapshot = await _build_snapshot()
        _cache["data"] = snapshot
        _cache["expires_at"] = time.monotonic() + _CACHE_TTL_SECONDS
        return snapshot


def _reset_cache_for_tests() -> None:
    """Testing hook — wipe the TTL cache between tests."""
    _cache["data"] = None
    _cache["expires_at"] = 0.0


# WIRE_ROUTER: from api.metrics import router as metrics_router; app.include_router(metrics_router)
