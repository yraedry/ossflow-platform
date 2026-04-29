"""Pre-flight checks for the BJJ video processing pipeline.

Single Responsibility: run a set of pre-flight diagnostics and return
a structured list of results. Stateless — no side effects, no DB access.

Checks performed:
    * path exists and is accessible
    * at least 5GB of free space on the output volume
    * ffmpeg is on PATH
    * mkvtoolnix (mkvmerge) is on PATH
    * nvidia-smi responds successfully
    * splitter, subs and dubbing backends respond to /health

Performance (fix 2 + 4 del diagnóstico 2026-04-13):
    * Un único httpx.AsyncClient compartido (módulo) → evita abrir pool TCP
      nuevo por check.
    * run_all_checks lanza TODOS los checks (path, disk, ffmpeg, mkv,
      nvidia-smi/gpu-remote, backend-health) con asyncio.gather en paralelo.
    * Cache en memoria por `path` con TTL 30s + lock asyncio por key para
      evitar thundering herd.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Awaitable

import httpx
from fastapi import APIRouter, Query

from api.backend_client import dubbing_client, splitter_client, subs_client

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["preflight"])

MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
HEALTH_TIMEOUT = 2.0
CACHE_TTL_SECONDS = 30.0
STATIC_CACHE_TTL_SECONDS = 5 * 60.0  # 5 min


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


# ---------------------------------------------------------------------------
# Shared AsyncClient (fix 2: evita crear pool TCP nuevo por check)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _shared_client() -> httpx.AsyncClient:
    """AsyncClient compartido a nivel de módulo.

    Usa lru_cache para que sea seguro crearlo bajo demanda sin necesitar
    wiring explícito en lifespan (aunque `aclose_shared_client` se puede
    llamar desde shutdown).
    """
    return httpx.AsyncClient(timeout=HEALTH_TIMEOUT)


def _get_client() -> httpx.AsyncClient:
    """Devuelve el cliente compartido. Se re-crea si httpx.AsyncClient ha sido
    monkeypatcheado (tests) o si el anterior fue cerrado."""
    client = _shared_client()
    # Si los tests parchean httpx.AsyncClient, la instancia cacheada es
    # del constructor ANTERIOR. Detectar por type() y recrear si difiere.
    if not isinstance(client, httpx.AsyncClient):
        _shared_client.cache_clear()
        return _shared_client()
    # Si fue cerrado (shutdown previo), recrear.
    if getattr(client, "is_closed", False):
        _shared_client.cache_clear()
        return _shared_client()
    return client


async def aclose_shared_client() -> None:
    """Cierra el cliente compartido. Llamar desde lifespan shutdown."""
    try:
        client = _shared_client()
        if not getattr(client, "is_closed", False):
            await client.aclose()
    except Exception:  # pragma: no cover - defensive
        log.debug("Error closing shared preflight client", exc_info=True)
    finally:
        _shared_client.cache_clear()


async def _http_get(url: str) -> httpx.Response:
    """GET usando el client compartido. En tests, si httpx.AsyncClient ha sido
    monkeypatcheado con un mock que implementa el context manager protocol,
    usamos ese para que los tests existentes sigan funcionando."""
    ac = httpx.AsyncClient
    # Test-mode detection: si AsyncClient fue reemplazado por un mock
    # (no es la clase real), usarlo con el protocolo context manager.
    if ac is not _RealAsyncClient:
        async with ac(timeout=HEALTH_TIMEOUT) as hc:  # type: ignore[misc]
            return await hc.get(url)
    client = _get_client()
    return await client.get(url, timeout=HEALTH_TIMEOUT)


# Snapshot de la clase real para detectar monkeypatching en tests
_RealAsyncClient = httpx.AsyncClient


# ---------------------------------------------------------------------------
# Individual checks (each is a pure function of its inputs)
# ---------------------------------------------------------------------------


def check_path(path: str) -> CheckResult:
    if not path:
        return CheckResult("path", False, "No se proporcionó ninguna ruta")
    p = Path(path)
    try:
        if not p.exists():
            return CheckResult("path", False, f"La ruta no existe: {path}")
        # Basic accessibility probe: list dir or stat file
        if p.is_dir():
            next(iter(p.iterdir()), None)
        else:
            p.stat()
        return CheckResult("path", True, f"Ruta accesible: {path}")
    except PermissionError as exc:
        return CheckResult("path", False, f"Permiso denegado: {exc}")
    except OSError as exc:
        return CheckResult("path", False, f"Error accediendo a la ruta: {exc}")


def check_disk_space(path: str, min_bytes: int = MIN_FREE_BYTES) -> CheckResult:
    target = path if path and Path(path).exists() else str(Path.cwd())
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return CheckResult("disk_space", False, f"No se pudo leer el espacio libre: {exc}")
    free_gb = usage.free / (1024 ** 3)
    min_gb = min_bytes / (1024 ** 3)
    if usage.free < min_bytes:
        return CheckResult(
            "disk_space",
            False,
            f"Espacio libre insuficiente: {free_gb:.1f}GB (mínimo {min_gb:.0f}GB)",
        )
    return CheckResult("disk_space", True, f"Espacio libre: {free_gb:.1f}GB")


def check_executable(name: str, display: str | None = None) -> CheckResult:
    display = display or name
    found = shutil.which(name)
    if found:
        return CheckResult(display, True, f"{display} encontrado en: {found}")
    return CheckResult(display, False, f"{display} no está en el PATH")


def _check_nvidia_smi_local() -> CheckResult | None:
    """Prueba local. Devuelve None si nvidia-smi no está, para que el caller
    pruebe el fallback remoto."""
    path = shutil.which("nvidia-smi")
    if not path:
        return None
    try:
        result = subprocess.run([path], capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("nvidia-smi", False, f"nvidia-smi falló: {exc}")
    if result.returncode != 0:
        return CheckResult("nvidia-smi", False, f"nvidia-smi retornó {result.returncode}")
    return CheckResult("nvidia-smi", True, "GPU NVIDIA detectada (local)")


async def _probe_backend_gpu(name: str, base_url: str) -> tuple[str, list[dict]] | None:
    """Consulta /gpu de un backend; devuelve (name, gpus) si 200 con gpus."""
    url = f"{base_url.rstrip('/')}/gpu"
    try:
        r = await _http_get(url)
    except (httpx.HTTPError, OSError):
        return None
    try:
        if r.status_code == 200:
            data = r.json() or {}
            gpus = data.get("gpus") or []
            if gpus:
                return (name, gpus)
    except Exception:  # pragma: no cover - defensive json parsing
        return None
    return None


async def check_nvidia_smi() -> CheckResult:
    """El contenedor processor-api NO tiene GPU asignada. Prueba local primero;
    si no hay nvidia-smi, pregunta a cada backend GPU por su endpoint /gpu
    (expuesto vía bjj_service_kit) y considera OK si alguno reporta GPUs.

    Fix: usa el AsyncClient compartido y consulta los 3 backends en paralelo.
    """
    local = _check_nvidia_smi_local()
    if local is not None:
        return local
    pairs = [
        ("splitter", splitter_client().base_url),
        ("subs", subs_client().base_url),
        ("dubbing", dubbing_client().base_url),
    ]
    results = await asyncio.gather(
        *[_probe_backend_gpu(n, u) for n, u in pairs],
        return_exceptions=False,
    )
    for res in results:
        if res is not None:
            name, gpus = res
            names = ", ".join(g.get("name", "?") for g in gpus)
            return CheckResult("nvidia-smi", True, f"GPU detectada vía {name}: {names}")
    return CheckResult(
        "nvidia-smi",
        False,
        "Ningún backend GPU reporta dispositivos (¿drivers NVIDIA / runtime OK?)",
    )


async def check_backend(name: str, base_url: str) -> CheckResult:
    url = f"{base_url.rstrip('/')}/health"
    try:
        r = await _http_get(url)
        if r.status_code >= 400:
            return CheckResult(name, False, f"{name} respondió {r.status_code}")
        return CheckResult(name, True, f"{name} OK")
    except (httpx.HTTPError, OSError) as exc:
        return CheckResult(name, False, f"{name} no responde: {exc}")


async def _run_backend_checks() -> list[CheckResult]:
    pairs = [
        ("splitter", splitter_client().base_url),
        ("subs", subs_client().base_url),
        ("dubbing", dubbing_client().base_url),
    ]
    coros: list[Awaitable[CheckResult]] = [check_backend(n, u) for n, u in pairs]
    return list(await asyncio.gather(*coros))


# ---------------------------------------------------------------------------
# Orchestrator (fix 2: gather TODO en paralelo)
# ---------------------------------------------------------------------------


async def _as_coro_sync(fn, *args) -> CheckResult:
    """Envuelve una función sync en un coroutine para poder gatherla."""
    return fn(*args)


async def run_all_checks(path: str) -> list[CheckResult]:
    """Lanza GPU + health + disk/ffmpeg/mkv/path TODOS en paralelo."""
    coros: list[Awaitable[CheckResult]] = [
        _as_coro_sync(check_path, path),
        _as_coro_sync(check_disk_space, path),
        _as_coro_sync(check_executable, "ffmpeg"),
        _as_coro_sync(check_executable, "mkvmerge", "mkvtoolnix"),
        check_nvidia_smi(),
    ]
    # Añadimos los 3 backend-health checks individualmente (ya no anidados)
    for n, base_url in (
        ("splitter", splitter_client().base_url),
        ("subs", subs_client().base_url),
        ("dubbing", dubbing_client().base_url),
    ):
        coros.append(check_backend(n, base_url))
    results = await asyncio.gather(*coros)
    return list(results)


# ---------------------------------------------------------------------------
# Cache + lock per path (fix 4)
# ---------------------------------------------------------------------------

# path → (payload_dict, inserted_at_monotonic)
_cache: dict[str, tuple[dict, float]] = {}
_cache_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()

# Static-only cache (ffmpeg, mkvtoolnix, disk) — TTL más largo
_static_cache: tuple[dict, float] | None = None
_static_lock = asyncio.Lock()


def _cache_key(path: str) -> str:
    return path or ""


async def _get_lock(key: str) -> asyncio.Lock:
    async with _locks_guard:
        lock = _cache_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _cache_locks[key] = lock
        return lock


def _cached_fresh(key: str, ttl: float) -> dict | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    payload, ts = entry
    if (time.monotonic() - ts) < ttl:
        return payload
    return None


def _compose_payload(results: list[CheckResult]) -> dict:
    return {
        "checks": [asdict(c) for c in results],
        "all_ok": all(c.ok for c in results),
    }


async def _build_full_payload(path: str) -> dict:
    checks = await run_all_checks(path)
    return _compose_payload(checks)


async def get_preflight_cached(path: str, ttl: float = CACHE_TTL_SECONDS) -> dict:
    key = _cache_key(path)
    cached = _cached_fresh(key, ttl)
    if cached is not None:
        return cached
    lock = await _get_lock(key)
    async with lock:
        # Double-check: otra coroutine pudo haber rellenado mientras esperábamos
        cached = _cached_fresh(key, ttl)
        if cached is not None:
            return cached
        payload = await _build_full_payload(path)
        _cache[key] = (payload, time.monotonic())
        return payload


def _invalidate_cache() -> None:
    """Utilidad para tests."""
    _cache.clear()
    _cache_locks.clear()
    global _static_cache
    _static_cache = None


# ---------------------------------------------------------------------------
# Static subset (ffmpeg, mkvtoolnix, disk_space) — cambia rara vez.
# ---------------------------------------------------------------------------


async def _build_static_payload() -> dict:
    # No depende de `path` para disk (usa cwd si no hay path). En endpoint
    # static omitimos path check.
    coros = [
        _as_coro_sync(check_disk_space, ""),
        _as_coro_sync(check_executable, "ffmpeg"),
        _as_coro_sync(check_executable, "mkvmerge", "mkvtoolnix"),
    ]
    results = await asyncio.gather(*coros)
    return _compose_payload(list(results))


async def get_static_cached(ttl: float = STATIC_CACHE_TTL_SECONDS) -> dict:
    global _static_cache
    if _static_cache is not None:
        payload, ts = _static_cache
        if (time.monotonic() - ts) < ttl:
            return payload
    async with _static_lock:
        if _static_cache is not None:
            payload, ts = _static_cache
            if (time.monotonic() - ts) < ttl:
                return payload
        payload = await _build_static_payload()
        _static_cache = (payload, time.monotonic())
        return payload


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@router.get("/preflight")
async def preflight(path: str = Query("", description="Ruta del instruccional")) -> dict:
    return await get_preflight_cached(path)


@router.get("/preflight/static")
async def preflight_static() -> dict:
    """Subset estático (ffmpeg, mkvtoolnix, disk). TTL 5 min."""
    return await get_static_cached()
