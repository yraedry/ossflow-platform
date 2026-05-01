"""Servicio de preflight: ejecuta checks en paralelo con TTL cache + lock por path.

Estado en instancia (no globales). ``shutil``/``subprocess``/``httpx`` se
exponen como atributos del módulo para que los tests puedan monkeypatchear
``ossflow_api.modules.preflight.service.shutil`` (mismo patrón que tenía
``api/preflight.py``).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Awaitable

import httpx

from ossflow_api.clients.dubbing import dubbing_client
from ossflow_api.clients.splitter import splitter_client
from ossflow_api.clients.subtitle import subs_client

from .schemas import (
    CACHE_TTL_SECONDS,
    CheckResult,
    HEALTH_TIMEOUT,
    MIN_FREE_BYTES,
    STATIC_CACHE_TTL_SECONDS,
)

log = logging.getLogger(__name__)

# Snapshot de la clase real para detectar monkeypatching en tests.
_RealAsyncClient = httpx.AsyncClient


# ---------------------------------------------------------------------------
# Checks individuales (puros respecto a sus inputs)
# ---------------------------------------------------------------------------


def check_path(path: str) -> CheckResult:
    if not path:
        return CheckResult("path", False, "No se proporcionó ninguna ruta")
    p = Path(path)
    try:
        if not p.exists():
            return CheckResult("path", False, f"La ruta no existe: {path}")
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


# ---------------------------------------------------------------------------
# Servicio
# ---------------------------------------------------------------------------


class PreflightService:
    """Orquesta los checks. Mantiene cache + locks por path como atributos."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[dict, float]] = {}
        self._cache_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._static_cache: tuple[dict, float] | None = None
        self._static_lock = asyncio.Lock()

    # --- Cliente HTTP compartido --------------------------------------------

    @staticmethod
    @lru_cache(maxsize=1)
    def _shared_client() -> httpx.AsyncClient:
        """AsyncClient compartido a nivel de proceso.

        Usa ``lru_cache`` para que sea seguro crearlo bajo demanda. ``aclose``
        se llama desde ``infrastructure.lifespan`` (rotura acoplamiento #5).
        """
        return httpx.AsyncClient(timeout=HEALTH_TIMEOUT)

    @classmethod
    def _get_client(cls) -> httpx.AsyncClient:
        client = cls._shared_client()
        if not isinstance(client, httpx.AsyncClient):
            cls._shared_client.cache_clear()
            return cls._shared_client()
        if getattr(client, "is_closed", False):
            cls._shared_client.cache_clear()
            return cls._shared_client()
        return client

    @classmethod
    async def aclose(cls) -> None:
        """Cierra el cliente compartido. Hook registrado en lifespan shutdown."""
        try:
            client = cls._shared_client()
            if not getattr(client, "is_closed", False):
                await client.aclose()
        except Exception:  # pragma: no cover
            log.debug("Error closing shared preflight client", exc_info=True)
        finally:
            cls._shared_client.cache_clear()

    @staticmethod
    async def _http_get(url: str) -> httpx.Response:
        """GET usando el cliente compartido. En tests, si ``httpx.AsyncClient``
        se ha monkeypatcheado con un mock con context manager, lo usamos."""
        ac = httpx.AsyncClient
        if ac is not _RealAsyncClient:
            async with ac(timeout=HEALTH_TIMEOUT) as hc:  # type: ignore[misc]
                return await hc.get(url)
        client = PreflightService._get_client()
        return await client.get(url, timeout=HEALTH_TIMEOUT)

    # --- Checks asíncronos --------------------------------------------------

    @staticmethod
    def _check_nvidia_smi_local() -> CheckResult | None:
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

    @staticmethod
    async def _probe_backend_gpu(name: str, base_url: str) -> tuple[str, list[dict]] | None:
        url = f"{base_url.rstrip('/')}/gpu"
        try:
            r = await PreflightService._http_get(url)
        except (httpx.HTTPError, OSError):
            return None
        try:
            if r.status_code == 200:
                data = r.json() or {}
                gpus = data.get("gpus") or []
                if gpus:
                    return (name, gpus)
        except Exception:  # pragma: no cover
            return None
        return None

    @classmethod
    async def check_nvidia_smi(cls) -> CheckResult:
        local = cls._check_nvidia_smi_local()
        if local is not None:
            return local
        pairs = [
            ("splitter", splitter_client().base_url),
            ("subs", subs_client().base_url),
            ("dubbing", dubbing_client().base_url),
        ]
        results = await asyncio.gather(
            *[cls._probe_backend_gpu(n, u) for n, u in pairs],
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

    @classmethod
    async def check_backend(cls, name: str, base_url: str) -> CheckResult:
        url = f"{base_url.rstrip('/')}/health"
        try:
            r = await cls._http_get(url)
            if r.status_code >= 400:
                return CheckResult(name, False, f"{name} respondió {r.status_code}")
            return CheckResult(name, True, f"{name} OK")
        except (httpx.HTTPError, OSError) as exc:
            return CheckResult(name, False, f"{name} no responde: {exc}")

    # --- Orquestador --------------------------------------------------------

    @staticmethod
    async def _as_coro_sync(fn, *args) -> CheckResult:
        return fn(*args)

    @classmethod
    async def run_all_checks(cls, path: str) -> list[CheckResult]:
        coros: list[Awaitable[CheckResult]] = [
            cls._as_coro_sync(check_path, path),
            cls._as_coro_sync(check_disk_space, path),
            cls._as_coro_sync(check_executable, "ffmpeg"),
            cls._as_coro_sync(check_executable, "mkvmerge", "mkvtoolnix"),
            cls.check_nvidia_smi(),
        ]
        for n, base_url in (
            ("splitter", splitter_client().base_url),
            ("subs", subs_client().base_url),
            ("dubbing", dubbing_client().base_url),
        ):
            coros.append(cls.check_backend(n, base_url))
        results = await asyncio.gather(*coros)
        return list(results)

    # --- Cache + lock por path ---------------------------------------------

    @staticmethod
    def _cache_key(path: str) -> str:
        return path or ""

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._cache_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._cache_locks[key] = lock
            return lock

    def _cached_fresh(self, key: str, ttl: float) -> dict | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        payload, ts = entry
        if (time.monotonic() - ts) < ttl:
            return payload
        return None

    @staticmethod
    def _compose_payload(results: list[CheckResult]) -> dict:
        return {
            "checks": [asdict(c) for c in results],
            "all_ok": all(c.ok for c in results),
        }

    async def _build_full_payload(self, path: str) -> dict:
        checks = await self.run_all_checks(path)
        return self._compose_payload(checks)

    async def get_preflight_cached(self, path: str, ttl: float = CACHE_TTL_SECONDS) -> dict:
        key = self._cache_key(path)
        cached = self._cached_fresh(key, ttl)
        if cached is not None:
            return cached
        lock = await self._get_lock(key)
        async with lock:
            cached = self._cached_fresh(key, ttl)
            if cached is not None:
                return cached
            payload = await self._build_full_payload(path)
            self._cache[key] = (payload, time.monotonic())
            return payload

    def invalidate_cache(self) -> None:
        """Limpia toda la cache (utilidad para tests)."""
        self._cache.clear()
        self._cache_locks.clear()
        self._static_cache = None

    # --- Subset estático (ffmpeg/mkv/disk) — TTL más largo -----------------

    async def _build_static_payload(self) -> dict:
        coros = [
            self._as_coro_sync(check_disk_space, ""),
            self._as_coro_sync(check_executable, "ffmpeg"),
            self._as_coro_sync(check_executable, "mkvmerge", "mkvtoolnix"),
        ]
        results = await asyncio.gather(*coros)
        return self._compose_payload(list(results))

    async def get_static_cached(self, ttl: float = STATIC_CACHE_TTL_SECONDS) -> dict:
        if self._static_cache is not None:
            payload, ts = self._static_cache
            if (time.monotonic() - ts) < ttl:
                return payload
        async with self._static_lock:
            if self._static_cache is not None:
                payload, ts = self._static_cache
                if (time.monotonic() - ts) < ttl:
                    return payload
            payload = await self._build_static_payload()
            self._static_cache = (payload, time.monotonic())
            return payload
