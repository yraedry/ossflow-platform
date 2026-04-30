"""Servicio de metrics con TTL cache + lock asíncrono.

Estado en instancia (no globales). Para mantener un único snapshot
compartido por todos los clientes en una app FastAPI, ``dependencies.py``
expone un singleton scope-app construido en lifespan.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

# Backends GPU consultados en paralelo. Todos montan la misma GPU física;
# usamos la primera respuesta no vacía.
_DEFAULT_GPU_BACKENDS = [
    os.environ.get("SPLITTER_URL", "http://chapter-splitter:8001"),
    os.environ.get("SUBS_URL", "http://subtitle-generator:8002"),
    os.environ.get("DUBBING_URL", "http://dubbing-generator:8003"),
]

_CACHE_TTL_SECONDS = 5.0


def _bytes_to_gb(n: int | float) -> float:
    return round(float(n) / (1024 ** 3), 2)


class MetricsService:
    """Recopila métricas y cachea el resultado durante TTL segundos."""

    def __init__(
        self,
        *,
        gpu_backends: list[str] | None = None,
        load_settings: Callable[[], dict] | None = None,
        ttl_seconds: float = _CACHE_TTL_SECONDS,
    ) -> None:
        self._gpu_backends = (
            list(gpu_backends) if gpu_backends is not None else _DEFAULT_GPU_BACKENDS
        )
        self._load_settings = load_settings
        self._ttl = ttl_seconds
        self._cache: dict[str, Any] = {"data": None, "expires_at": 0.0}
        self._cache_lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None

    @property
    def gpu_backends(self) -> list[str]:
        return self._gpu_backends

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(1.5))
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def reset_cache(self) -> None:
        self._cache["data"] = None
        self._cache["expires_at"] = 0.0

    # --- recolectores síncronos -------------------------------------------

    @staticmethod
    def _cpu_temp_c() -> float | None:
        """Best-effort CPU temp; ``None`` si el kernel no la expone."""
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

    def _collect_cpu_ram(self) -> tuple[float, dict[str, float], float | None]:
        import psutil
        cpu = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        ram = {
            "used_gb": _bytes_to_gb(vm.total - vm.available),
            "total_gb": _bytes_to_gb(vm.total),
            "percent": float(vm.percent),
        }
        return cpu, ram, self._cpu_temp_c()

    @staticmethod
    def _disk_entry(label: str, path: str) -> dict[str, Any] | None:
        import psutil
        try:
            usage = psutil.disk_usage(path)
        except Exception as exc:  # noqa: BLE001
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

    def _collect_disks(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        local = self._disk_entry("Local", os.path.abspath(os.sep))
        if local:
            entries.append(local)

        try:
            settings = self._load_settings() if self._load_settings else {}
            lib = (settings or {}).get("library_path") or ""
        except Exception:  # noqa: BLE001
            lib = ""
        lib_mount = "/media" if Path("/media").exists() else lib
        if lib_mount and (not local or lib_mount != local["path"]):
            lib_entry = self._disk_entry("Biblioteca", lib_mount)
            if lib_entry:
                lib_entry["host_path"] = lib or None
                entries.append(lib_entry)
        return entries

    @staticmethod
    def _collect_gpus_local() -> list[dict[str, Any]]:
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
        except Exception:  # noqa: BLE001
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

    async def _collect_gpus(self) -> list[dict[str, Any]]:
        local = self._collect_gpus_local()
        if local:
            return local

        client = self._get_http_client()
        coros: list[Awaitable[Any]] = [
            client.get(f"{base}/gpu", timeout=1.5) for base in self._gpu_backends
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for base, resp in zip(self._gpu_backends, results):
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
            except Exception as exc:  # noqa: BLE001 - defensive
                log.debug("GPU backend %s bad response: %s", base, exc)
                continue
        return []

    async def _build_snapshot(self) -> dict[str, Any]:
        cpu_percent, ram, cpu_temp = self._collect_cpu_ram()
        disks = self._collect_disks()
        gpus = await self._collect_gpus()
        return {
            "cpu_percent": cpu_percent,
            "cpu_temp_c": cpu_temp,
            "ram": ram,
            "disk": disks[0] if disks else None,
            "disks": disks,
            "gpus": gpus,
            "ram_note": "container_visible",
        }

    async def snapshot(self) -> dict[str, Any]:
        """Devuelve un snapshot completo, cacheado durante ``ttl`` segundos.

        Doble check para que ``N`` requests concurrentes con cache expirada
        produzcan un único refresh.
        """
        now = time.monotonic()
        if self._cache["data"] is not None and now < self._cache["expires_at"]:
            return self._cache["data"]

        async with self._cache_lock:
            now = time.monotonic()
            if self._cache["data"] is not None and now < self._cache["expires_at"]:
                return self._cache["data"]
            data = await self._build_snapshot()
            self._cache["data"] = data
            self._cache["expires_at"] = time.monotonic() + self._ttl
            return data
