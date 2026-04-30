"""Servicio de logs: ring buffer local + proxy HTTP a /logs de backends.

El ring buffer se instala una sola vez al arrancar la app. ``router.py``
y otros consumidores acceden al singleton vía ``LogsService``.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Deque, Optional

import httpx

log = logging.getLogger(__name__)


class RingBufferHandler(logging.Handler):
    """Logging handler que mantiene los últimos N registros en memoria."""

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__(level=logging.DEBUG)
        self.buffer: Deque[dict] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(
                {
                    "timestamp": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
            )
        except Exception:  # pragma: no cover
            pass


# Singleton del proceso. Lo expone ``install_local_ring_buffer`` y lo lee
# ``LogsService`` cuando alguien pide los logs locales.
_LOCAL_BUFFER: Optional[RingBufferHandler] = None


def install_local_ring_buffer() -> RingBufferHandler:
    """Instala el handler en el root logger; idempotente.

    Se invoca desde ``infrastructure.lifespan`` como hook de startup, en
    sustitución del side-effect que tenía el import de ``api/logs_view.py``
    en el código pre-refactor.
    """
    global _LOCAL_BUFFER
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, RingBufferHandler):
            _LOCAL_BUFFER = h
            return h
    handler = RingBufferHandler()
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    _LOCAL_BUFFER = handler
    return handler


# Mapa servicio → URL backend (None = ring buffer local).
SERVICE_URLS: dict[str, Optional[str]] = {
    "processor-api": None,
    "chapter-splitter": os.environ.get("SPLITTER_URL", "http://chapter-splitter:8001"),
    "subtitle-generator": os.environ.get("SUBS_URL", "http://subtitle-generator:8002"),
    "dubbing-generator": os.environ.get("DUBBING_URL", "http://dubbing-generator:8003"),
    "telegram-fetcher": os.environ.get("TELEGRAM_FETCHER_URL", "http://telegram-fetcher:8004"),
}

ALLOWED_LEVELS = {"INFO", "WARN", "WARNING", "ERROR", "DEBUG", "ALL"}


def normalize_level(level: Optional[str]) -> Optional[str]:
    if not level:
        return None
    lvl = level.upper()
    if lvl == "ALL":
        return None
    if lvl == "WARN":
        lvl = "WARNING"
    if lvl not in {"INFO", "WARNING", "ERROR", "DEBUG"}:
        return None
    return lvl


class LogsService:
    """Devuelve líneas del ring buffer local o proxy a backends remotos."""

    def __init__(self, service_urls: dict[str, Optional[str]] | None = None) -> None:
        self._service_urls = service_urls if service_urls is not None else SERVICE_URLS

    def is_known(self, service: str) -> bool:
        return service in self._service_urls

    def known_services(self) -> list[str]:
        return sorted(self._service_urls)

    def get_local_lines(self, level: Optional[str], tail: int) -> list[dict]:
        if _LOCAL_BUFFER is None:
            return []
        buf = list(_LOCAL_BUFFER.buffer)
        if level:
            buf = [r for r in buf if r.get("level") == level]
        if tail > 0:
            buf = buf[-tail:]
        return [
            {
                "timestamp": r.get("timestamp"),
                "level": r.get("level"),
                "message": r.get("message"),
            }
            for r in buf
        ]

    def fetch_remote(self, service: str, level: Optional[str], tail: int) -> dict:
        """Proxy al endpoint ``/logs`` del backend remoto.

        Lanza ``RuntimeError`` con mensaje legible si la llamada falla; el
        router lo traduce a 502.
        """
        base_url = self._service_urls.get(service)
        if base_url is None:
            raise RuntimeError(f"service {service} no es remoto")
        try:
            resp = httpx.get(
                f"{base_url}/logs",
                params={"level": level or "ALL", "tail": tail},
                timeout=3.0,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(f"backend {service} unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise RuntimeError(f"backend {service} returned {resp.status_code}")
        data = resp.json() or {}
        return {
            "service": service,
            "lines": data.get("lines", []),
            "truncated": bool(data.get("truncated", False)),
        }

    def is_local(self, service: str) -> bool:
        return self._service_urls.get(service) is None
