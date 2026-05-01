"""Servicio de health: hace ping HTTP a cada backend y normaliza la respuesta."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx

# Mapa servicio → URL leído de env vars en import time. Se mantiene como
# constante a nivel de módulo porque las URLs no cambian en runtime.
BACKENDS: dict[str, str] = {
    "chapter-splitter": os.environ.get("SPLITTER_URL", "http://chapter-splitter:8001"),
    "subtitle-generator": os.environ.get("SUBS_URL", "http://subtitle-generator:8002"),
    "dubbing-generator": os.environ.get("DUBBING_URL", "http://dubbing-generator:8003"),
    "ollama": os.environ.get("OLLAMA_URL", "http://ollama:11434"),
}


class HealthService:
    """Orquesta los pings de salud de los backends."""

    def __init__(self, backends: dict[str, str] | None = None) -> None:
        # Inyectable para tests sin tocar env vars.
        self._backends = backends if backends is not None else BACKENDS

    async def ping_one(self, service: str) -> dict:
        base = self._backends.get(service)
        if not base:
            return {"service": service, "status": "unknown"}
        async with httpx.AsyncClient() as client:
            return await self._ping(client, service, base)

    async def ping_all(self) -> dict:
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *[self._ping(client, s, url) for s, url in self._backends.items()]
            )
        return {"services": results}

    def is_known(self, service: str) -> bool:
        return service in self._backends

    @staticmethod
    async def _ping(client: httpx.AsyncClient, service: str, base: str) -> dict:
        # Ollama no expone /health — usa /api/tags como liveness probe.
        health_path = "/api/tags" if service == "ollama" else "/health"
        try:
            r = await client.get(f"{base}{health_path}", timeout=3.0)
            if r.status_code == 200:
                return {"service": service, "status": "up", "body": r.json()}
            return {"service": service, "status": "down", "error": f"HTTP {r.status_code}"}
        except Exception as exc:  # noqa: BLE001 — capturamos cualquier fallo de red
            return {"service": service, "status": "down", "error": str(exc)}
