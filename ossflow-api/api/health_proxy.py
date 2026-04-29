"""Health proxy — aggregates backend /health so the frontend can avoid CORS."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/health", tags=["health"])

BACKENDS: dict[str, str] = {
    "chapter-splitter": os.environ.get("SPLITTER_URL", "http://chapter-splitter:8001"),
    "subtitle-generator": os.environ.get("SUBS_URL", "http://subtitle-generator:8002"),
    "dubbing-generator": os.environ.get("DUBBING_URL", "http://dubbing-generator:8003"),
    "ollama": os.environ.get("OLLAMA_URL", "http://ollama:11434"),
}


async def _ping(client: httpx.AsyncClient, service: str, base: str) -> dict:
    # Ollama no expone /health — usa /api/tags como liveness probe.
    health_path = "/api/tags" if service == "ollama" else "/health"
    try:
        r = await client.get(f"{base}{health_path}", timeout=3.0)
        if r.status_code == 200:
            return {"service": service, "status": "up", "body": r.json()}
        return {"service": service, "status": "down", "error": f"HTTP {r.status_code}"}
    except Exception as exc:
        return {"service": service, "status": "down", "error": str(exc)}


@router.get("/backends")
async def backends_health() -> dict:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_ping(client, s, url) for s, url in BACKENDS.items()]
        )
    return {"services": results}


@router.get("/{service}")
async def one_backend(service: str) -> dict:
    base: Optional[str] = BACKENDS.get(service)
    if not base:
        raise HTTPException(status_code=404, detail=f"unknown service '{service}'")
    async with httpx.AsyncClient() as client:
        return await _ping(client, service, base)
