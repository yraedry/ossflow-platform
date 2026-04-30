"""Endpoints HTTP del módulo preflight.

IMPORTANTE: comparte prefix ``/api/pipeline`` con ``pipeline_router``, y este
último tiene una ruta catch-all ``GET /{pipeline_id}``. ``main.py`` debe
registrar este router ANTES que el de pipeline para que ``/preflight`` no
sea capturado como id.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from .dependencies import get_preflight_service
from .service import PreflightService

router = APIRouter(prefix="/api/pipeline", tags=["preflight"])


@router.get("/preflight")
async def preflight(
    path: str = Query("", description="Ruta del instruccional"),
    svc: PreflightService = Depends(get_preflight_service),
) -> dict:
    return await svc.get_preflight_cached(path)


@router.get("/preflight/static")
async def preflight_static(
    svc: PreflightService = Depends(get_preflight_service),
) -> dict:
    """Subset estático (ffmpeg, mkvtoolnix, disk). TTL 5 min."""
    return await svc.get_static_cached()
