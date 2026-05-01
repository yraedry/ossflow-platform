"""Endpoints HTTP del módulo voices.

Mantiene el contrato exacto del legacy ``api/app.py`` (incluyendo el
shape ``{"error": ...}`` con status_code en lugar de
``HTTPException``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ossflow_api.shared.exceptions import ApiError

from .dependencies import get_voices_service
from .service import VoicesService

log = logging.getLogger(__name__)

router = APIRouter(tags=["voices"])


def _err(exc: ApiError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=exc.status_code)


@router.get("/api/voice-profiles")
async def api_list_voice_profiles(
    svc: VoicesService = Depends(get_voices_service),
) -> Any:
    """Lista todos los perfiles de voz por instructor."""
    try:
        return svc.list_profiles()
    except ApiError as exc:
        return _err(exc)


@router.post("/api/voice-profiles")
async def api_create_voice_profile(
    request: Request,
    svc: VoicesService = Depends(get_voices_service),
):
    """Extrae y guarda un perfil de voz para un instructor."""
    # Parseo de body al estilo del legacy ``_parse_json_body`` (400 en JSON
    # malformado o no-objeto).
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse(
            {"error": f"Invalid JSON body: {exc}"}, status_code=400,
        )
    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "Body must be a JSON object"}, status_code=400,
        )

    video_path = body.get("video_path", "")
    instructor = body.get("instructor", "")
    start_sec = float(body.get("start_sec", 60))
    duration = float(body.get("duration", 15))

    try:
        return svc.create_profile(
            video_path=video_path,
            instructor=instructor,
            start_sec=start_sec,
            duration=duration,
        )
    except ApiError as exc:
        # El legacy devolvía 400 sin ``Missing`` envuelto, pero con
        # mensaje en clave ``error``. Lo reproducimos exacto.
        return _err(exc)


@router.delete("/api/voice-profiles/{instructor}")
async def api_delete_voice_profile(
    instructor: str,
    svc: VoicesService = Depends(get_voices_service),
):
    """Borra el perfil de voz de un instructor."""
    try:
        return svc.delete_profile(instructor)
    except ApiError as exc:
        return _err(exc)
