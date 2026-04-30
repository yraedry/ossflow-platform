"""Endpoints HTTP del módulo export.

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

from .dependencies import get_export_service
from .service import ExportService

log = logging.getLogger(__name__)

router = APIRouter(tags=["export"])


def _err(exc: ApiError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=exc.status_code)


async def _parse_body(request: Request) -> dict | JSONResponse:
    """Parsea body JSON; devuelve ``JSONResponse`` (400) en error."""
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
    return body


@router.post("/api/export/ossflow")
async def api_export_to_ossflow(
    request: Request,
    svc: ExportService = Depends(get_export_service),
):
    """Exporta un instructional al backend OssFlow."""
    body = await _parse_body(request)
    if isinstance(body, JSONResponse):
        return body

    path = body.get("path", "")
    instructor = body.get("instructor", "")
    base_url = body.get("base_url", "http://localhost:8080")

    try:
        return svc.export_to_ossflow(
            path=path, instructor=instructor, base_url=base_url,
        )
    except ApiError as exc:
        return _err(exc)


@router.get("/api/export/ossflow/status")
async def api_ossflow_status(
    svc: ExportService = Depends(get_export_service),
) -> Any:
    """Comprueba si el backend OssFlow es alcanzable."""
    return svc.ossflow_status()


@router.post("/api/export/plex")
async def api_export_plex(
    request: Request,
    svc: ExportService = Depends(get_export_service),
):
    """Exporta vídeos procesados en formato compatible con Plex."""
    body = await _parse_body(request)
    if isinstance(body, JSONResponse):
        return body

    name = body.get("name", "")
    chapters = body.get("chapters", [])
    source_dir = body.get("source_dir", "")
    output_dir = body.get("output_dir", "")

    try:
        return svc.export_to_plex(
            name=name,
            chapters=chapters,
            source_dir=source_dir,
            output_dir=output_dir,
        )
    except ApiError as exc:
        return _err(exc)
