"""Endpoints HTTP del módulo chapters."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ossflow_api.shared.exceptions import ApiError

from .dependencies import get_chapters_service
from .service import ChaptersService

router = APIRouter(prefix="/api/chapters", tags=["chapters"])


def _to_http(exc: ApiError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


async def _read_json_body(request: Request) -> object:
    """Lee el body como JSON o devuelve 422 si está mal formado."""
    try:
        return await request.json()
    except Exception as exc:  # noqa: BLE001 — FastAPI devuelve 422 para cualquier fallo
        raise HTTPException(status_code=422, detail="Invalid JSON body") from exc


@router.post("/rename-by-oracle")
async def rename_season_by_oracle(
    request: Request,
    svc: ChaptersService = Depends(get_chapters_service),
):
    """Renombra todos los capítulos de una Season según los títulos del oracle.

    Body: ``{"season_path": "...", "oracle": {<OracleResult>}}``.
    """
    body = await _read_json_body(request)
    try:
        return svc.rename_by_oracle(body)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.patch("/rename")
async def rename_chapter(
    request: Request,
    svc: ChaptersService = Depends(get_chapters_service),
):
    """Renombra un capítulo (y sus sidecars) preservando el prefijo SNNeMM.

    Body: ``{"old_path": "...", "new_title": "..."}``.
    """
    body = await _read_json_body(request)
    try:
        return svc.rename_one(body)
    except ApiError as exc:
        raise _to_http(exc) from exc
