"""Endpoints HTTP del módulo subtitles.

Reproduce el contrato del antiguo ``api/subtitles.py``: mismos paths,
mismos métodos, mismos payloads. La lógica vive en ``SubtitlesService``;
este router sólo valida con Pydantic y delega.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ossflow_api.shared.exceptions import ApiError

from .dependencies import get_subtitles_service
from .schemas import (
    AnalyzeBody,
    ApplyBody,
    RegenerateBody,
    TranslateBody,
    ValidateBody,
)
from .service import SubtitlesService

router = APIRouter(prefix="/api/subtitles", tags=["subtitles"])


def _to_http(exc: ApiError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.post("/validate")
async def validate(
    body: ValidateBody,
    svc: SubtitlesService = Depends(get_subtitles_service),
) -> dict:
    try:
        return await svc.validate(body)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/regenerate-segment")
async def regenerate_segment(
    body: RegenerateBody,
    svc: SubtitlesService = Depends(get_subtitles_service),
) -> dict:
    try:
        return await svc.regenerate_segment(body)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/apply-segment")
async def apply_segment(
    body: ApplyBody,
    svc: SubtitlesService = Depends(get_subtitles_service),
) -> dict:
    try:
        return await svc.apply_segment(body)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/maintenance/clear-locks")
async def clear_locks(
    svc: SubtitlesService = Depends(get_subtitles_service),
) -> dict:
    """Limpia locks colgados del HuggingFace hub en subtitle-generator."""
    try:
        return await svc.clear_locks()
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/maintenance/restart")
async def restart_subtitle_service(
    svc: SubtitlesService = Depends(get_subtitles_service),
) -> dict:
    """Proxy de restart graceful (libera VRAM; Docker reinicia el contenedor)."""
    try:
        return await svc.restart()
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/translate")
async def translate(
    body: TranslateBody,
    svc: SubtitlesService = Depends(get_subtitles_service),
) -> dict:
    try:
        return await svc.translate(body)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/analyze")
async def analyze_video(
    body: AnalyzeBody,
    svc: SubtitlesService = Depends(get_subtitles_service),
) -> dict:
    try:
        return await svc.analyze(body)
    except ApiError as exc:
        raise _to_http(exc) from exc
