"""Endpoints HTTP del módulo promote.

Reproduce el contrato del antiguo ``api/promote.py``: mismos paths,
mismos métodos, mismos payloads. La lógica vive en ``PromoteService``;
este router sólo valida con Pydantic y delega.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_promote_service
from .schemas import PromoteChapterBody, PromoteSeasonBody
from .service import PromoteService

router = APIRouter(prefix="/api/promote", tags=["promote"])


@router.post("/chapter")
def promote_chapter(
    body: PromoteChapterBody,
    svc: PromoteService = Depends(get_promote_service),
) -> dict:
    """Promueve un único capítulo doblado a su forma final multi-track."""
    return svc.promote_one(body.video_path)


@router.post("/season")
def promote_season(
    body: PromoteSeasonBody,
    svc: PromoteService = Depends(get_promote_service),
) -> dict:
    """Promueve cada capítulo doblado bajo ``season_path``. Secuencial."""
    return svc.promote_season(body.season_path)
