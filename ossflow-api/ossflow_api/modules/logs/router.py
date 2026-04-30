"""Endpoints HTTP del módulo logs."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .dependencies import get_logs_service
from .service import ALLOWED_LEVELS, LogsService, normalize_level

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/")
def get_logs(
    service: str = Query(..., description="Service name"),
    level: Optional[str] = Query(None, description="Filter by level (INFO/WARN/ERROR/DEBUG/ALL)"),
    tail: int = Query(500, ge=1, le=5000),
    svc: LogsService = Depends(get_logs_service),
):
    if not svc.is_known(service):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown service '{service}'. Allowed: {svc.known_services()}",
        )
    if level is not None and level.upper() not in ALLOWED_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level '{level}'. Allowed: INFO, WARN, ERROR, DEBUG, ALL",
        )
    normalized = normalize_level(level)

    if svc.is_local(service):
        return {
            "service": service,
            "lines": svc.get_local_lines(normalized, tail),
            "truncated": False,
        }

    try:
        return svc.fetch_remote(service, normalized, tail)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
