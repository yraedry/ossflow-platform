"""Endpoints HTTP del módulo duplicates."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_duplicates_service
from .service import DuplicatesService

router = APIRouter(prefix="/api/duplicates", tags=["duplicates"])


def _validate_to_path(svc: DuplicatesService, path: str):
    try:
        return svc.validate_path(path)
    except ValueError as exc:
        raise HTTPException(
            status_code=404 if "no existe" in str(exc) else 400,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/scan")
async def scan(
    path: str,
    deep: bool = False,
    svc: DuplicatesService = Depends(get_duplicates_service),
) -> dict[str, Any]:
    root = _validate_to_path(svc, path)
    return svc.scan(root, deep=deep)


@router.post("/start")
async def start_scan(
    path: str,
    deep: bool = False,
    svc: DuplicatesService = Depends(get_duplicates_service),
):
    try:
        job_id = svc.submit_scan(path, deep=deep)
    except ValueError as exc:
        raise HTTPException(
            status_code=404 if "no existe" in str(exc) else 400,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"job_id": job_id}


@router.get("/job/{job_id}")
async def get_job(
    job_id: str,
    svc: DuplicatesService = Depends(get_duplicates_service),
):
    job = svc._jobs.get(job_id)  # noqa: SLF001 — proxy fino
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()
