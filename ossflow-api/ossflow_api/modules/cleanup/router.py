"""Endpoints HTTP del módulo cleanup."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .dependencies import get_cleanup_service
from .service import CleanupService

router = APIRouter(prefix="/api/cleanup", tags=["cleanup"])


@router.get("/scan")
async def scan(
    path: str,
    svc: CleanupService = Depends(get_cleanup_service),
):
    """Escanea ``path`` y devuelve candidatos a borrar agrupados por categoría."""
    if not path:
        raise HTTPException(status_code=400, detail="path es obligatorio")
    try:
        target = svc.resolve_under_library(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        return svc.scan_tree(target)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/start")
async def start_scan(
    path: str,
    svc: CleanupService = Depends(get_cleanup_service),
):
    """Lanza el escaneo como background job. Devuelve ``{job_id}``."""
    if not path:
        raise HTTPException(status_code=400, detail="path es obligatorio")
    try:
        job_id = svc.submit_scan(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"job_id": job_id}


@router.get("/job/{job_id}")
async def get_job(
    job_id: str,
    svc: CleanupService = Depends(get_cleanup_service),
):
    job = svc._jobs.get(job_id)  # noqa: SLF001 — proxy fino
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.post("/apply")
async def apply(
    request: Request,
    svc: CleanupService = Depends(get_cleanup_service),
):
    """Borra los paths indicados (revalidando cada uno bajo library_path)."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"detail": "Body debe ser JSON"}, status_code=422)

    if not isinstance(body, dict):
        return JSONResponse({"detail": "Body debe ser un objeto JSON"}, status_code=422)

    paths = body.get("paths") or []
    dry_run = bool(body.get("dry_run", False))
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        return JSONResponse(
            {"detail": "paths debe ser lista de strings"}, status_code=422
        )

    return svc.apply_deletions(paths, dry_run=dry_run)
