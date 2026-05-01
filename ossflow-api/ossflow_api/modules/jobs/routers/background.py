"""Router HTTP de ``/api/background-jobs/*``.

**Endpoint sin trailing slash**: solo se registra ``@router.get("")``. La
forma con slash recibe ``307 Temporary Redirect`` automáticamente porque
FastAPI usa ``redirect_slashes=True`` por defecto. Decisión documentada
en el spec anexo §0 (deuda eliminada vs el legacy que registraba ambos).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_background_jobs_service
from ..schemas import BackgroundJobListResponse, BackgroundJobResponse
from ..services.background import BackgroundJobsService

router = APIRouter(prefix="/api/background-jobs", tags=["background"])


@router.get("", response_model=BackgroundJobListResponse)
async def list_jobs(
    type: Optional[str] = None,
    svc: BackgroundJobsService = Depends(get_background_jobs_service),
) -> dict:
    return {"jobs": [j.to_dict() for j in svc.list_all(type_filter=type)]}


@router.get("/{job_id}", response_model=BackgroundJobResponse)
async def get_job(
    job_id: str,
    svc: BackgroundJobsService = Depends(get_background_jobs_service),
) -> dict:
    job = svc.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()
