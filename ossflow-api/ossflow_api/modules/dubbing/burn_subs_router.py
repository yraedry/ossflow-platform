"""Router HTTP de ``/api/burn-subs/*``.

Endpoint absorbido del antiguo ``api/burn_subs.py``. Sigue exponiendo
``POST /api/burn-subs`` para no romper el frontend, pero la lógica vive
en el módulo ``dubbing`` por afinidad funcional.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ossflow_api.modules.jobs.dependencies import get_background_jobs_service
from ossflow_api.modules.jobs.services.background import BackgroundJobsService

from .burn_subs_service import BurnSubsService

router = APIRouter(prefix="/api/burn-subs", tags=["burn-subs"])


def get_burn_subs_service(
    jobs: BackgroundJobsService = Depends(get_background_jobs_service),
) -> BurnSubsService:
    from api.settings import get_library_path

    return BurnSubsService(jobs=jobs, library_path_loader=get_library_path)


@router.post("")
async def start_burn(
    request: Request,
    svc: BurnSubsService = Depends(get_burn_subs_service),
):
    """Inicia un job de burn-subs. Body: ``{"path": "..."}``."""
    if not svc.ffmpeg_available():
        raise HTTPException(status_code=503, detail="ffmpeg not available")

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=422, detail="Invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Body must be an object")

    raw_path = body.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise HTTPException(status_code=422, detail="path is required")

    try:
        return svc.submit(raw_path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        msg = str(exc)
        if "not configured" in msg:
            code = 400
        elif "Not found" in msg:
            code = 404
        elif "No videos" in msg:
            return JSONResponse(
                status_code=409,
                content={"error": msg, "path": raw_path},
            )
        else:
            code = 422
        raise HTTPException(status_code=code, detail=msg) from exc
