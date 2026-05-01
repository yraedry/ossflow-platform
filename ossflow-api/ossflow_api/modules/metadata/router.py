"""Endpoints HTTP del módulo metadata."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ossflow_api.shared.exceptions import ApiError

from .dependencies import get_metadata_service
from .service import MetadataService

router = APIRouter(prefix="/api/library", tags=["metadata"])


def _to_http(exc: ApiError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.get("/{name}/metadata")
async def get_metadata(name: str, svc: MetadataService = Depends(get_metadata_service)):
    try:
        return svc.get(name)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.put("/{name}/metadata")
async def put_metadata(
    name: str,
    request: Request,
    svc: MetadataService = Depends(get_metadata_service),
):
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid JSON") from exc
    try:
        return svc.put(name, body)
    except ApiError as exc:
        raise _to_http(exc) from exc
