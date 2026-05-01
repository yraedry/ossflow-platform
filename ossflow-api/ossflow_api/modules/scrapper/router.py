"""Endpoints HTTP del módulo scrapper."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ossflow_api.shared.exceptions import ApiError

from .dependencies import get_scrapper_service
from .service import ScrapperService

router = APIRouter(prefix="/api/scrapper", tags=["scrapper"])


def _to_http(exc: ApiError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.get("/providers")
async def list_providers(
    svc: ScrapperService = Depends(get_scrapper_service),
) -> Any:
    try:
        return await svc.list_providers()
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.get("/{instructional_path:path}")
async def get_oracle(
    instructional_path: str,
    svc: ScrapperService = Depends(get_scrapper_service),
):
    try:
        return JSONResponse(svc.get_oracle(instructional_path))
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/{instructional_path:path}/resolve")
async def resolve_oracle(
    instructional_path: str,
    request: Request,
    svc: ScrapperService = Depends(get_scrapper_service),
):
    try:
        body = await request.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        return await svc.resolve(instructional_path, body)
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.post("/{instructional_path:path}/scrape")
async def scrape_oracle(
    instructional_path: str,
    request: Request,
    svc: ScrapperService = Depends(get_scrapper_service),
):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid JSON")
    try:
        return JSONResponse(await svc.scrape(instructional_path, body))
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.put("/{instructional_path:path}")
async def put_oracle(
    instructional_path: str,
    request: Request,
    svc: ScrapperService = Depends(get_scrapper_service),
):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid JSON")
    try:
        return JSONResponse(await svc.put_oracle(instructional_path, body))
    except ApiError as exc:
        raise _to_http(exc) from exc


@router.delete("/{instructional_path:path}")
async def delete_oracle(
    instructional_path: str,
    svc: ScrapperService = Depends(get_scrapper_service),
):
    try:
        return svc.delete_oracle(instructional_path)
    except ApiError as exc:
        raise _to_http(exc) from exc
