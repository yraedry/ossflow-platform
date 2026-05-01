"""Endpoints HTTP del módulo settings."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .dependencies import get_settings_service
from .service import SettingsService

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def get_settings(svc: SettingsService = Depends(get_settings_service)):
    return svc.mask_secrets(svc.load())


@router.get("/internal")
async def get_settings_internal(
    request: Request,
    svc: SettingsService = Depends(get_settings_service),
):
    """Settings sin enmascarar para llamadas service-to-service.

    Telegram-fetcher necesita el ``telegram_api_hash`` real para llamar a la
    API de Telethon; no puede usar el endpoint público enmascarado (mandaría
    "***" → ApiIdInvalidError). Restringido a rangos de red Docker privada
    para que nunca sea accesible desde el navegador del host.
    """
    client_host = request.client.host if request.client else ""
    allowed = (
        client_host.startswith("172.")
        or client_host.startswith("10.")
        or client_host.startswith("192.168.")
        or client_host in ("127.0.0.1", "localhost", "::1", "testclient")
    )
    if not allowed:
        return JSONResponse(
            {"error": "internal endpoint, network-restricted"},
            status_code=403,
        )
    return svc.load()


@router.put("")
async def put_settings(
    request: Request,
    svc: SettingsService = Depends(get_settings_service),
):
    body = await request.json()
    error, updated = svc.update(body)
    if error is not None:
        return JSONResponse({"error": error}, status_code=422)
    return updated
