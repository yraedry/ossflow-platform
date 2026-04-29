"""Auth endpoints: status, send-code, sign-in, 2fa, logout."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..errors import AuthRequiredError


router = APIRouter(prefix="/telegram", tags=["telegram-auth"])


class _PhoneBody(BaseModel):
    phone: str


class _SignInBody(BaseModel):
    phone: str
    code: str


class _PasswordBody(BaseModel):
    password: str


@router.get("/status")
async def status(request: Request) -> dict:
    svc = getattr(request.app.state, "telegram_service", None)
    auth_store = request.app.state.auth_store
    state = auth_store.get_state().model_dump()
    connected = False
    if svc is not None:
        try:
            connected = bool(await svc._is_connected())  # noqa: SLF001
        except Exception:  # noqa: BLE001
            connected = False
    config = getattr(request.app.state, "config", None)
    have_creds = bool(config and config.have_credentials())
    return {
        **state,
        "connected": connected,
        "have_credentials": have_creds,
    }


@router.post("/auth/send-code")
async def send_code(body: _PhoneBody, request: Request) -> dict:
    svc = _require_service(request)
    # Hash is stored in the auth_store (memory only); we do NOT return it.
    await svc.send_code(body.phone)
    return {"ok": True}


@router.post("/auth/sign-in")
async def sign_in(body: _SignInBody, request: Request) -> dict:
    svc = _require_service(request)
    auth_store = request.app.state.auth_store
    phone_code_hash: Optional[str] = auth_store.get_state().phone_code_hash
    if not phone_code_hash:
        raise HTTPException(status_code=400, detail="no active send-code; call /auth/send-code first")
    await svc.sign_in_code(body.phone, body.code, phone_code_hash)
    state = auth_store.get_state().state
    if state == "awaiting_2fa":
        return {"ok": True, "needs_2fa": True}
    return {"ok": True, "needs_2fa": False}


@router.post("/auth/2fa")
async def sign_in_2fa(body: _PasswordBody, request: Request) -> dict:
    svc = _require_service(request)
    await svc.sign_in_2fa(body.password)
    return {"ok": True}


@router.post("/auth/logout")
async def logout(request: Request) -> dict:
    svc = _require_service(request)
    await svc.logout()
    return {"ok": True}


def _require_service(request: Request):
    svc = getattr(request.app.state, "telegram_service", None)
    if svc is None:
        raise AuthRequiredError("Telegram service not configured (missing API credentials)")
    return svc
