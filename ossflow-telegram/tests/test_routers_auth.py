"""Tests for /telegram auth endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.asyncio


async def test_status_happy(wired_app) -> None:
    app, _db, _svc, _auth = wired_app
    with TestClient(app) as client:
        r = client.get("/telegram/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "disconnected"
    assert body["connected"] is True  # FakeTelegramService reports connected
    assert "have_credentials" in body


async def test_send_code_happy(wired_app) -> None:
    app, _db, svc, auth = wired_app
    with TestClient(app) as client:
        r = client.post("/telegram/auth/send-code", json={"phone": "+34111"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    svc.send_code.assert_awaited_once()
    assert auth.get_state().state == "awaiting_code"
    assert auth.get_state().phone_code_hash == "hash-xyz"


async def test_send_code_auth_failed(wired_app) -> None:
    app, _db, svc, _auth = wired_app
    svc.next_send_behavior = "auth_failed"
    with TestClient(app) as client:
        r = client.post("/telegram/auth/send-code", json={"phone": "+34"})
    assert r.status_code == 403
    body = r.json()
    assert body["code"] == "AuthFailedError"


async def test_sign_in_happy(wired_app) -> None:
    app, _db, _svc, auth = wired_app
    with TestClient(app) as client:
        client.post("/telegram/auth/send-code", json={"phone": "+34"})
        r = client.post("/telegram/auth/sign-in", json={"phone": "+34", "code": "12345"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["needs_2fa"] is False
    assert auth.get_state().state == "authenticated"


async def test_sign_in_needs_2fa(wired_app) -> None:
    app, _db, svc, auth = wired_app
    svc.next_code_behavior = "needs_2fa"
    with TestClient(app) as client:
        client.post("/telegram/auth/send-code", json={"phone": "+34"})
        r = client.post("/telegram/auth/sign-in", json={"phone": "+34", "code": "12345"})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "needs_2fa": True}
    assert auth.get_state().state == "awaiting_2fa"


async def test_sign_in_auth_failed(wired_app) -> None:
    app, _db, svc, _auth = wired_app
    svc.next_code_behavior = "auth_failed"
    with TestClient(app) as client:
        client.post("/telegram/auth/send-code", json={"phone": "+34"})
        r = client.post("/telegram/auth/sign-in", json={"phone": "+34", "code": "bad"})
    assert r.status_code == 403


async def test_2fa_happy(wired_app) -> None:
    app, _db, svc, auth = wired_app
    svc.next_code_behavior = "needs_2fa"
    with TestClient(app) as client:
        client.post("/telegram/auth/send-code", json={"phone": "+34"})
        client.post("/telegram/auth/sign-in", json={"phone": "+34", "code": "12345"})
        r = client.post("/telegram/auth/2fa", json={"password": "hunter2"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert auth.get_state().state == "authenticated"


async def test_logout_resets_state(wired_app) -> None:
    app, _db, _svc, auth = wired_app
    with TestClient(app) as client:
        client.post("/telegram/auth/send-code", json={"phone": "+34"})
        client.post("/telegram/auth/sign-in", json={"phone": "+34", "code": "12345"})
        assert auth.get_state().state == "authenticated"
        r = client.post("/telegram/auth/logout")
    assert r.status_code == 200
    assert auth.get_state().state == "disconnected"
