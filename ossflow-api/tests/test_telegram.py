"""Tests for /api/telegram/* proxy router + telegram settings."""

from __future__ import annotations

import importlib

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    db_path = tmp_path / "bjj.db"
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("BJJ_DB_PATH", str(db_path))
    monkeypatch.setenv("TELEGRAM_FETCHER_URL", "http://telegram-fetcher:8004")

    from bjj_service_kit.db import engine as _eng, session as _sess
    _eng.reset_engine()
    _sess.reset_factory()

    import api.settings as settings_mod
    importlib.reload(settings_mod)

    import api.telegram as telegram_mod
    importlib.reload(telegram_mod)

    app = FastAPI()
    app.include_router(telegram_mod.router)
    app.include_router(settings_mod.router)

    return {
        "client": TestClient(app),
        "telegram_mod": telegram_mod,
        "settings_mod": settings_mod,
    }


def _patch_httpx(monkeypatch, telegram_mod, handler):
    """Replace httpx.AsyncClient with a MockTransport-backed client."""
    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(telegram_mod.httpx, "AsyncClient", _Client)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_status_happy(env, monkeypatch):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/telegram/status"
        return httpx.Response(200, json={"authenticated": True, "phone": "+34..."})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].get("/api/telegram/status")
    assert r.status_code == 200
    assert r.json()["authenticated"] is True


def test_status_backend_unreachable(env, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].get("/api/telegram/status")
    assert r.status_code == 502


def test_status_timeout(env, monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow")

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].get("/api/telegram/status")
    assert r.status_code == 504


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_auth_send_code_ok(env, monkeypatch):
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = request.content
        return httpx.Response(200, json={"phone_code_hash": "abc"})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post(
        "/api/telegram/auth/send-code", json={"phone": "+341234"}
    )
    assert r.status_code == 200
    assert r.json() == {"phone_code_hash": "abc"}
    assert seen["path"] == "/telegram/auth/send-code"
    assert b"+341234" in seen["body"]


def test_auth_send_code_missing_phone(env, monkeypatch):
    def handler(request):
        pytest.fail("backend must not be called")

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post("/api/telegram/auth/send-code", json={})
    assert r.status_code == 422


def test_auth_send_code_backend_error_forwarded(env, monkeypatch):
    def handler(request):
        return httpx.Response(
            400, json={"detail": "invalid phone format"}
        )

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post(
        "/api/telegram/auth/send-code", json={"phone": "bogus"}
    )
    assert r.status_code == 400
    assert "invalid" in r.json()["detail"]


def test_auth_sign_in_ok(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/telegram/auth/sign-in"
        return httpx.Response(200, json={"ok": True})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post(
        "/api/telegram/auth/sign-in",
        json={"phone": "+1", "code": "12345", "phone_code_hash": "h"},
    )
    assert r.status_code == 200


def test_auth_2fa_ok(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/telegram/auth/2fa"
        return httpx.Response(200, json={"ok": True})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post(
        "/api/telegram/auth/2fa", json={"password": "secret"}
    )
    assert r.status_code == 200


def test_auth_logout_ok(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/telegram/auth/logout"
        return httpx.Response(200, json={"ok": True})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post("/api/telegram/auth/logout")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Channels + Sync
# ---------------------------------------------------------------------------

def test_channels_list(env, monkeypatch):
    body = [{"username": "foo", "title": "Foo Channel"}]

    def handler(request):
        assert request.url.path == "/telegram/channels"
        return httpx.Response(200, json=body)

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].get("/api/telegram/channels")
    assert r.status_code == 200
    assert r.json() == body


def test_sync_channel_enqueues(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/telegram/channels/somechannel/sync"
        assert request.method == "POST"
        assert b"100" in request.content
        return httpx.Response(202, json={"job_id": "job-123"})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post(
        "/api/telegram/channels/somechannel/sync", json={"limit": 100}
    )
    assert r.status_code == 202
    assert r.json()["job_id"] == "job-123"


def test_sync_channel_bad_limit(env, monkeypatch):
    def handler(request):
        pytest.fail("backend must not be called")

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post(
        "/api/telegram/channels/foo/sync", json={"limit": "huge"}
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

def test_media_list_chronological(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/telegram/media"
        q = dict(request.url.params)
        assert q["view"] == "chronological"
        assert q["channel"] == "foo"
        assert q["page"] == "2"
        return httpx.Response(200, json={"items": [], "total": 0})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].get(
        "/api/telegram/media",
        params={"channel": "foo", "view": "chronological", "page": 2, "page_size": 25},
    )
    assert r.status_code == 200


def test_media_list_by_author(env, monkeypatch):
    def handler(request):
        q = dict(request.url.params)
        assert q["view"] == "by_author"
        return httpx.Response(200, json={"groups": []})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].get("/api/telegram/media", params={"view": "by_author"})
    assert r.status_code == 200
    assert "groups" in r.json()


def test_media_list_bad_view(env, monkeypatch):
    def handler(request):
        pytest.fail("backend must not be called")

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].get("/api/telegram/media", params={"view": "wat"})
    assert r.status_code == 422


def test_media_put_metadata(env, monkeypatch):
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = request.content
        return httpx.Response(200, json={"ok": True})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].put(
        "/api/telegram/media/chan/42",
        json={"author": "Danaher", "title": "Tripod", "chapter_num": 3},
    )
    assert r.status_code == 200
    assert seen["path"] == "/telegram/media/chan/42"
    assert b"Danaher" in seen["body"]


def test_media_put_metadata_empty_body(env, monkeypatch):
    def handler(request):
        pytest.fail("backend must not be called")

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].put("/api/telegram/media/chan/42", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def test_download_start(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/telegram/download"
        assert request.method == "POST"
        return httpx.Response(202, json={"job_id": "dl-1"})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post(
        "/api/telegram/download",
        json={"channel_id": "foo", "author": "A", "title": "T"},
    )
    assert r.status_code == 202
    assert r.json()["job_id"] == "dl-1"


def test_download_start_missing_fields(env, monkeypatch):
    def handler(request):
        pytest.fail("backend must not be called")

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post(
        "/api/telegram/download", json={"channel_id": "foo"}
    )
    assert r.status_code == 422


def test_download_cancel(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/telegram/download/dl-1/cancel"
        assert request.method == "POST"
        return httpx.Response(200, json={"cancelled": True})

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].post("/api/telegram/download/dl-1/cancel")
    assert r.status_code == 200


def test_download_jobs_list(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/telegram/download/jobs"
        q = dict(request.url.params)
        assert q.get("status") == "running"
        return httpx.Response(200, json=[{"id": "dl-1", "status": "running"}])

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    r = env["client"].get(
        "/api/telegram/download/jobs", params={"status": "running"}
    )
    assert r.status_code == 200
    assert r.json()[0]["id"] == "dl-1"


# ---------------------------------------------------------------------------
# SSE proxy
# ---------------------------------------------------------------------------

def test_sse_download_events_proxy(env, monkeypatch):
    """At least one event should be forwarded from the backend stream."""

    def handler(request):
        assert request.url.path == "/telegram/download/dl-1/events"
        payload = (
            b"event: progress\ndata: {\"percent\": 10}\n\n"
            b"event: done\ndata: {}\n\n"
        )
        return httpx.Response(
            200, content=payload, headers={"content-type": "text/event-stream"}
        )

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    with env["client"].stream(
        "GET", "/api/telegram/download/dl-1/events"
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = b"".join(chunk for chunk in r.iter_bytes())
    assert b"event: progress" in body
    assert b"percent" in body


def test_sse_sync_events_proxy(env, monkeypatch):
    def handler(request):
        assert "/channels/foo/sync/job-1/events" in request.url.path
        return httpx.Response(
            200,
            content=b"event: tick\ndata: 1\n\n",
            headers={"content-type": "text/event-stream"},
        )

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    with env["client"].stream(
        "GET", "/api/telegram/channels/foo/sync/job-1/events"
    ) as r:
        assert r.status_code == 200
        body = b"".join(chunk for chunk in r.iter_bytes())
    assert b"event: tick" in body


def test_sse_backend_error_yields_error_event(env, monkeypatch):
    def handler(request):
        return httpx.Response(500, content=b"boom")

    _patch_httpx(monkeypatch, env["telegram_mod"], handler)
    with env["client"].stream(
        "GET", "/api/telegram/download/x/events"
    ) as r:
        assert r.status_code == 200  # SSE stream is always 200; error in payload
        body = b"".join(chunk for chunk in r.iter_bytes())
    assert b"event: error" in body


# ---------------------------------------------------------------------------
# Settings: telegram credentials
# ---------------------------------------------------------------------------

def test_settings_accepts_telegram_credentials(env):
    client = env["client"]
    r = client.put(
        "/api/settings",
        json={
            "telegram_api_id": 123456,
            "telegram_api_hash": "a" * 32,
        },
    )
    assert r.status_code == 200
    body = r.json()
    # PUT echoes the unmasked value back so the frontend's optimistic update
    # has the real hash without an extra round-trip.
    assert body["telegram_api_id"] == 123456
    assert body["telegram_api_hash"] == "a" * 32

    # Public GET masks the hash to avoid leaking it to the browser.
    r = client.get("/api/settings")
    assert r.status_code == 200
    assert r.json()["telegram_api_id"] == 123456
    assert r.json()["telegram_api_hash"] == "***"

    # The internal endpoint (used by telegram-fetcher) returns the real hash.
    r = client.get("/api/settings/internal")
    assert r.status_code == 200
    assert r.json()["telegram_api_id"] == 123456
    assert r.json()["telegram_api_hash"] == "a" * 32


def test_settings_defaults_telegram_none(env):
    r = env["client"].get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["telegram_api_id"] is None
    assert body["telegram_api_hash"] is None


def test_settings_rejects_bad_api_id(env):
    r = env["client"].put(
        "/api/settings", json={"telegram_api_id": "not-an-int"}
    )
    assert r.status_code == 422


def test_settings_rejects_bad_api_hash(env):
    r = env["client"].put(
        "/api/settings", json={"telegram_api_hash": "short"}
    )
    assert r.status_code == 422

    r = env["client"].put(
        "/api/settings", json={"telegram_api_hash": "z" * 32}  # not hex
    )
    assert r.status_code == 422


def test_settings_allows_null_telegram(env):
    client = env["client"]
    client.put(
        "/api/settings",
        json={"telegram_api_id": 42, "telegram_api_hash": "a" * 32},
    )
    r = client.put(
        "/api/settings",
        json={"telegram_api_id": None, "telegram_api_hash": None},
    )
    assert r.status_code == 200
    assert r.json()["telegram_api_id"] is None
    assert r.json()["telegram_api_hash"] is None
