"""Tests del router del módulo telegram vía FastAPI TestClient + respx.

El servicio se inyecta vía ``app.dependency_overrides`` con un
``TelegramService`` real apuntando a una URL falsa que respx
intercepta. Así verificamos también la composición ``base_url`` →
``base_url/telegram`` que hace el servicio internamente.
"""

from __future__ import annotations

import json as _json
from typing import Any, Optional

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from ossflow_api.modules.telegram import telegram_router
from ossflow_api.modules.telegram.dependencies import get_telegram_service
from ossflow_api.modules.telegram.service import TelegramService

# El servicio compone ``{base_url}/telegram`` para cada llamada, por eso
# las rutas mockeadas incluyen el prefijo ``/telegram``.
FAKE_TG = "http://fake-tg"
FAKE_TG_PREFIX = f"{FAKE_TG}/telegram"


def _build_client(
    *,
    settings: Optional[dict[str, Any]] = None,
    registrar_calls: Optional[list[tuple]] = None,
) -> TestClient:
    """Construye un TestClient con un ``TelegramService`` aislado.

    ``settings`` se devuelve por el loader inyectado (para
    ``author_aliases``). ``registrar_calls`` (lista mutable) recoge las
    llamadas que el servicio hace al registrar de bg jobs, para que
    los tests puedan aseverar sobre ellas.
    """
    settings = settings or {}

    def _settings_loader() -> dict[str, Any]:
        return settings

    if registrar_calls is None:
        registrar_calls = []

    def _registrar(kind: str, job_id: str, sse_path: str, params: dict) -> None:
        registrar_calls.append((kind, job_id, sse_path, params))

    svc = TelegramService(
        base_url=FAKE_TG,
        settings_loader=_settings_loader,
        bg_job_registrar=_registrar,
    )
    app = FastAPI()
    app.include_router(telegram_router)
    app.dependency_overrides[get_telegram_service] = lambda: svc
    return TestClient(app)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@respx.mock
def test_status_happy() -> None:
    route = respx.get(f"{FAKE_TG_PREFIX}/status").mock(
        return_value=Response(200, json={"authenticated": True, "phone": "+34..."})
    )
    client = _build_client()

    r = client.get("/api/telegram/status")

    assert r.status_code == 200
    assert r.json()["authenticated"] is True
    assert route.called


@respx.mock
def test_status_backend_unreachable_returns_502() -> None:
    respx.get(f"{FAKE_TG_PREFIX}/status").mock(
        side_effect=httpx.ConnectError("refused")
    )
    client = _build_client()

    r = client.get("/api/telegram/status")
    assert r.status_code == 502


@respx.mock
def test_status_timeout_returns_504() -> None:
    respx.get(f"{FAKE_TG_PREFIX}/status").mock(
        side_effect=httpx.ReadTimeout("slow")
    )
    client = _build_client()

    r = client.get("/api/telegram/status")
    assert r.status_code == 504


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@respx.mock
def test_auth_send_code_ok() -> None:
    route = respx.post(f"{FAKE_TG_PREFIX}/auth/send-code").mock(
        return_value=Response(200, json={"phone_code_hash": "abc"})
    )
    client = _build_client()

    r = client.post("/api/telegram/auth/send-code", json={"phone": "+341234"})

    assert r.status_code == 200
    assert r.json() == {"phone_code_hash": "abc"}
    body = _json.loads(route.calls.last.request.content)
    assert body == {"phone": "+341234"}


def test_auth_send_code_missing_phone_short_circuits() -> None:
    client = _build_client()
    r = client.post("/api/telegram/auth/send-code", json={})
    assert r.status_code == 422


@respx.mock
def test_auth_send_code_backend_4xx_forwarded() -> None:
    respx.post(f"{FAKE_TG_PREFIX}/auth/send-code").mock(
        return_value=Response(400, json={"detail": "invalid phone format"})
    )
    client = _build_client()

    r = client.post("/api/telegram/auth/send-code", json={"phone": "bogus"})
    assert r.status_code == 400
    assert "invalid" in r.json()["detail"]


@respx.mock
def test_auth_sign_in_forwards_phone_code_hash() -> None:
    route = respx.post(f"{FAKE_TG_PREFIX}/auth/sign-in").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = _build_client()

    r = client.post(
        "/api/telegram/auth/sign-in",
        json={"phone": "+1", "code": "12345", "phone_code_hash": "h"},
    )
    assert r.status_code == 200
    body = _json.loads(route.calls.last.request.content)
    assert body == {"phone": "+1", "code": "12345", "phone_code_hash": "h"}


@respx.mock
def test_auth_2fa_ok() -> None:
    respx.post(f"{FAKE_TG_PREFIX}/auth/2fa").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = _build_client()
    r = client.post("/api/telegram/auth/2fa", json={"password": "secret"})
    assert r.status_code == 200


@respx.mock
def test_auth_logout_ok() -> None:
    respx.post(f"{FAKE_TG_PREFIX}/auth/logout").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = _build_client()
    r = client.post("/api/telegram/auth/logout")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Channels + Sync
# ---------------------------------------------------------------------------


@respx.mock
def test_channels_list() -> None:
    body = [{"username": "foo", "title": "Foo Channel"}]
    respx.get(f"{FAKE_TG_PREFIX}/channels").mock(
        return_value=Response(200, json=body)
    )
    client = _build_client()
    r = client.get("/api/telegram/channels")
    assert r.status_code == 200
    assert r.json() == body


@respx.mock
def test_sync_channel_enqueues_and_registers_job() -> None:
    route = respx.post(
        f"{FAKE_TG_PREFIX}/channels/somechannel/sync"
    ).mock(return_value=Response(202, json={"job_id": "job-123"}))

    calls: list[tuple] = []
    client = _build_client(registrar_calls=calls)

    r = client.post(
        "/api/telegram/channels/somechannel/sync", json={"limit": 100}
    )

    assert r.status_code == 202
    assert r.json()["job_id"] == "job-123"
    body = _json.loads(route.calls.last.request.content)
    assert body == {"limit": 100}
    # Se registró un job en el dashboard.
    assert calls, "expected the registrar to be called on success"
    kind, job_id, sse_path, params = calls[0]
    assert kind == "telegram_sync"
    assert job_id == "job-123"
    assert sse_path == "/channels/somechannel/sync/job-123/events"
    assert params == {"username": "somechannel", "channel": "somechannel"}


def test_sync_channel_bad_limit_short_circuits() -> None:
    client = _build_client()
    r = client.post(
        "/api/telegram/channels/foo/sync", json={"limit": "huge"}
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


@respx.mock
def test_media_list_chronological_forwards_query() -> None:
    route = respx.get(f"{FAKE_TG_PREFIX}/media").mock(
        return_value=Response(200, json={"items": [], "total": 0})
    )
    client = _build_client()
    r = client.get(
        "/api/telegram/media",
        params={
            "channel": "foo",
            "view": "chronological",
            "page": 2,
            "page_size": 25,
        },
    )
    assert r.status_code == 200
    q = dict(route.calls.last.request.url.params)
    assert q["view"] == "chronological"
    assert q["channel"] == "foo"
    assert q["page"] == "2"
    assert q["page_size"] == "25"


@respx.mock
def test_media_list_by_author_no_aliases() -> None:
    respx.get(f"{FAKE_TG_PREFIX}/media").mock(
        return_value=Response(200, json={"authors": [{"name": "Ryan", "instructionals": []}]})
    )
    client = _build_client()
    r = client.get("/api/telegram/media", params={"view": "by_author"})
    assert r.status_code == 200
    assert r.json() == {
        "authors": [{"name": "Ryan", "instructionals": []}]
    }


def test_apply_author_aliases_merges_buckets_directly() -> None:
    """Cobertura unitaria de ``_apply_author_aliases`` sobre un dict.

    En el flujo real ``_proxy_json`` devuelve un ``JSONResponse``, así que
    el camino ``if isinstance(data, dict)`` del legacy era dead-code y se
    preserva tal cual (anti-objetivo: no refactorizar la lógica del proxy).
    Este test ejerce el helper directamente para que la fusión por
    alias quede cubierta y documentada.
    """
    svc = TelegramService(
        base_url=FAKE_TG,
        settings_loader=lambda: {"author_aliases": {"danaher": "John Danaher"}},
    )
    out = svc._apply_author_aliases(
        {
            "authors": [
                {"name": "John Danaher", "instructionals": ["a", "b"]},
                {"name": "danaher", "instructionals": ["c"]},
                {"name": "Gordon", "instructionals": ["d"]},
            ]
        }
    )
    by_name = {a["name"]: a for a in out["authors"]}
    assert "John Danaher" in by_name
    assert sorted(by_name["John Danaher"]["instructionals"]) == ["a", "b", "c"]
    assert by_name["Gordon"]["instructionals"] == ["d"]


def test_media_list_bad_view_short_circuits() -> None:
    client = _build_client()
    r = client.get("/api/telegram/media", params={"view": "wat"})
    assert r.status_code == 422


@respx.mock
def test_media_put_metadata_payload_built_from_body() -> None:
    route = respx.put(f"{FAKE_TG_PREFIX}/media/chan/42").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = _build_client()
    r = client.put(
        "/api/telegram/media/chan/42",
        json={"author": "Danaher", "title": "Tripod", "chapter_num": 3},
    )
    assert r.status_code == 200
    body = _json.loads(route.calls.last.request.content)
    assert body == {"author": "Danaher", "title": "Tripod", "chapter_num": 3}


def test_media_put_metadata_empty_body_short_circuits() -> None:
    client = _build_client()
    r = client.put("/api/telegram/media/chan/42", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@respx.mock
def test_download_start_registers_job() -> None:
    respx.post(f"{FAKE_TG_PREFIX}/download").mock(
        return_value=Response(202, json={"job_id": "dl-1"})
    )
    calls: list[tuple] = []
    client = _build_client(registrar_calls=calls)

    r = client.post(
        "/api/telegram/download",
        json={"channel_id": "foo", "author": "A", "title": "T"},
    )
    assert r.status_code == 202
    assert r.json()["job_id"] == "dl-1"
    assert calls
    kind, job_id, sse_path, params = calls[0]
    assert kind == "telegram_download"
    assert job_id == "dl-1"
    assert sse_path == "/download/dl-1/events"
    assert params == {"author": "A", "title": "T", "channel_id": "foo"}


def test_download_start_missing_fields_short_circuits() -> None:
    client = _build_client()
    r = client.post("/api/telegram/download", json={"channel_id": "foo"})
    assert r.status_code == 422


@respx.mock
def test_download_cancel() -> None:
    respx.post(f"{FAKE_TG_PREFIX}/download/dl-1/cancel").mock(
        return_value=Response(200, json={"cancelled": True})
    )
    client = _build_client()
    r = client.post("/api/telegram/download/dl-1/cancel")
    assert r.status_code == 200
    assert r.json()["cancelled"] is True


@respx.mock
def test_download_jobs_list_forwards_status_filter() -> None:
    route = respx.get(f"{FAKE_TG_PREFIX}/download/jobs").mock(
        return_value=Response(200, json=[{"id": "dl-1", "status": "running"}])
    )
    client = _build_client()
    r = client.get("/api/telegram/download/jobs", params={"status": "running"})
    assert r.status_code == 200
    q = dict(route.calls.last.request.url.params)
    assert q.get("status") == "running"


# ---------------------------------------------------------------------------
# SSE proxy
# ---------------------------------------------------------------------------


@respx.mock
def test_sse_download_events_proxy_forwards_payload() -> None:
    payload = (
        b"event: progress\ndata: {\"percent\": 10}\n\n"
        b"event: done\ndata: {}\n\n"
    )
    respx.get(f"{FAKE_TG_PREFIX}/download/dl-1/events").mock(
        return_value=Response(
            200, content=payload, headers={"content-type": "text/event-stream"}
        )
    )
    client = _build_client()
    with client.stream("GET", "/api/telegram/download/dl-1/events") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = b"".join(chunk for chunk in r.iter_bytes())
    assert b"event: progress" in body
    assert b"percent" in body


@respx.mock
def test_sse_sync_events_proxy() -> None:
    respx.get(
        f"{FAKE_TG_PREFIX}/channels/foo/sync/job-1/events"
    ).mock(
        return_value=Response(
            200,
            content=b"event: tick\ndata: 1\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )
    client = _build_client()
    with client.stream(
        "GET", "/api/telegram/channels/foo/sync/job-1/events"
    ) as r:
        assert r.status_code == 200
        body = b"".join(chunk for chunk in r.iter_bytes())
    assert b"event: tick" in body


@respx.mock
def test_sse_backend_error_yields_error_event() -> None:
    respx.get(f"{FAKE_TG_PREFIX}/download/x/events").mock(
        return_value=Response(500, content=b"boom")
    )
    client = _build_client()
    with client.stream("GET", "/api/telegram/download/x/events") as r:
        assert r.status_code == 200  # SSE wrapper siempre devuelve 200
        body = b"".join(chunk for chunk in r.iter_bytes())
    assert b"event: error" in body
