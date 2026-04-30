"""Tests del router del módulo subtitles vía FastAPI TestClient + respx.

El servicio se inyecta vía ``app.dependency_overrides`` con un
``SubtitlesService`` real apuntando a una URL falsa que respx
intercepta. Así verificamos también el path translation y la lógica
de ``translate`` (provider/api_key/fallback).
"""

from __future__ import annotations

from typing import Any

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from ossflow_api.modules.subtitles import subtitles_router
from ossflow_api.modules.subtitles.dependencies import get_subtitles_service
from ossflow_api.modules.subtitles.service import SubtitlesService

FAKE_SUBS = "http://fake-subs"


def _build_client(
    *,
    library_path: str | None = "/host/lib",
    settings: dict[str, Any] | None = None,
) -> TestClient:
    settings = settings or {}

    def _getter(key: str) -> Any:
        return settings.get(key)

    svc = SubtitlesService(
        library_path=library_path,
        subs_url=FAKE_SUBS,
        setting_getter=_getter,
    )
    app = FastAPI()
    app.include_router(subtitles_router)
    app.dependency_overrides[get_subtitles_service] = lambda: svc
    return TestClient(app)


# ---------------------------------------------------------------------------
# /validate
# ---------------------------------------------------------------------------


@respx.mock
def test_validate_translates_path_and_proxies() -> None:
    route = respx.post(f"{FAKE_SUBS}/validate").mock(
        return_value=Response(200, json={"ok": True, "issues": []})
    )
    client = _build_client(library_path="/host/lib")

    r = client.post("/api/subtitles/validate", json={"srt_path": "/host/lib/a.srt"})

    assert r.status_code == 200
    assert r.json() == {"ok": True, "issues": []}
    assert route.called
    sent = route.calls.last.request
    # ``to_container_path`` debería haber traducido /host/lib/a.srt → /media/a.srt.
    import json as _json

    body = _json.loads(sent.content)
    assert body == {"srt_path": "/media/a.srt"}


@respx.mock
def test_validate_propagates_backend_error_status() -> None:
    respx.post(f"{FAKE_SUBS}/validate").mock(
        return_value=Response(503, text="backend boom")
    )
    client = _build_client()

    r = client.post(
        "/api/subtitles/validate", json={"srt_path": "/host/lib/a.srt"}
    )

    assert r.status_code == 503


def test_validate_path_outside_library_returns_400() -> None:
    client = _build_client(library_path="/host/lib")

    r = client.post(
        "/api/subtitles/validate", json={"srt_path": "/elsewhere/a.srt"}
    )

    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /regenerate-segment
# ---------------------------------------------------------------------------


@respx.mock
def test_regenerate_segment_sends_video_path_when_provided() -> None:
    route = respx.post(f"{FAKE_SUBS}/regenerate-segment").mock(
        return_value=Response(200, json={"text": "regenerated"})
    )
    client = _build_client(library_path="/host/lib")

    r = client.post(
        "/api/subtitles/regenerate-segment",
        json={
            "srt_path": "/host/lib/a.srt",
            "segment_idx": 5,
            "video_path": "/host/lib/a.mkv",
            "context_seconds": 2.0,
        },
    )

    assert r.status_code == 200
    assert r.json() == {"text": "regenerated"}
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["srt_path"] == "/media/a.srt"
    assert body["video_path"] == "/media/a.mkv"
    assert body["segment_idx"] == 5
    assert body["context_seconds"] == 2.0
    # Defaults preservados.
    assert body["model"] == "large-v3"
    assert body["language"] == "en"


# ---------------------------------------------------------------------------
# /apply-segment
# ---------------------------------------------------------------------------


@respx.mock
def test_apply_segment_omits_optional_when_missing() -> None:
    route = respx.post(f"{FAKE_SUBS}/apply-segment").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = _build_client(library_path="/host/lib")

    r = client.post(
        "/api/subtitles/apply-segment",
        json={
            "srt_path": "/host/lib/a.srt",
            "segment_idx": 2,
            "text": "hola",
        },
    )

    assert r.status_code == 200
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert "start" not in body
    assert "end" not in body
    assert body["text"] == "hola"


# ---------------------------------------------------------------------------
# /maintenance/*
# ---------------------------------------------------------------------------


@respx.mock
def test_clear_locks_proxies_to_clear_hf_locks() -> None:
    route = respx.post(f"{FAKE_SUBS}/maintenance/clear-hf-locks").mock(
        return_value=Response(200, json={"cleared": 3})
    )
    client = _build_client()

    r = client.post("/api/subtitles/maintenance/clear-locks")

    assert r.status_code == 200
    assert r.json() == {"cleared": 3}
    assert route.called


@respx.mock
def test_restart_proxies_to_maintenance_restart() -> None:
    route = respx.post(f"{FAKE_SUBS}/maintenance/restart").mock(
        return_value=Response(200, json={"restarting": True})
    )
    client = _build_client()

    r = client.post("/api/subtitles/maintenance/restart")

    assert r.status_code == 200
    assert r.json() == {"restarting": True}
    assert route.called


# ---------------------------------------------------------------------------
# /translate — el endpoint con más lógica de negocio
# ---------------------------------------------------------------------------


@respx.mock
def test_translate_ollama_no_api_key_required() -> None:
    route = respx.post(f"{FAKE_SUBS}/translate").mock(
        return_value=Response(200, json={"out": "/media/a.es.srt"})
    )
    client = _build_client(
        library_path="/host/lib",
        settings={"translation_provider": "ollama", "translation_model": "qwen"},
    )

    r = client.post(
        "/api/subtitles/translate",
        json={"srt_path": "/host/lib/a.srt"},
    )

    assert r.status_code == 200, r.text
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["provider"] == "ollama"
    assert body["model"] == "qwen"
    assert "api_key" not in body  # ollama no manda key


@respx.mock
def test_translate_openai_without_api_key_returns_400() -> None:
    client = _build_client(
        library_path="/host/lib",
        settings={"translation_provider": "openai", "openai_api_key": None},
    )

    r = client.post(
        "/api/subtitles/translate",
        json={"srt_path": "/host/lib/a.srt"},
    )

    assert r.status_code == 400
    assert "openai" in r.json()["detail"].lower()


@respx.mock
def test_translate_openai_uses_setting_api_key_and_fallback() -> None:
    route = respx.post(f"{FAKE_SUBS}/translate").mock(
        return_value=Response(200, json={"out": "/media/a.es.srt"})
    )
    client = _build_client(
        library_path="/host/lib",
        settings={
            "translation_provider": "openai",
            "translation_model": "gpt-4o-mini",
            "openai_api_key": "sk-secret",
            "translation_fallback_provider": "ollama",
        },
    )

    r = client.post(
        "/api/subtitles/translate",
        json={"srt_path": "/host/lib/a.srt", "dubbing_mode": True, "dubbing_cps": 18.0},
    )

    assert r.status_code == 200
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["provider"] == "openai"
    assert body["api_key"] == "sk-secret"
    assert body["fallback_provider"] == "ollama"
    assert body["dubbing_mode"] is True
    assert body["dubbing_cps"] == 18.0


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


@respx.mock
def test_analyze_translates_video_path() -> None:
    route = respx.post(f"{FAKE_SUBS}/analyze").mock(
        return_value=Response(200, json={"language": "en", "duration": 60.0})
    )
    client = _build_client(library_path="/host/lib")

    r = client.post(
        "/api/subtitles/analyze",
        json={"video_path": "/host/lib/a.mkv"},
    )

    assert r.status_code == 200
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body["video_path"] == "/media/a.mkv"
    assert body["language"] == "en"
    assert body["model"] == "large-v3"


# ---------------------------------------------------------------------------
# Pasarela bajada (ConnectError → 502)
# ---------------------------------------------------------------------------


@respx.mock
def test_backend_unreachable_returns_502() -> None:
    import httpx

    respx.post(f"{FAKE_SUBS}/validate").mock(
        side_effect=httpx.ConnectError("boom")
    )
    client = _build_client()

    r = client.post(
        "/api/subtitles/validate", json={"srt_path": "/host/lib/a.srt"}
    )

    assert r.status_code == 502


# ---------------------------------------------------------------------------
# Sin library_path: pass-through del path original.
# ---------------------------------------------------------------------------


@respx.mock
def test_validate_passes_through_path_when_no_library_configured() -> None:
    route = respx.post(f"{FAKE_SUBS}/validate").mock(
        return_value=Response(200, json={"ok": True})
    )
    client = _build_client(library_path=None)

    r = client.post(
        "/api/subtitles/validate", json={"srt_path": "/anywhere/file.srt"}
    )

    assert r.status_code == 200
    import json as _json

    body = _json.loads(route.calls.last.request.content)
    assert body == {"srt_path": "/anywhere/file.srt"}
