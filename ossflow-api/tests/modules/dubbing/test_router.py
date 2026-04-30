"""Tests del router del módulo dubbing vía FastAPI TestClient + respx.

El servicio se inyecta vía ``app.dependency_overrides`` con un
``DubbingService`` real apuntando a una URL falsa que respx
intercepta. Así verificamos también el path translation, el
fallback a ``voice_profile_loader`` y los endpoints que leen disco.
"""

from __future__ import annotations

import json as _json
from typing import Any, Optional

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from ossflow_api.modules.dubbing import dubbing_router
from ossflow_api.modules.dubbing.dependencies import get_dubbing_service
from ossflow_api.modules.dubbing.service import DubbingService

FAKE_DUB = "http://fake-dub"


def _build_client(
    *,
    library_path: str | None = "/host/lib",
    voice_profile: str = "",
    scan_cache: Optional[dict] = None,
) -> TestClient:
    svc = DubbingService(
        library_path=library_path,
        dubbing_url=FAKE_DUB,
        voice_profile_loader=lambda _path: voice_profile,
        scan_cache_loader=lambda: scan_cache,
    )
    app = FastAPI()
    app.include_router(dubbing_router)
    app.dependency_overrides[get_dubbing_service] = lambda: svc
    return TestClient(app)


# ---------------------------------------------------------------------------
# /voices
# ---------------------------------------------------------------------------


@respx.mock
def test_list_voices_proxies_to_backend() -> None:
    route = respx.get(f"{FAKE_DUB}/voices").mock(
        return_value=Response(200, json={"voices": ["a.wav", "b.wav"]})
    )
    client = _build_client()

    r = client.get("/api/dubbing/voices")

    assert r.status_code == 200
    assert r.json() == {"voices": ["a.wav", "b.wav"]}
    assert route.called


# ---------------------------------------------------------------------------
# /voices/{filename}/transcript
# ---------------------------------------------------------------------------


@respx.mock
def test_save_voice_transcript_proxies_payload() -> None:
    route = respx.put(f"{FAKE_DUB}/voices/sample.wav/transcript").mock(
        return_value=Response(200, json={"saved": True})
    )
    client = _build_client()

    r = client.put(
        "/api/dubbing/voices/sample.wav/transcript",
        json={"transcript": "hola mundo"},
    )

    assert r.status_code == 200
    assert r.json() == {"saved": True}
    body = _json.loads(route.calls.last.request.content)
    assert body == {"transcript": "hola mundo"}


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


@respx.mock
def test_analyze_translates_paths_and_uses_voice_profile_fallback() -> None:
    route = respx.post(f"{FAKE_DUB}/analyze").mock(
        return_value=Response(200, json={"phrases": []})
    )
    client = _build_client(library_path="/host/lib", voice_profile="alex")

    r = client.post(
        "/api/dubbing/analyze",
        json={
            "video_path": "/host/lib/a.mkv",
            "srt_path": "/host/lib/a.srt",
            "max_phrases": 3,
        },
    )

    assert r.status_code == 200, r.text
    body = _json.loads(route.calls.last.request.content)
    assert body["video_path"] == "/media/a.mkv"
    assert body["srt_path"] == "/media/a.srt"
    assert body["synthesize"] is False
    assert body["max_phrases"] == 3
    # Sin voice_profile en el body, cae al loader inyectado.
    assert body["voice_profile"] == "alex"


@respx.mock
def test_analyze_body_voice_profile_overrides_loader() -> None:
    route = respx.post(f"{FAKE_DUB}/analyze").mock(
        return_value=Response(200, json={"phrases": []})
    )
    client = _build_client(library_path="/host/lib", voice_profile="from_sidecar")

    r = client.post(
        "/api/dubbing/analyze",
        json={"video_path": "/host/lib/a.mkv", "voice_profile": "explicit"},
    )

    assert r.status_code == 200
    body = _json.loads(route.calls.last.request.content)
    assert body["voice_profile"] == "explicit"


def test_analyze_path_outside_library_returns_400() -> None:
    client = _build_client(library_path="/host/lib")

    r = client.post(
        "/api/dubbing/analyze", json={"video_path": "/elsewhere/a.mkv"}
    )

    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /maintenance/restart
# ---------------------------------------------------------------------------


@respx.mock
def test_restart_proxies_to_backend() -> None:
    route = respx.post(f"{FAKE_DUB}/maintenance/restart").mock(
        return_value=Response(200, json={"restarting": True})
    )
    client = _build_client()

    r = client.post("/api/dubbing/maintenance/restart")

    assert r.status_code == 200
    assert r.json() == {"restarting": True}
    assert route.called


@respx.mock
def test_restart_swallows_connection_drop_as_success() -> None:
    """El backend mata su PID 1 ~0.5s tras responder; aceptamos la caída."""
    import httpx

    respx.post(f"{FAKE_DUB}/maintenance/restart").mock(
        side_effect=httpx.ConnectError("container went down")
    )
    client = _build_client()

    r = client.post("/api/dubbing/maintenance/restart")

    assert r.status_code == 200
    assert r.json()["ok"] is True


# ---------------------------------------------------------------------------
# /qa (lee sidecar de disco)
# ---------------------------------------------------------------------------


def test_get_dub_qa_returns_sidecar_when_present(tmp_path) -> None:
    video = tmp_path / "chapter.mkv"
    video.touch()
    sidecar = tmp_path / "chapter.dub-qa.json"
    sidecar.write_text(_json.dumps({"verdict": {"level": "green", "mos": 4.2}}))

    client = _build_client(library_path=None)

    r = client.get("/api/dubbing/qa", params={"video_path": str(video)})

    assert r.status_code == 200
    assert r.json() == {"verdict": {"level": "green", "mos": 4.2}}


def test_get_dub_qa_returns_404_when_missing(tmp_path) -> None:
    video = tmp_path / "chapter.mkv"
    video.touch()
    client = _build_client(library_path=None)

    r = client.get("/api/dubbing/qa", params={"video_path": str(video)})

    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /qa/instructional/{name}
# ---------------------------------------------------------------------------


def test_get_instructional_qa_404_when_no_cache() -> None:
    client = _build_client(scan_cache=None)

    r = client.get("/api/dubbing/qa/instructional/MyCourse")

    assert r.status_code == 404


def test_get_instructional_qa_aggregates_chapters(tmp_path) -> None:
    v1 = tmp_path / "ep1.mkv"
    v2 = tmp_path / "ep2.mkv"
    v1.touch()
    v2.touch()
    (tmp_path / "ep1.dub-qa.json").write_text(
        _json.dumps({"verdict": {"level": "green", "mos": 4.5}})
    )
    (tmp_path / "ep2.dub-qa.json").write_text(
        _json.dumps({"verdict": {"level": "amber", "mos": 3.0}})
    )

    cache = {
        "instructionals": [
            {
                "name": "MyCourse",
                "videos": [
                    {"filename": "ep1.mkv", "path": str(v1), "has_dubbing": True},
                    {"filename": "ep2.mkv", "path": str(v2), "has_dubbing": True},
                ],
            }
        ]
    }
    client = _build_client(scan_cache=cache)

    r = client.get("/api/dubbing/qa/instructional/MyCourse")

    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "MyCourse"
    assert body["summary"]["total_chapters"] == 2
    assert body["summary"]["with_qa"] == 2
    assert body["summary"]["levels"] == {"green": 1, "amber": 1, "red": 0}
    assert body["summary"]["avg_mos"] == 3.75
    assert body["summary"]["worst"] == {"filename": "ep2.mkv", "mos": 3.0}
    assert len(body["chapters"]) == 2


# ---------------------------------------------------------------------------
# Backend caído → 502
# ---------------------------------------------------------------------------


@respx.mock
def test_voices_backend_unreachable_returns_502() -> None:
    import httpx

    respx.get(f"{FAKE_DUB}/voices").mock(side_effect=httpx.ConnectError("boom"))
    client = _build_client()

    r = client.get("/api/dubbing/voices")

    assert r.status_code == 502
