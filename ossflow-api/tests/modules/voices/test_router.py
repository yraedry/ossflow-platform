"""Tests del módulo voices.

Usan ``app.dependency_overrides`` con un manager fake para no tocar el
filesystem real (``voice_profiles/samples/`` ni ``registry.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.voices import voices_router
from ossflow_api.modules.voices.dependencies import get_voices_service
from ossflow_api.modules.voices.service import VoicesService


@dataclass
class _FakeProfile:
    instructor: str
    sample_path: str
    duration_seconds: float = 15.0
    created_at: str = "2026-04-30T00:00:00Z"

    def to_dict(self):
        return {
            "instructor": self.instructor,
            "sample_path": self.sample_path,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
        }


class _FakeManager:
    """Manager fake con la misma superficie pública que ``VoiceProfileManager``."""

    def __init__(self) -> None:
        self.profiles: dict[str, _FakeProfile] = {}

    def list_profiles(self) -> list[_FakeProfile]:
        return list(self.profiles.values())

    def extract_sample(
        self,
        video_path: Path,
        instructor: str,
        *,
        start_sec: float = 60.0,
        duration: float = 15.0,
    ) -> _FakeProfile:
        prof = _FakeProfile(
            instructor=instructor,
            sample_path=f"/fake/samples/{instructor}.wav",
            duration_seconds=duration,
        )
        self.profiles[instructor] = prof
        return prof

    def delete_profile(self, instructor: str) -> bool:
        return self.profiles.pop(instructor, None) is not None


@pytest.fixture
def env(tmp_path):
    fake_mgr = _FakeManager()
    svc = VoicesService(manager_factory=lambda: fake_mgr)

    app = FastAPI()
    app.include_router(voices_router)
    app.dependency_overrides[get_voices_service] = lambda: svc

    # Vídeo dummy que pasa el ``Path(video_path).exists()``.
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00")

    return {
        "client": TestClient(app),
        "manager": fake_mgr,
        "video_path": str(video),
    }


def test_list_empty(env):
    r = env["client"].get("/api/voice-profiles")
    assert r.status_code == 200
    assert r.json() == {"profiles": []}


def test_post_creates_profile(env):
    r = env["client"].post(
        "/api/voice-profiles",
        json={
            "video_path": env["video_path"],
            "instructor": "John",
            "start_sec": 30,
            "duration": 10,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["profile"]["instructor"] == "John"
    assert body["profile"]["duration_seconds"] == 10.0


def test_post_missing_fields_returns_400(env):
    r = env["client"].post("/api/voice-profiles", json={"instructor": ""})
    assert r.status_code == 400
    assert "video_path" in r.json()["error"] or "instructor" in r.json()["error"]


def test_post_video_not_found_returns_404(env):
    r = env["client"].post(
        "/api/voice-profiles",
        json={"video_path": "/no/such/file.mp4", "instructor": "John"},
    )
    assert r.status_code == 404


def test_delete_existing(env):
    env["manager"].profiles["John"] = _FakeProfile(
        instructor="John", sample_path="/x"
    )
    r = env["client"].delete("/api/voice-profiles/John")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "John" not in env["manager"].profiles


def test_delete_missing_returns_404(env):
    r = env["client"].delete("/api/voice-profiles/Nope")
    assert r.status_code == 404
