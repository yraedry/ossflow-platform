"""Tests for api.elevenlabs_dubbing + api.elevenlabs_dubbing_client.

We mock the ElevenLabs SDK at import time via monkeypatch so the tests
never actually hit the network. Two layers:

  1. ``ElevenLabsDubbingClient`` — thin SDK wrapper, verified in isolation.
  2. ``_run_elevenlabs_dubbing`` — async orchestration, verified via a
     fake client that returns a scripted status transition.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from api import elevenlabs_dubbing_client as client_module
from api.elevenlabs_dubbing_client import (
    DubbingJob,
    ElevenLabsDubbingClient,
    ElevenLabsDubbingError,
    resolve_output_path,
)


# ---------------------------------------------------------------------------
# resolve_output_path
# ---------------------------------------------------------------------------

def test_resolve_output_path_creates_sibling_elevenlabs_dir(tmp_path):
    season = tmp_path / "Instructional" / "Season 01"
    season.mkdir(parents=True)
    source = season / "S01E02 - Foo.mp4"
    source.write_bytes(b"stub")

    out = resolve_output_path(source)

    assert out == season / "elevenlabs" / "S01E02 - Foo.mp4"
    assert out.parent.is_dir()


def test_resolve_output_path_keeps_original_filename(tmp_path):
    season = tmp_path / "Season 02"
    season.mkdir()
    source = season / "Anything with - dashes.mkv"
    source.write_bytes(b"")

    out = resolve_output_path(source)

    assert out.name == "Anything with - dashes.mkv"
    assert out.name == source.name  # no suffix added


# ---------------------------------------------------------------------------
# ElevenLabsDubbingClient
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a fake ``elevenlabs.client.ElevenLabs`` on sys.modules."""
    fake_client_cls = MagicMock()
    fake_instance = MagicMock()
    fake_client_cls.return_value = fake_instance

    import types
    elevenlabs_mod = types.ModuleType("elevenlabs")
    elevenlabs_client_mod = types.ModuleType("elevenlabs.client")
    elevenlabs_client_mod.ElevenLabs = fake_client_cls
    elevenlabs_mod.client = elevenlabs_client_mod
    monkeypatch.setitem(sys.modules, "elevenlabs", elevenlabs_mod)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", elevenlabs_client_mod)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test_fake")

    return fake_instance


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ElevenLabsDubbingError, match="ELEVENLABS_API_KEY"):
        ElevenLabsDubbingClient()


def test_client_start_returns_dubbing_id(fake_sdk, tmp_path):
    resp = MagicMock(dubbing_id="dub_abc123", status="created")
    fake_sdk.dubbing.create.return_value = resp

    client = ElevenLabsDubbingClient()
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    with f.open("rb") as fh:
        job = client.start(
            file=fh,
            filename="v.mp4",
            target_lang="es",
            source_lang="en",
            num_speakers=1,
            watermark=True,
            name="v",
        )

    assert isinstance(job, DubbingJob)
    assert job.dubbing_id == "dub_abc123"
    assert job.status == "created"
    call = fake_sdk.dubbing.create.call_args
    assert call.kwargs["source_lang"] == "en"
    assert call.kwargs["target_lang"] == "es"
    assert call.kwargs["watermark"] is True
    assert call.kwargs["num_speakers"] == 1


def test_client_start_raises_when_sdk_fails(fake_sdk, tmp_path):
    fake_sdk.dubbing.create.side_effect = RuntimeError("429 rate limit")
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    with pytest.raises(ElevenLabsDubbingError, match="dubbing.create failed"):
        with f.open("rb") as fh:
            ElevenLabsDubbingClient().start(
                file=fh, filename="v.mp4", target_lang="es",
            )


def test_client_start_raises_without_dubbing_id(fake_sdk, tmp_path):
    fake_sdk.dubbing.create.return_value = MagicMock(spec=["status"], status="created")
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    with pytest.raises(ElevenLabsDubbingError, match="no dubbing_id"):
        with f.open("rb") as fh:
            ElevenLabsDubbingClient().start(
                file=fh, filename="v.mp4", target_lang="es",
            )


def test_client_poll_returns_status(fake_sdk):
    fake_sdk.dubbing.get.return_value = MagicMock(status="dubbed")
    job = ElevenLabsDubbingClient().poll("dub_abc123")
    assert job.dubbing_id == "dub_abc123"
    assert job.status == "dubbed"


def test_client_download_concatenates_bytes(fake_sdk):
    fake_sdk.dubbing.audio.get.return_value = iter([b"AAA", b"BBB", b"CC"])
    data = ElevenLabsDubbingClient().download("dub_abc123", "es")
    assert data == b"AAABBBCC"


def test_client_download_returns_bytes_directly(fake_sdk):
    fake_sdk.dubbing.audio.get.return_value = b"RAWBYTES"
    data = ElevenLabsDubbingClient().download("dub_abc123", "es")
    assert data == b"RAWBYTES"


def test_client_download_raises_on_empty(fake_sdk):
    fake_sdk.dubbing.audio.get.return_value = iter([])
    with pytest.raises(ElevenLabsDubbingError, match="no bytes"):
        ElevenLabsDubbingClient().download("dub_abc123", "es")


# ---------------------------------------------------------------------------
# _run_elevenlabs_dubbing orchestration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _run_elevenlabs_dubbing orchestration
# ---------------------------------------------------------------------------
#
# The orchestrator late-imports ``api.app`` via ``_job_host()`` to avoid
# a circular import at module load. In unit tests we *don't* want to pull
# in ``api.app`` either (it drags SQLAlchemy / bjj_service_kit, which is
# only installed inside the container). We fake the host registry with a
# minimal stand-in that exposes the same three attributes the
# orchestrator touches: ``_jobs``, ``_job_events``, ``_persist_job``,
# plus ``JobInfo`` and ``JobStatus``.


from dataclasses import dataclass, field
from enum import Enum


class _StubJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class _StubJobInfo:
    job_id: str
    job_type: str
    video_path: str
    status: _StubJobStatus = _StubJobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    completed_at: str | None = None
    result: dict | None = None


class _StubHost:
    def __init__(self):
        self._jobs: dict = {}
        self._job_events: dict = {}
        self.JobStatus = _StubJobStatus
        self.JobInfo = _StubJobInfo

    def _persist_job(self, job):
        pass  # no-op in tests


@pytest.fixture
def patched_client(monkeypatch):
    """Replace ElevenLabsDubbingClient in the module with a scripted fake
    and stub ``_job_host`` so tests never import ``api.app``.
    """
    from api import elevenlabs_dubbing as mod

    class ScriptedClient:
        statuses = ["dubbing", "dubbing", "dubbed"]

        def __init__(self): pass

        def start(self, **kw):
            return DubbingJob(dubbing_id="dub_xyz", status="created")

        def poll(self, dubbing_id):
            nxt = self.statuses.pop(0) if self.statuses else "dubbed"
            return DubbingJob(dubbing_id=dubbing_id, status=nxt)

        def download(self, dubbing_id, target_lang):
            return b"FAKE_MP4_BYTES"

    host = _StubHost()

    monkeypatch.setattr(mod, "ElevenLabsDubbingClient", ScriptedClient)
    monkeypatch.setattr(mod, "_POLL_INTERVAL_S", 0)  # no sleep in tests
    monkeypatch.setattr(mod, "_job_host", lambda: host)
    return mod, host


def test_orchestration_happy_path(tmp_path, patched_client):
    """Full flow: start → 2 polls → download → file on disk."""
    mod, host = patched_client

    source = tmp_path / "Season 01" / "S01E02 - Foo.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"SRC")

    job_id = "testjob1"
    job = host.JobInfo(
        job_id=job_id,
        job_type="elevenlabs_dubbing",
        video_path=str(source),
    )
    host._jobs[job_id] = job
    host._job_events[job_id] = asyncio.Queue()

    asyncio.run(
        mod._run_elevenlabs_dubbing(
            job_id, source,
            source_lang="en", target_lang="es",
            num_speakers=1, watermark=True,
        )
    )

    assert job.status.value == "completed"
    assert job.progress == 100
    output = source.parent / "elevenlabs" / source.name
    assert output.exists()
    assert output.read_bytes() == b"FAKE_MP4_BYTES"
    assert job.result["dubbing_id"] == "dub_xyz"
    assert job.result["provider"] == "elevenlabs"
    assert job.result["output_path"] == str(output)


def test_orchestration_failure_sets_failed(monkeypatch, tmp_path, patched_client):
    """A 'failed' status from ElevenLabs should flip the job to FAILED."""
    mod, host = patched_client

    class FailingClient:
        def __init__(self): pass
        def start(self, **kw): return DubbingJob("dub_x", "created")
        def poll(self, dubbing_id): return DubbingJob(dubbing_id, "failed")
        def download(self, *a, **k): raise AssertionError("should not reach")

    monkeypatch.setattr(mod, "ElevenLabsDubbingClient", FailingClient)

    source = tmp_path / "v.mp4"
    source.write_bytes(b"SRC")

    job_id = "testjob2"
    job = host.JobInfo(
        job_id=job_id,
        job_type="elevenlabs_dubbing",
        video_path=str(source),
    )
    host._jobs[job_id] = job
    host._job_events[job_id] = asyncio.Queue()

    asyncio.run(
        mod._run_elevenlabs_dubbing(
            job_id, source,
            source_lang="en", target_lang="es",
            num_speakers=1, watermark=True,
        )
    )

    assert job.status.value == "failed"
    assert "failed" in (job.message or "").lower()


# ---------------------------------------------------------------------------
# Resume on startup
# ---------------------------------------------------------------------------

def test_resume_without_dubbing_id_marks_failed(tmp_path, patched_client):
    """A zombie job (running, no dubbing_id) must be marked failed on startup."""
    mod, host = patched_client

    source = tmp_path / "v.mp4"
    source.write_bytes(b"SRC")
    job = host.JobInfo(
        job_id="zombie1",
        job_type="elevenlabs_dubbing",
        video_path=str(source),
    )
    job.status = host.JobStatus.RUNNING
    job.result = {"provider": "elevenlabs"}  # no dubbing_id
    host._jobs["zombie1"] = job

    summary = mod.resume_orphan_jobs()

    assert "zombie1" in summary["failed"]
    assert job.status.value == "failed"
    assert "lost on container restart" in (job.message or "")


def test_resume_missing_source_video_marks_failed(tmp_path, patched_client):
    """Job with dubbing_id but source file gone → failed (we can't write output)."""
    mod, host = patched_client

    job = host.JobInfo(
        job_id="gone1",
        job_type="elevenlabs_dubbing",
        video_path=str(tmp_path / "does_not_exist.mp4"),
    )
    job.status = host.JobStatus.RUNNING
    job.result = {"dubbing_id": "dub_gone"}
    host._jobs["gone1"] = job

    summary = mod.resume_orphan_jobs()

    assert "gone1" in summary["failed"]
    assert job.status.value == "failed"
    assert "source video missing" in (job.message or "")


def test_resume_with_dubbing_id_schedules_task(tmp_path, patched_client):
    """Running job with dubbing_id + existing source → resume task is scheduled."""
    mod, host = patched_client

    source = tmp_path / "v.mp4"
    source.write_bytes(b"SRC")
    job = host.JobInfo(
        job_id="resume1",
        job_type="elevenlabs_dubbing",
        video_path=str(source),
    )
    job.status = host.JobStatus.RUNNING
    job.result = {"dubbing_id": "dub_abc", "estimated_total_sec": 120}
    host._jobs["resume1"] = job

    # asyncio.create_task needs a running loop. Drive the call inside
    # a short-lived loop and then cancel pending tasks so the test exits.
    async def _drive():
        summary = mod.resume_orphan_jobs()
        # Give the resume task one tick to register itself
        await asyncio.sleep(0)
        # Cancel any tasks the resume scheduled so we don't hang
        for t in asyncio.all_tasks():
            if t is asyncio.current_task():
                continue
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        return summary

    summary = asyncio.run(_drive())
    assert "resume1" in summary["resumed"]
    assert "resume1" not in summary["failed"]
