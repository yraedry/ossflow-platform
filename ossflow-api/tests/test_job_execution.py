"""Tests for job execution via BackendClient (HTTP, no subprocess).

TDD red -> green: these tests assert that run_chapter_detection /
run_subtitle_generation / run_translation / run_dubbing invoke the
appropriate BackendClient factory and emit SSE events back to the
job queue, and that the pipeline._run_step helper does the same.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from api import app as app_module
from api import pipeline as pipeline_module
from api.app import JobInfo, JobStatus


# ---------------------------------------------------------------------------
# Fake BackendClient
# ---------------------------------------------------------------------------

class FakeBackendClient:
    """Test double capturing ``run`` calls and replaying a scripted stream."""

    def __init__(self, events: list[dict], *, remote_id: str = "remote-1"):
        self._events = events
        self._remote_id = remote_id
        self.run_calls: list[dict] = []
        self.base_url = "http://fake.test"

    async def run(self, payload: dict) -> str:
        self.run_calls.append(payload)
        return self._remote_id

    async def stream(self, job_id: str) -> AsyncIterator[dict]:
        for ev in self._events:
            yield ev


def _install_fake(monkeypatch, attr: str, events: list[dict]) -> FakeBackendClient:
    """Patch the factory on ``api.app`` so run_X picks up our fake."""
    fake = FakeBackendClient(events)
    monkeypatch.setattr(app_module, attr, lambda: fake)
    return fake


async def _drain(job_id: str, *, limit: int = 50) -> list[dict]:
    q = app_module._job_events[job_id]
    out: list[dict] = []
    for _ in range(limit):
        try:
            out.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


def _make_job(job_type: str, path: str = "/videos/sample.mkv") -> JobInfo:
    job = JobInfo(job_id=f"j-{job_type}", job_type=job_type, video_path=path)
    app_module._jobs[job.job_id] = job
    app_module._job_events[job.job_id] = asyncio.Queue()
    return job


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chapter_job_calls_splitter_client_and_emits_events(monkeypatch, tmp_path):
    events = [
        {"status": "running", "progress": 0.2, "message": "scanning"},
        {"status": "done", "progress": 1.0, "result": {"chapters": ["c1", "c2"]}},
    ]
    fake = _install_fake(monkeypatch, "splitter_client", events)
    job = _make_job("chapters", str(tmp_path / "v.mkv"))

    await app_module.run_chapter_detection(job)

    assert len(fake.run_calls) == 1
    assert fake.run_calls[0]["input_path"] == job.video_path
    assert job.status == JobStatus.COMPLETED
    emitted = await _drain(job.job_id)
    assert any(e.get("status") == "running" for e in emitted)
    assert any(e.get("status") == "completed" for e in emitted)


@pytest.mark.asyncio
async def test_subtitle_job_calls_subs_client(monkeypatch, tmp_path):
    events = [{"status": "done", "progress": 1.0, "result": {"log": ["ok"]}}]
    fake = _install_fake(monkeypatch, "subs_client", events)
    job = _make_job("subtitles", str(tmp_path / "v.mkv"))

    await app_module.run_subtitle_generation(job)

    assert len(fake.run_calls) == 1
    assert job.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_dubbing_job_calls_dubbing_client_with_translation_options(monkeypatch, tmp_path):
    events = [{"status": "done", "progress": 1.0, "result": {}}]
    fake = _install_fake(monkeypatch, "dubbing_client", events)
    job = _make_job("dubbing", str(tmp_path / "v.mkv"))

    await app_module.run_dubbing(job, voice_profile="gordon_ryan", use_model_voice=True)

    assert len(fake.run_calls) == 1
    opts = fake.run_calls[0].get("options", {})
    assert opts.get("voice_profile") == "gordon_ryan"
    assert opts.get("use_model_voice") is True
    # Translation is handled by subtitle-generator; dubbing always skips it
    assert opts.get("skip_translation") is True
    assert job.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_translate_job_uses_subs_client_with_openai(monkeypatch, tmp_path):
    """run_translation should hit the subtitle backend with OpenAI translation opts."""
    events = [{"status": "done", "progress": 1.0, "result": {}}]
    fake = _install_fake(monkeypatch, "subs_client", events)
    monkeypatch.setattr(
        "api.settings.get_setting",
        lambda k, default=None: {
            "translation_provider": "openai",
            "translation_model": "gpt-4o-mini",
            "translation_fallback_provider": "",
            "openai_api_key": "sk-test",
        }.get(k, default),
    )
    job = _make_job("translate", str(tmp_path / "v.mkv"))

    await app_module.run_translation(job)

    assert len(fake.run_calls) == 1
    opts = fake.run_calls[0].get("options", {})
    assert opts.get("translate_only") is True
    assert opts.get("provider") == "openai"
    assert opts.get("api_key") == "sk-test"
    assert opts.get("target_lang") == "ES"
    assert job.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_backend_error_marks_job_failed(monkeypatch, tmp_path):
    events = [
        {"status": "running", "progress": 0.1},
        {"status": "failed", "message": "backend blew up"},
    ]
    _install_fake(monkeypatch, "splitter_client", events)
    job = _make_job("chapters", str(tmp_path / "v.mkv"))

    await app_module.run_chapter_detection(job)

    assert job.status == JobStatus.FAILED
    assert "backend" in (job.message or "").lower()


@pytest.mark.asyncio
async def test_job_persisted_on_each_transition(monkeypatch, tmp_path):
    events = [
        {"status": "running", "progress": 0.1},
        {"status": "done", "progress": 1.0, "result": {}},
    ]
    _install_fake(monkeypatch, "subs_client", events)
    job = _make_job("subtitles", str(tmp_path / "v.mkv"))

    persisted: list[str] = []
    original = app_module._persist_job

    def spy(j):
        persisted.append(j.status.value)
        return original(j)

    monkeypatch.setattr(app_module, "_persist_job", spy)
    await app_module.run_subtitle_generation(job)

    assert "running" in persisted
    assert "completed" in persisted


@pytest.mark.asyncio
async def test_backend_error_bjj_kit_shape_marks_job_failed(monkeypatch, tmp_path):
    """New contract: {"type":"error","data":{"message":"boom"}} -> job failed."""
    events = [
        {"type": "progress", "data": {"percent": 0.1, "message": "starting"}},
        {"type": "error", "data": {"message": "boom"}},
    ]
    _install_fake(monkeypatch, "splitter_client", events)
    job = _make_job("chapters", str(tmp_path / "v.mkv"))

    await app_module.run_chapter_detection(job)

    assert job.status == JobStatus.FAILED
    assert "boom" in (job.message or "")


@pytest.mark.asyncio
async def test_progress_events_propagate(monkeypatch, tmp_path):
    """Progress events (new contract) must update job.progress."""
    events = [
        {"type": "progress", "data": {"percent": 0.33, "message": "third"}},
        {"type": "progress", "data": {"percent": 0.66, "message": "twothirds"}},
        {"type": "done", "data": {"result": {}}},
    ]
    _install_fake(monkeypatch, "splitter_client", events)
    job = _make_job("chapters", str(tmp_path / "v.mkv"))

    await app_module.run_chapter_detection(job)

    assert job.status == JobStatus.COMPLETED
    assert job.progress == 100
    emitted = await _drain(job.job_id)
    progresses = [e.get("progress") for e in emitted if "progress" in e]
    # Should include at least one intermediate progress around 33 or 66
    assert any(p is not None and 30 <= p <= 70 for p in progresses), progresses


@pytest.mark.asyncio
async def test_pipeline_runs_3_stages_in_order(monkeypatch, tmp_path):
    """Pipeline should call splitter -> subs -> dubbing in sequence."""
    call_order: list[str] = []

    def make_fake(name: str):
        def factory():
            client = FakeBackendClient([
                {"status": "done", "progress": 1.0, "result": {}},
            ])
            original_run = client.run

            async def tracked_run(payload):
                call_order.append(name)
                return await original_run(payload)

            client.run = tracked_run
            return client
        return factory

    monkeypatch.setattr(pipeline_module, "splitter_client", make_fake("chapters"))
    monkeypatch.setattr(pipeline_module, "subs_client", make_fake("subtitles"))
    monkeypatch.setattr(pipeline_module, "dubbing_client", make_fake("dubbing"))
    # Ensure no stale library_path from previous tests pollutes path mapping
    monkeypatch.setattr(pipeline_module, "get_library_path", lambda: "")

    video = tmp_path / "v.mkv"
    video.write_bytes(b"x")

    pipeline = pipeline_module.PipelineInfo(
        pipeline_id="p1",
        path=str(video),
        steps=[
            pipeline_module.StepInfo(name="chapters"),
            pipeline_module.StepInfo(name="subtitles"),
            pipeline_module.StepInfo(name="dubbing"),
        ],
        options={},
    )
    queue: asyncio.Queue = asyncio.Queue()
    await pipeline_module._run_pipeline(pipeline, queue)

    # Drain queue for debugging on failure
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert call_order == ["chapters", "subtitles", "dubbing"], f"queue={drained}"
    assert pipeline.status == pipeline_module.StepStatus.COMPLETED
