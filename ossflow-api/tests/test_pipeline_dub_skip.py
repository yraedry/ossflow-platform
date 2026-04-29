"""Skip dubbing step when the Season is already dubbed.

After ``promote`` merges the dubbed audio as a second track and removes
``doblajes/``, the only evidence of dubbing is the embedded ES audio stream.
Without this skip, re-running the dubbing step on a promoted Season would
re-dub from scratch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from api import pipeline as pmod
from api.pipeline import (
    PipelineInfo,
    StepInfo,
    StepStatus,
    _chapter_is_dubbed,
    _run_step,
    _season_already_dubbed,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(pmod, "HISTORY_FILE", tmp_path / "pipeline_history.json")
    snap = dict(pmod._pipelines)
    pmod._pipelines.clear()
    yield
    pmod._pipelines.clear()
    pmod._pipelines.update(snap)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


# ---------------------------------------------------------------------------
# _chapter_is_dubbed
# ---------------------------------------------------------------------------

def test_chapter_dubbed_via_legacy_DOBLADO_sidecar(tmp_path):
    season = tmp_path / "Season 01"
    chapter = _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E01_DOBLADO.mkv")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_is_dubbed(chapter) is True


def test_chapter_dubbed_via_doblajes_folder(tmp_path):
    season = tmp_path / "Season 01"
    chapter = _touch(season / "Foo - S01E01.mkv")
    _touch(season / "doblajes" / "Foo - S01E01.mkv")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_is_dubbed(chapter) is True


def test_chapter_dubbed_via_elevenlabs_folder(tmp_path):
    season = tmp_path / "Season 01"
    chapter = _touch(season / "Foo - S01E01.mkv")
    _touch(season / "elevenlabs" / "Foo - S01E01.mkv")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_is_dubbed(chapter) is True


def test_chapter_dubbed_via_embedded_audio_track(tmp_path):
    """Post-promote: only the ES audio stream tag remains as evidence."""
    season = tmp_path / "Season 01"
    chapter = _touch(season / "Foo - S01E01.mkv")
    with patch("api.library_refresh._probe_track_languages",
               return_value=(["eng", "spa"], [])):
        assert _chapter_is_dubbed(chapter) is True


def test_chapter_not_dubbed_when_only_english_audio(tmp_path):
    season = tmp_path / "Season 01"
    chapter = _touch(season / "Foo - S01E01.mkv")
    with patch("api.library_refresh._probe_track_languages",
               return_value=(["eng"], [])):
        assert _chapter_is_dubbed(chapter) is False


# ---------------------------------------------------------------------------
# _season_already_dubbed
# ---------------------------------------------------------------------------

def test_season_all_dubbed(tmp_path):
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E02.mkv")
    with patch("api.library_refresh._probe_track_languages",
               return_value=(["eng", "spa"], [])):
        all_d, dubbed, total = _season_already_dubbed(season)
        assert all_d is True
        assert dubbed == 2
        assert total == 2


def test_season_partial_dubbed_returns_false(tmp_path):
    season = tmp_path / "Season 01"
    c1 = _touch(season / "Foo - S01E01.mkv")
    c2 = _touch(season / "Foo - S01E02.mkv")

    def fake_probe(path: Path):
        # Only c1 has spanish; c2 doesn't
        if path == c1:
            return ["eng", "spa"], []
        return ["eng"], []

    with patch("api.library_refresh._probe_track_languages", side_effect=fake_probe):
        all_d, dubbed, total = _season_already_dubbed(season)
        assert all_d is False
        assert dubbed == 1
        assert total == 2


def test_season_with_no_chapters_returns_false(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    # File without S01E01 pattern is ignored
    _touch(season / "trailer.mkv")
    all_d, dubbed, total = _season_already_dubbed(season)
    assert all_d is False
    assert total == 0


def test_season_nonexistent_returns_false(tmp_path):
    all_d, dubbed, total = _season_already_dubbed(tmp_path / "missing")
    assert (all_d, dubbed, total) == (False, 0, 0)


# ---------------------------------------------------------------------------
# _run_step integration: dubbing step skips when already dubbed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_step_dubbing_skipped_when_all_chapters_dubbed(tmp_path):
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E02.mkv")

    pipeline = PipelineInfo(
        pipeline_id="test1",
        path=str(season),
        steps=[StepInfo(name="dubbing")],
        options={},
    )
    pmod._pipelines["test1"] = pipeline
    queue: asyncio.Queue = asyncio.Queue()

    # Fake ffprobe -> all chapters have spanish audio
    with patch("api.library_refresh._probe_track_languages",
               return_value=(["eng", "spa"], [])):
        # Backend client must NOT be hit
        with patch.object(pmod, "_client_and_payload",
                          side_effect=AssertionError("backend should not be called")):
            ok = await _run_step(pipeline, 0, queue)

    assert ok is True
    assert pipeline.steps[0].status == StepStatus.SKIPPED
    assert "ya tienen audio ES" in pipeline.steps[0].message


@pytest.mark.asyncio
async def test_run_step_dubbing_not_skipped_when_force_true(tmp_path):
    """force=true bypasses the skip and proceeds to call the backend.

    _run_step swallows generic exceptions and marks the step FAILED, so we
    verify "backend was called" by asserting the failure message rather than
    expecting a raise to bubble out.
    """
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")

    pipeline = PipelineInfo(
        pipeline_id="test2",
        path=str(season),
        steps=[StepInfo(name="dubbing")],
        options={"force": True},
    )
    pmod._pipelines["test2"] = pipeline
    queue: asyncio.Queue = asyncio.Queue()

    sentinel = RuntimeError("backend-was-called-marker")
    with patch("api.library_refresh._probe_track_languages",
               return_value=(["eng", "spa"], [])):
        with patch.object(pmod, "_client_and_payload", side_effect=sentinel):
            ok = await _run_step(pipeline, 0, queue)

    assert ok is False
    assert pipeline.steps[0].status == StepStatus.FAILED
    assert "backend-was-called-marker" in pipeline.steps[0].message


@pytest.mark.asyncio
async def test_run_step_dubbing_not_skipped_when_chapters_not_dubbed(tmp_path):
    """When chapters lack ES audio, dubbing proceeds normally."""
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")

    pipeline = PipelineInfo(
        pipeline_id="test3",
        path=str(season),
        steps=[StepInfo(name="dubbing")],
        options={},
    )
    pmod._pipelines["test3"] = pipeline
    queue: asyncio.Queue = asyncio.Queue()

    sentinel = RuntimeError("backend-was-called-marker")
    with patch("api.library_refresh._probe_track_languages",
               return_value=(["eng"], [])):
        with patch.object(pmod, "_client_and_payload", side_effect=sentinel):
            ok = await _run_step(pipeline, 0, queue)

    assert ok is False
    assert pipeline.steps[0].status == StepStatus.FAILED
    assert "backend-was-called-marker" in pipeline.steps[0].message
