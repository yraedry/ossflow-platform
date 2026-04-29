"""Skip subtitles/translate steps when chapters already have the artifact.

Mirror of test_pipeline_dub_skip.py for the other two GPU/network-heavy
steps. After ``promote`` removes external sidecars and merges everything
into a multi-track .mkv, the only evidence of subtitles is the embedded
subtitle stream — without these checks the pipeline would re-transcribe
and re-translate from scratch.
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
    _chapter_has_en_subs,
    _chapter_has_es_subs,
    _run_step,
    _season_already_subbed_en,
    _season_already_subbed_es,
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
# _chapter_has_en_subs
# ---------------------------------------------------------------------------

def test_en_subs_via_dot_en_srt(tmp_path):
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    _touch(tmp_path / "Foo - S01E01.en.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_has_en_subs(chapter) is True


def test_en_subs_via_plain_srt(tmp_path):
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    _touch(tmp_path / "Foo - S01E01.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_has_en_subs(chapter) is True


def test_en_subs_via_embedded_stream(tmp_path):
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    with patch("api.library_refresh._probe_track_languages",
               return_value=([], ["eng"])):
        assert _chapter_has_en_subs(chapter) is True


def test_no_en_subs(tmp_path):
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    with patch("api.library_refresh._probe_track_languages",
               return_value=([], ["spa"])):
        assert _chapter_has_en_subs(chapter) is False


# ---------------------------------------------------------------------------
# _chapter_has_es_subs (literal vs dubbing-mode)
# ---------------------------------------------------------------------------

def test_es_subs_literal_via_dot_es_srt(tmp_path):
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    _touch(tmp_path / "Foo - S01E01.es.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_has_es_subs(chapter, dubbing_mode=False) is True


def test_es_subs_literal_does_not_count_dub_es_srt(tmp_path):
    """A .dub.es.srt file is the dubbing-adapted track — it does NOT satisfy
    a request for the literal ES subtitle."""
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    _touch(tmp_path / "Foo - S01E01.dub.es.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_has_es_subs(chapter, dubbing_mode=False) is False


def test_es_subs_dubbing_via_dub_es_srt(tmp_path):
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    _touch(tmp_path / "Foo - S01E01.dub.es.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_has_es_subs(chapter, dubbing_mode=True) is True


def test_es_subs_dubbing_does_not_count_literal_es_srt(tmp_path):
    """Dubbing mode wants the anchor-aware adapted track; the literal track
    won't help — the TTS step needs the .dub.es.srt artifact."""
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    _touch(tmp_path / "Foo - S01E01.es.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        assert _chapter_has_es_subs(chapter, dubbing_mode=True) is False


def test_es_subs_via_embedded_stream_satisfies_either(tmp_path):
    """Embedded ES sub streams come from promote() and only carry the
    literal track — but for skip purposes either request is satisfied
    because the artifact is on disk in some form."""
    chapter = _touch(tmp_path / "Foo - S01E01.mkv")
    with patch("api.library_refresh._probe_track_languages",
               return_value=([], ["spa"])):
        assert _chapter_has_es_subs(chapter, dubbing_mode=False) is True
        assert _chapter_has_es_subs(chapter, dubbing_mode=True) is True


# ---------------------------------------------------------------------------
# _season_already_subbed_*
# ---------------------------------------------------------------------------

def test_season_subbed_en_all(tmp_path):
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E02.mkv")
    _touch(season / "Foo - S01E01.en.srt")
    _touch(season / "Foo - S01E02.en.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        ok, n, total = _season_already_subbed_en(season)
        assert (ok, n, total) == (True, 2, 2)


def test_season_subbed_en_partial(tmp_path):
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E02.mkv")
    _touch(season / "Foo - S01E01.en.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        ok, n, total = _season_already_subbed_en(season)
        assert (ok, n, total) == (False, 1, 2)


def test_season_subbed_es_dub_mode(tmp_path):
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E01.dub.es.srt")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        ok, n, total = _season_already_subbed_es(season, dubbing_mode=True)
        assert (ok, n, total) == (True, 1, 1)


# ---------------------------------------------------------------------------
# _run_step integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_step_subtitles_skipped_when_all_have_en(tmp_path):
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E01.en.srt")

    pipeline = PipelineInfo(
        pipeline_id="ts1",
        path=str(season),
        steps=[StepInfo(name="subtitles")],
        options={},
    )
    pmod._pipelines["ts1"] = pipeline
    queue: asyncio.Queue = asyncio.Queue()

    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        with patch.object(pmod, "_client_and_payload",
                          side_effect=AssertionError("backend should not be called")):
            ok = await _run_step(pipeline, 0, queue)

    assert ok is True
    assert pipeline.steps[0].status == StepStatus.SKIPPED
    assert "subtítulos EN" in pipeline.steps[0].message


@pytest.mark.asyncio
async def test_run_step_translate_skipped_dub_mode(tmp_path):
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E01.dub.es.srt")

    pipeline = PipelineInfo(
        pipeline_id="tt1",
        path=str(season),
        steps=[StepInfo(name="translate")],
        options={"dubbing_mode": True},
    )
    pmod._pipelines["tt1"] = pipeline
    queue: asyncio.Queue = asyncio.Queue()

    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        with patch.object(pmod, "_client_and_payload",
                          side_effect=AssertionError("backend should not be called")):
            ok = await _run_step(pipeline, 0, queue)

    assert ok is True
    assert pipeline.steps[0].status == StepStatus.SKIPPED
    assert "subtítulos ES (dub)" in pipeline.steps[0].message


@pytest.mark.asyncio
async def test_run_step_translate_not_skipped_when_dub_mode_but_only_literal(tmp_path):
    """Literal .es.srt does not satisfy a dubbing_mode=True request."""
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E01.es.srt")

    pipeline = PipelineInfo(
        pipeline_id="tt2",
        path=str(season),
        steps=[StepInfo(name="translate")],
        options={"dubbing_mode": True},
    )
    pmod._pipelines["tt2"] = pipeline
    queue: asyncio.Queue = asyncio.Queue()

    sentinel = RuntimeError("backend-was-called-marker")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        with patch.object(pmod, "_client_and_payload", side_effect=sentinel):
            ok = await _run_step(pipeline, 0, queue)

    assert ok is False
    assert pipeline.steps[0].status == StepStatus.FAILED
    assert "backend-was-called-marker" in pipeline.steps[0].message


@pytest.mark.asyncio
async def test_run_step_subtitles_force_bypasses_skip(tmp_path):
    season = tmp_path / "Season 01"
    _touch(season / "Foo - S01E01.mkv")
    _touch(season / "Foo - S01E01.en.srt")

    pipeline = PipelineInfo(
        pipeline_id="ts2",
        path=str(season),
        steps=[StepInfo(name="subtitles")],
        options={"force": True},
    )
    pmod._pipelines["ts2"] = pipeline
    queue: asyncio.Queue = asyncio.Queue()

    sentinel = RuntimeError("backend-was-called-marker")
    with patch("api.library_refresh._probe_track_languages", return_value=([], [])):
        with patch.object(pmod, "_client_and_payload", side_effect=sentinel):
            ok = await _run_step(pipeline, 0, queue)

    assert ok is False
    assert "backend-was-called-marker" in pipeline.steps[0].message
