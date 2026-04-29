"""Strict mode for ``DubbingPipeline.process_directory``.

Before 2026-04-29: a chapter without ES SRT was logged WARNING + skipped,
and a chapter whose ``process_file`` raised was logged + swallowed. The
return value was the (possibly partial) list of dubbed paths and the
backend reported the job as completed even when only 4/9 chapters had
actually been dubbed. The user only noticed because the multi-track .mkv
was missing several episodes.

The strict version raises ``RuntimeError`` listing the missing/failed
chapters so the orchestrator marks the step FAILED.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SVC_ROOT = Path(__file__).resolve().parents[1]
if str(_SVC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SVC_ROOT))

from dubbing_generator.pipeline import DubbingPipeline  # noqa: E402


def _make_pipeline(extensions=(".mkv",)) -> DubbingPipeline:
    """Return a DubbingPipeline whose ctor side effects are bypassed.

    The real __init__ instantiates Demucs and a synthesizer; for these
    tests we only need ``process_directory`` to walk the tree and call
    ``process_file``. We bypass __init__ via ``__new__`` and stub the
    config attribute it reads.
    """
    pipeline = DubbingPipeline.__new__(DubbingPipeline)
    pipeline.cfg = MagicMock()
    pipeline.cfg.extensions = extensions
    return pipeline


def _make_chapter_with_srt(season: Path, name: str) -> Path:
    season.mkdir(parents=True, exist_ok=True)
    video = season / f"{name}.mkv"
    srt = season / f"{name}.es.srt"
    video.write_bytes(b"x")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhola\n", encoding="utf-8")
    return video


def _make_chapter_no_srt(season: Path, name: str) -> Path:
    season.mkdir(parents=True, exist_ok=True)
    video = season / f"{name}.mkv"
    video.write_bytes(b"x")
    return video


# ---------------------------------------------------------------------------
# Happy path: all chapters processed
# ---------------------------------------------------------------------------

def test_process_directory_returns_results_when_all_succeed(tmp_path):
    season = tmp_path / "Season 01"
    v1 = _make_chapter_with_srt(season, "Foo - S01E01")
    v2 = _make_chapter_with_srt(season, "Foo - S01E02")

    pipeline = _make_pipeline()
    pipeline.process_file = MagicMock(
        side_effect=lambda video, srt: video.with_name(f"{video.stem}_DOBLADO.mkv")
    )

    out = pipeline.process_directory(season)

    assert len(out) == 2
    assert pipeline.process_file.call_count == 2


# ---------------------------------------------------------------------------
# Strict: missing SRTs raise
# ---------------------------------------------------------------------------

def test_process_directory_raises_when_chapter_lacks_srt(tmp_path):
    season = tmp_path / "Season 01"
    _make_chapter_with_srt(season, "Foo - S01E01")
    _make_chapter_no_srt(season, "Foo - S01E02")  # SRT missing

    pipeline = _make_pipeline()
    pipeline.process_file = MagicMock(
        side_effect=lambda video, srt: video.with_name(f"{video.stem}_DOBLADO.mkv")
    )

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process_directory(season)

    msg = str(excinfo.value)
    assert "1/2" in msg  # 1 dubbed of 2 total
    assert "sin SRT ES" in msg
    assert "Foo - S01E02.mkv" in msg


# ---------------------------------------------------------------------------
# Strict: TTS failures raise (no longer swallowed)
# ---------------------------------------------------------------------------

def test_process_directory_raises_when_process_file_fails(tmp_path):
    season = tmp_path / "Season 01"
    _make_chapter_with_srt(season, "Foo - S01E01")
    v2 = _make_chapter_with_srt(season, "Foo - S01E02")

    pipeline = _make_pipeline()

    def _maybe_fail(video, srt):
        if video == v2:
            raise RuntimeError("TTS OOM")
        return video.with_name(f"{video.stem}_DOBLADO.mkv")

    pipeline.process_file = MagicMock(side_effect=_maybe_fail)

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process_directory(season)

    msg = str(excinfo.value)
    assert "1/2" in msg
    assert "fallaron TTS" in msg
    assert "Foo - S01E02.mkv" in msg


def test_process_directory_combines_missing_and_failed(tmp_path):
    season = tmp_path / "Season 01"
    _make_chapter_with_srt(season, "Foo - S01E01")
    v2 = _make_chapter_with_srt(season, "Foo - S01E02")  # will fail TTS
    _make_chapter_no_srt(season, "Foo - S01E03")          # missing SRT

    pipeline = _make_pipeline()

    def _maybe_fail(video, srt):
        if video == v2:
            raise RuntimeError("synth crashed")
        return video.with_name(f"{video.stem}_DOBLADO.mkv")

    pipeline.process_file = MagicMock(side_effect=_maybe_fail)

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process_directory(season)

    msg = str(excinfo.value)
    assert "1/3" in msg
    assert "sin SRT ES" in msg
    assert "fallaron TTS" in msg


# ---------------------------------------------------------------------------
# Edge: artefact folders skipped
# ---------------------------------------------------------------------------

def test_process_directory_skips_doblajes_and_elevenlabs(tmp_path):
    season = tmp_path / "Season 01"
    _make_chapter_with_srt(season, "Foo - S01E01")
    # These should NOT be picked up
    (season / "doblajes").mkdir()
    (season / "doblajes" / "Foo - S01E01.mkv").write_bytes(b"x")
    (season / "elevenlabs").mkdir()
    (season / "elevenlabs" / "Foo - S01E01.mkv").write_bytes(b"x")

    pipeline = _make_pipeline()
    pipeline.process_file = MagicMock(
        side_effect=lambda video, srt: video.with_name(f"{video.stem}_DOBLADO.mkv")
    )

    out = pipeline.process_directory(season)

    assert len(out) == 1
    assert pipeline.process_file.call_count == 1


def test_process_directory_skips_existing_DOBLADO_files(tmp_path):
    """A previously dubbed file shouldn't be re-fed into the pipeline."""
    season = tmp_path / "Season 01"
    _make_chapter_with_srt(season, "Foo - S01E01")
    (season / "Foo - S01E01_DOBLADO.mkv").write_bytes(b"x")

    pipeline = _make_pipeline()
    pipeline.process_file = MagicMock(
        side_effect=lambda video, srt: video.with_name(f"{video.stem}_DOBLADO.mkv")
    )

    out = pipeline.process_directory(season)

    assert len(out) == 1
    assert pipeline.process_file.call_count == 1


# ---------------------------------------------------------------------------
# Empty directory: silent OK (no chapters → nothing to fail on)
# ---------------------------------------------------------------------------

def test_process_directory_empty_returns_empty(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()

    pipeline = _make_pipeline()
    pipeline.process_file = MagicMock()

    out = pipeline.process_directory(season)

    assert out == []
    pipeline.process_file.assert_not_called()
