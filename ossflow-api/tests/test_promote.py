"""Tests for the dub promotion endpoints."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from api import promote


def _make_chapter(season: Path, name: str, *, with_subs: bool = True) -> tuple[Path, Path]:
    """Create a fake original .mp4 + dubbed doblajes/<name>.mkv pair, return (orig, dubbed)."""
    season.mkdir(parents=True, exist_ok=True)
    (season / "doblajes").mkdir(exist_ok=True)
    orig = season / f"{name}.mp4"
    dubbed = season / "doblajes" / f"{name}.mkv"
    orig.write_bytes(b"\0" * 64)
    dubbed.write_bytes(b"\0" * 64)
    if with_subs:
        (season / f"{name}.es.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhola\n", encoding="utf-8")
        (season / f"{name}.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    return orig, dubbed


def test_resolve_inputs_collects_paths(tmp_path):
    season = tmp_path / "Season 01"
    orig, dubbed = _make_chapter(season, "S01E01")
    inp = promote._resolve_inputs(str(orig))
    assert inp.original == orig
    assert inp.dubbed == dubbed
    assert inp.output == season / "S01E01.mkv"
    assert inp.output_tmp == season / "S01E01.mkv.tmp"
    assert inp.es_srt == season / "S01E01.es.srt"
    assert inp.en_srt == season / "S01E01.en.srt"
    # All known sidecars listed for deletion (existing + non-existing — unlink is best-effort).
    paths = [str(p) for p in inp.sidecars_to_delete]
    assert any(p.endswith("_VOCALS.wav") for p in paths)
    assert any(p.endswith(".dub-qa.json") for p in paths)


def test_resolve_inputs_missing_original(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    with pytest.raises(HTTPException) as ei:
        promote._resolve_inputs(str(season / "ghost.mp4"))
    assert ei.value.status_code == 409
    assert ei.value.detail["code"] == "original_missing"


def test_resolve_inputs_missing_dubbed(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    orig = season / "S01E01.mp4"
    orig.write_bytes(b"\0")
    with pytest.raises(HTTPException) as ei:
        promote._resolve_inputs(str(orig))
    assert ei.value.detail["code"] == "dubbed_missing"


def test_resolve_inputs_already_promoted_collision(tmp_path):
    """If a <name>.mkv exists side-by-side with the .mp4 original, refuse."""
    season = tmp_path / "Season 01"
    orig, _ = _make_chapter(season, "S01E01")
    # Pre-existing .mkv (e.g. partial run from before the .mp4 was discovered)
    (season / "S01E01.mkv").write_bytes(b"\0")
    with pytest.raises(HTTPException) as ei:
        promote._resolve_inputs(str(orig))
    assert ei.value.detail["code"] == "already_promoted"


def test_build_ffmpeg_argv_full(tmp_path):
    season = tmp_path / "Season 01"
    orig, _ = _make_chapter(season, "S01E01")
    argv = promote._build_ffmpeg_argv(promote._resolve_inputs(str(orig)))
    # Two audio inputs first (dubbed, original), then 2 srts.
    assert argv[0] == "ffmpeg"
    # Map flags must request video from input 0 and both audios.
    assert "-map" in argv and "0:v:0" in argv and "0:a:0?" in argv and "1:a:0?" in argv
    # Spanish marked as default
    idx = argv.index("-disposition:a:0")
    assert argv[idx + 1] == "default"
    # Spanish title metadata present
    assert "title=Español (doblaje IA)" in argv
    assert "title=English (original)" in argv
    # Both subtitle inputs included as -map 2:0? and -map 3:0?
    assert "2:0?" in argv and "3:0?" in argv


def test_build_ffmpeg_argv_no_subs(tmp_path):
    season = tmp_path / "Season 01"
    orig, _ = _make_chapter(season, "S01E01", with_subs=False)
    argv = promote._build_ffmpeg_argv(promote._resolve_inputs(str(orig)))
    # No subtitle map flags
    assert "2:0?" not in argv
    assert "3:0?" not in argv
    # Output is still .mkv.tmp at the end
    assert argv[-1].endswith(".mkv.tmp")


def test_promote_one_happy_path(tmp_path, monkeypatch):
    season = tmp_path / "Season 01"
    orig, dubbed = _make_chapter(season, "S01E01")

    def fake_run(argv, **kwargs):
        # ffmpeg writes the output file and returns 0
        out = Path(argv[-1])
        out.write_bytes(b"\0muxed")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(promote.subprocess, "run", fake_run)
    monkeypatch.setattr(promote, "_pipeline_active_for", lambda *_: None)
    monkeypatch.setattr(promote, "_refresh_cache_for", lambda *_: None)

    result = promote._promote_one(str(orig))

    assert result["ok"] is True
    final = season / "S01E01.mkv"
    assert final.exists()
    assert final.read_bytes() == b"\0muxed"
    # Original and dubbed removed
    assert not orig.exists()
    assert not dubbed.exists()
    # doblajes/ folder swept (was empty)
    assert not (season / "doblajes").exists()
    # sidecars (srts) gone
    assert not (season / "S01E01.es.srt").exists()
    assert not (season / "S01E01.en.srt").exists()


def test_promote_one_ffmpeg_failure_keeps_inputs(tmp_path, monkeypatch):
    """When ffmpeg returns non-zero, the .tmp must be cleaned up and
    nothing else touched."""
    season = tmp_path / "Season 01"
    orig, dubbed = _make_chapter(season, "S01E01")

    def fake_run(argv, **kwargs):
        # Simulate ffmpeg writing a partial .tmp before crashing
        Path(argv[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(argv, 1, "", "fake stderr line\n")

    monkeypatch.setattr(promote.subprocess, "run", fake_run)
    monkeypatch.setattr(promote, "_pipeline_active_for", lambda *_: None)

    with pytest.raises(HTTPException) as ei:
        promote._promote_one(str(orig))
    assert ei.value.status_code == 500
    assert ei.value.detail["code"] == "ffmpeg_failed"

    # Inputs intact
    assert orig.exists()
    assert dubbed.exists()
    assert (season / "S01E01.es.srt").exists()
    # No .mkv landed
    assert not (season / "S01E01.mkv").exists()
    # .tmp swept
    assert not (season / "S01E01.mkv.tmp").exists()


def test_promote_one_blocks_when_pipeline_active(tmp_path, monkeypatch):
    season = tmp_path / "Season 01"
    orig, _ = _make_chapter(season, "S01E01")

    monkeypatch.setattr(promote, "_pipeline_active_for", lambda paths: "abc1234")

    with pytest.raises(HTTPException) as ei:
        promote._promote_one(str(orig))
    assert ei.value.status_code == 409
    assert ei.value.detail["code"] == "pipeline_active"
    # Nothing was written/deleted
    assert orig.exists()


def test_promote_season_aggregates_results(tmp_path, monkeypatch):
    season = tmp_path / "Season 01"
    o1, _ = _make_chapter(season, "S01E01")
    o2, _ = _make_chapter(season, "S01E02")
    # A third candidate has no dubbed sibling — should be skipped silently
    # by the season iterator (not flagged as a failure).
    (season / "S01E03.mp4").write_bytes(b"\0")

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"ok")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(promote.subprocess, "run", fake_run)
    monkeypatch.setattr(promote, "_pipeline_active_for", lambda *_: None)
    monkeypatch.setattr(promote, "_refresh_cache_for", lambda *_: None)

    result = promote.promote_season(promote.PromoteSeasonBody(season_path=str(season)))
    assert result["promoted_count"] == 2
    assert result["failed_count"] == 0
    # E03 didn't have doblajes sibling, was never enqueued (not in skipped either).
    assert all("S01E03" not in p for p in result["promoted"])
