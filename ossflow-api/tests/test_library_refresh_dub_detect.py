"""Tests for ffprobe-based has_dubbing / has_subtitles detection
introduced for the promote feature."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from api import library_refresh as lr


def _patch_probe(audio: list[str], subs: list[str] | None = None):
    """Helper: patch the single ffprobe call with fixed return values."""
    return patch.object(
        lr, "_probe_track_languages", return_value=(audio, subs or []),
    )


def test_has_dubbing_via_doblajes_sibling(tmp_path):
    """Pre-promotion: doblajes/<name>.mkv exists → has_dubbing without
    needing the audio probe (but the probe still runs to seed the cache)."""
    season = tmp_path / "Season 01"
    season.mkdir()
    (season / "doblajes").mkdir()
    video = season / "S01E01.mp4"
    video.write_bytes(b"\0" * 32)
    (season / "doblajes" / "S01E01.mkv").write_bytes(b"\0" * 32)

    with _patch_probe([], []) as probe:
        flags = lr._video_flags(video, cached=None)
    assert flags["has_dubbing"] is True
    assert flags["is_promoted"] is False
    assert probe.call_count == 1


def test_has_dubbing_via_audio_probe_post_promotion(tmp_path):
    """Post-promotion: only <name>.mkv remains, but it has Spanish audio."""
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    with _patch_probe(["spa", "eng"], ["spa", "eng"]):
        flags = lr._video_flags(video, cached=None)
    assert flags["has_dubbing"] is True
    assert flags["is_promoted"] is True
    assert flags["audio_languages"] == ["spa", "eng"]
    assert flags["subtitle_languages"] == ["spa", "eng"]


def test_no_dubbing_for_plain_original(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mp4"
    video.write_bytes(b"\0" * 32)

    with _patch_probe(["eng"], []):
        flags = lr._video_flags(video, cached=None)
    assert flags["has_dubbing"] is False
    assert flags["is_promoted"] is False


def test_subtitles_via_external_sidecars(tmp_path):
    """No embedded subs but .es.srt and .en.srt exist on disk."""
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mp4"
    video.write_bytes(b"\0" * 32)
    (season / "S01E01.en.srt").write_text("1\n", encoding="utf-8")
    (season / "S01E01.es.srt").write_text("1\n", encoding="utf-8")

    with _patch_probe(["eng"], []):
        flags = lr._video_flags(video, cached=None)
    assert flags["has_subtitles_en"] is True
    assert flags["has_subtitles_es"] is True


def test_subtitles_via_embedded_streams_post_promote(tmp_path):
    """After promote: no .srt sidecars left, but the .mkv has embedded
    subtitle streams. has_subtitles_en/es must still be True."""
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    with _patch_probe(["spa", "eng"], ["spa", "eng"]):
        flags = lr._video_flags(video, cached=None)
    assert flags["has_subtitles_en"] is True
    assert flags["has_subtitles_es"] is True


def test_subtitles_only_one_language_embedded(tmp_path):
    """A .mkv with ONLY english embedded subs → has_subtitles_en True,
    has_subtitles_es False (no sidecar, no embedded ES)."""
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    with _patch_probe(["eng"], ["eng"]):
        flags = lr._video_flags(video, cached=None)
    assert flags["has_subtitles_en"] is True
    assert flags["has_subtitles_es"] is False


def test_probe_cached_when_fingerprint_unchanged(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    fingerprint = lr._file_fingerprint(video)
    cached = {
        "audio_fingerprint": list(fingerprint),
        "audio_languages": ["spa", "eng"],
        "subtitle_languages": ["spa", "eng"],
    }

    with patch.object(lr, "_probe_track_languages") as probe:
        flags = lr._video_flags(video, cached=cached)
    probe.assert_not_called()
    assert flags["audio_languages"] == ["spa", "eng"]
    assert flags["has_dubbing"] is True
    assert flags["has_subtitles_es"] is True


def test_probe_reruns_when_file_modified(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    cached = {
        "audio_fingerprint": [0, 1],  # bogus fingerprint → must invalidate
        "audio_languages": ["eng"],
        "subtitle_languages": [],
    }

    with _patch_probe(["spa", "eng"], ["spa", "eng"]) as probe:
        flags = lr._video_flags(video, cached=cached)
    probe.assert_called_once()
    assert flags["audio_languages"] == ["spa", "eng"]
    assert flags["has_dubbing"] is True
    assert flags["has_subtitles_es"] is True


def test_probe_reruns_when_subtitle_languages_missing_from_cache(tmp_path):
    """Older library.json entries (pre-subtitle-cache) don't have
    subtitle_languages — _video_flags must re-probe to fill it in even
    when the audio fingerprint matches."""
    season = tmp_path / "Season 01"
    season.mkdir()
    video = season / "S01E01.mkv"
    video.write_bytes(b"\0" * 32)

    fingerprint = lr._file_fingerprint(video)
    cached = {
        "audio_fingerprint": list(fingerprint),
        "audio_languages": ["spa", "eng"],
        # no subtitle_languages key — pre-migration cache shape
    }

    with _patch_probe(["spa", "eng"], ["eng"]) as probe:
        flags = lr._video_flags(video, cached=cached)
    probe.assert_called_once()
    assert flags["subtitle_languages"] == ["eng"]


def test_spanish_audio_detection_accepts_aliases():
    assert lr._has_spanish_audio(["spa"]) is True
    assert lr._has_spanish_audio(["es"]) is True
    assert lr._has_spanish_audio(["esp"]) is True
    assert lr._has_spanish_audio(["spanish"]) is True
    assert lr._has_spanish_audio(["eng", "fre"]) is False
    assert lr._has_spanish_audio([]) is False
    assert lr._has_spanish_audio(None) is False
    assert lr._has_spanish_audio([""]) is False


def test_english_subtitle_detection_accepts_aliases():
    assert lr._has_english_subtitle(["eng"]) is True
    assert lr._has_english_subtitle(["en"]) is True
    assert lr._has_english_subtitle(["english"]) is True
    assert lr._has_english_subtitle(["spa"]) is False
    assert lr._has_english_subtitle([""]) is False
    assert lr._has_english_subtitle(None) is False


def test_spanish_subtitle_detection_accepts_aliases():
    assert lr._has_spanish_subtitle(["spa"]) is True
    assert lr._has_spanish_subtitle(["es"]) is True
    assert lr._has_spanish_subtitle(["esp"]) is True
    assert lr._has_spanish_subtitle(["eng"]) is False
