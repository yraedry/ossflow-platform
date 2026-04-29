"""Tests for pipeline chaining: after `chapters` succeeds, subsequent steps
operate on the freshly-created Season_NN/ folder instead of the original."""

from __future__ import annotations

from pathlib import Path

from api import pipeline as pipeline_module
from api.pipeline import _client_and_payload, _detect_season_folder


def test_detect_season_folder_prefers_season_dir():
    added = [
        "Season 01/Ep01.mkv",
        "Season 01/Ep02.mkv",
        "Season 01/Ep03.mkv",
        "noise/readme.txt",
    ]
    out = _detect_season_folder(Path("/media/Show"), added)
    assert out is not None
    assert out.replace("\\", "/").endswith("/Show/Season 01")


def test_detect_season_folder_falls_back_to_majority_parent():
    added = ["chapters/a.mkv", "chapters/b.mkv", "chapters/c.mkv"]
    out = _detect_season_folder(Path("/media/Show"), added)
    assert out is not None
    assert out.replace("\\", "/").endswith("/Show/chapters")


def test_detect_season_folder_returns_none_when_no_videos():
    assert _detect_season_folder(Path("/media/Show"), ["a.txt", "b.json"]) is None


def test_detect_season_folder_returns_none_when_no_target():
    assert _detect_season_folder(None, ["Season 01/x.mkv"]) is None


def test_client_and_payload_uses_original_for_chapters(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_library_path", lambda: "")
    client, payload, _ = _client_and_payload(
        "chapters",
        "/media/Show/original.mkv",
        {},
        chained_path="/media/Show/Season 01",
    )
    # chapters sigue siempre sobre el fichero original
    assert payload["input_path"] == "/media/Show/original.mkv"


def test_client_and_payload_uses_chained_for_subtitles(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_library_path", lambda: "")
    client, payload, _ = _client_and_payload(
        "subtitles",
        "/media/Show/original.mkv",
        {},
        chained_path="/media/Show/Season 01",
    )
    assert payload["input_path"] == "/media/Show/Season 01"


def test_client_and_payload_uses_chained_for_dubbing(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_library_path", lambda: "")
    # voice_profile only propagates for non-s2pro engines (s2pro reads its
    # voice exclusively from s2_voice_profile / s2_ref_audio_path).
    client, payload, _ = _client_and_payload(
        "dubbing",
        "/media/Show/original.mkv",
        {"voice_profile": "gordon", "tts_engine": "elevenlabs"},
        chained_path="/media/Show/Season 01",
    )
    assert payload["input_path"] == "/media/Show/Season 01"
    assert payload["options"]["voice_profile"] == "gordon"


def test_client_and_payload_no_chain_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_library_path", lambda: "")
    client, payload, _ = _client_and_payload(
        "subtitles", "/media/Show/original.mkv", {}, chained_path=None,
    )
    assert payload["input_path"] == "/media/Show/original.mkv"
