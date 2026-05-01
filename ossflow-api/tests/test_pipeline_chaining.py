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
    # Tras T22.5 solo S2-Pro: el motor lee la voz de s2_voice_profile, NO
    # del campo voice_profile (que era estado XTTS/ElevenLabs-era). Lo que
    # debe propagarse es input_path encadenado y los opts de S2-Pro.
    client, payload, _ = _client_and_payload(
        "dubbing",
        "/media/Show/original.mkv",
        {"s2_voice_profile": "gordon.wav"},
        chained_path="/media/Show/Season 01",
    )
    assert payload["input_path"] == "/media/Show/Season 01"
    assert payload["options"]["tts_engine"] == "s2pro"
    assert payload["options"]["s2_ref_audio_path"] == "/voices/gordon.wav"
    # voice_profile (legacy XTTS) NO se propaga para S2-Pro.
    assert "voice_profile" not in payload["options"]


def test_client_and_payload_no_chain_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_library_path", lambda: "")
    client, payload, _ = _client_and_payload(
        "subtitles", "/media/Show/original.mkv", {}, chained_path=None,
    )
    assert payload["input_path"] == "/media/Show/original.mkv"
