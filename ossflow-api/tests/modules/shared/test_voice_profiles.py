"""Tests para ossflow_api.shared.voice_profiles."""

from __future__ import annotations

import json
from pathlib import Path

from ossflow_api.shared.voice_profiles import (
    SIDECAR_NAME,
    load_voice_profile_for_path,
)


def test_returns_empty_when_no_sidecar(tmp_path: Path) -> None:
    video = tmp_path / "Season_01" / "S01E01.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    assert load_voice_profile_for_path(str(video)) == ""


def test_finds_sidecar_one_level_up(tmp_path: Path) -> None:
    instructional = tmp_path / "MyInstructional"
    season = instructional / "Season_01"
    season.mkdir(parents=True)
    video = season / "S01E01.mp4"
    video.touch()
    sidecar = instructional / SIDECAR_NAME
    sidecar.write_text(
        json.dumps({"voice_profile": "instructor_smith"}), encoding="utf-8"
    )

    assert load_voice_profile_for_path(str(video)) == "instructor_smith"


def test_walks_up_at_most_4_levels(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "video.mp4"
    deep.parent.mkdir(parents=True)
    deep.touch()
    # Sidecar 5 levels up: should NOT be found
    far = tmp_path / SIDECAR_NAME
    far.write_text(
        json.dumps({"voice_profile": "too_far"}), encoding="utf-8"
    )

    assert load_voice_profile_for_path(str(deep)) == ""


def test_returns_empty_when_sidecar_lacks_voice_profile(tmp_path: Path) -> None:
    folder = tmp_path / "instr"
    folder.mkdir()
    (folder / SIDECAR_NAME).write_text(
        json.dumps({"title": "x"}), encoding="utf-8"
    )
    video = folder / "video.mp4"
    video.touch()

    assert load_voice_profile_for_path(str(video)) == ""


def test_returns_empty_when_voice_profile_is_empty_string(tmp_path: Path) -> None:
    folder = tmp_path / "instr"
    folder.mkdir()
    (folder / SIDECAR_NAME).write_text(
        json.dumps({"voice_profile": ""}), encoding="utf-8"
    )
    video = folder / "video.mp4"
    video.touch()

    assert load_voice_profile_for_path(str(video)) == ""


def test_handles_corrupt_sidecar(tmp_path: Path) -> None:
    folder = tmp_path / "instr"
    folder.mkdir()
    (folder / SIDECAR_NAME).write_text("{not valid json", encoding="utf-8")
    video = folder / "video.mp4"
    video.touch()

    assert load_voice_profile_for_path(str(video)) == ""
