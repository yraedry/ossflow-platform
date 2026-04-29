"""Verify dubbing-generator routes file vs directory to the correct pipeline call."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PY_ROOT = Path(__file__).resolve().parents[2]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

_SVC_ROOT = Path(__file__).resolve().parents[1]
if str(_SVC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SVC_ROOT))

from bjj_service_kit import JobEvent, RunRequest  # noqa: E402


def _patch_pipeline_modules(monkeypatch, fake_pipeline_cls):
    monkeypatch.setitem(
        sys.modules,
        "dubbing_generator.config",
        type("M", (), {"DubbingConfig": MagicMock()}),
    )
    monkeypatch.setitem(
        sys.modules,
        "dubbing_generator.pipeline",
        type("M", (), {"DubbingPipeline": fake_pipeline_cls}),
    )
    monkeypatch.setitem(
        sys.modules,
        "dubbing_generator.translation",
        type("M", (), {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "dubbing_generator.translation.translator",
        type("M", (), {"Translator": MagicMock()}),
    )


def test_run_file_calls_process_file(tmp_path, monkeypatch):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x")
    srt = tmp_path / "episode_ESP_DUB.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhola\n", encoding="utf-8")

    events: list[JobEvent] = []
    fake_pipeline = MagicMock()
    fake_pipeline.process_file.return_value = tmp_path / "episode_DOBLADO.mkv"
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline)
    _patch_pipeline_modules(monkeypatch, fake_pipeline_cls)

    import app as dub_app  # noqa: E402

    req = RunRequest(input_path=str(video), options={"skip_translation": True})
    dub_app._run_dubbing_generator(req, events.append)

    fake_pipeline.process_file.assert_called_once_with(video, srt)
    fake_pipeline.process_directory.assert_not_called()


def test_run_file_without_srt_raises(tmp_path, monkeypatch):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x")

    events: list[JobEvent] = []
    fake_pipeline = MagicMock()
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline)
    _patch_pipeline_modules(monkeypatch, fake_pipeline_cls)

    import app as dub_app  # noqa: E402

    req = RunRequest(input_path=str(video), options={"skip_translation": True})
    with pytest.raises(FileNotFoundError):
        dub_app._run_dubbing_generator(req, events.append)


def test_run_directory_calls_process_directory(tmp_path, monkeypatch):
    events: list[JobEvent] = []
    fake_pipeline = MagicMock()
    fake_pipeline.process_directory.return_value = []
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline)
    _patch_pipeline_modules(monkeypatch, fake_pipeline_cls)

    import app as dub_app  # noqa: E402

    req = RunRequest(input_path=str(tmp_path), options={"skip_translation": True})
    dub_app._run_dubbing_generator(req, events.append)

    fake_pipeline.process_directory.assert_called_once_with(tmp_path)
    fake_pipeline.process_file.assert_not_called()


def test_resolve_input_rejects_missing_path(tmp_path):
    import app as dub_app  # noqa: E402

    with pytest.raises(FileNotFoundError):
        dub_app._resolve_input(tmp_path / "nope")
