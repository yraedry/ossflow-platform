"""Verify subtitle-generator routes file vs directory to the correct pipeline call."""

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
        "subtitle_generator.config",
        type(
            "M",
            (),
            {
                "DEFAULT_INITIAL_PROMPT": "prompt",
                "SubtitleConfig": MagicMock(),
                "TranscriptionConfig": MagicMock(),
                "generate_prompt": lambda **_k: "p",
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "subtitle_generator.cuda_setup",
        type(
            "M",
            (),
            {
                "setup_nvidia_dlls": lambda: None,
                "setup_pytorch_safety": lambda: None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "subtitle_generator.pipeline",
        type("M", (), {"SubtitlePipeline": fake_pipeline_cls}),
    )


def test_run_file_calls_process_file(tmp_path, monkeypatch):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x")

    events: list[JobEvent] = []
    fake_pipeline = MagicMock()
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline)
    _patch_pipeline_modules(monkeypatch, fake_pipeline_cls)

    import app as sub_app  # noqa: E402

    req = RunRequest(input_path=str(video))
    sub_app._run_subtitle_generator(req, events.append)

    fake_pipeline.process_file.assert_called_once_with(video)
    fake_pipeline.process_directory.assert_not_called()


def test_run_directory_calls_process_directory(tmp_path, monkeypatch):
    events: list[JobEvent] = []
    fake_pipeline = MagicMock()
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline)
    _patch_pipeline_modules(monkeypatch, fake_pipeline_cls)

    import app as sub_app  # noqa: E402

    req = RunRequest(input_path=str(tmp_path))
    sub_app._run_subtitle_generator(req, events.append)

    fake_pipeline.process_directory.assert_called_once_with(tmp_path)
    fake_pipeline.process_file.assert_not_called()


def test_resolve_input_rejects_missing_path(tmp_path):
    import app as sub_app  # noqa: E402

    with pytest.raises(FileNotFoundError):
        sub_app._resolve_input(tmp_path / "nope")
