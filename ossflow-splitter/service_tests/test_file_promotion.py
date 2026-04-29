"""Verify that _run_chapter_splitter promotes a file path to its parent dir."""

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


def test_run_promotes_file_to_parent(tmp_path, monkeypatch):
    video = tmp_path / "Tripod Passing.mp4"
    video.write_bytes(b"x")

    events: list[JobEvent] = []

    import app as chapter_app  # noqa: E402

    fake_pipeline_cls = MagicMock()
    fake_config_cls = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "chapter_splitter.config",
        type("M", (), {"Config": fake_config_cls}),
    )
    monkeypatch.setitem(
        sys.modules,
        "chapter_splitter.pipeline",
        type("M", (), {"Pipeline": fake_pipeline_cls}),
    )
    monkeypatch.setitem(
        sys.modules,
        "chapter_splitter.utils",
        type("M", (), {"setup_logging": lambda *_a, **_k: None}),
    )

    req = RunRequest(input_path=str(video))
    # Must not raise FileNotFoundError
    chapter_app._run_chapter_splitter(req, events.append)

    # Config was built with the PARENT dir
    kwargs = fake_config_cls.call_args.kwargs
    assert kwargs["root_dir"] == tmp_path

    # Promotion message logged
    msgs = [e.data.get("message", "") for e in events if e.type == "log"]
    assert any("Promoted file to directory" in m for m in msgs)


def test_resolve_root_rejects_missing_path(tmp_path):
    import app as chapter_app  # noqa: E402

    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        chapter_app._resolve_root(missing, lambda _e: None)
