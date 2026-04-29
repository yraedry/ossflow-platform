from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dubbing_generator.config import DubbingConfig
from dubbing_generator.tts.s2pro_server_manager import S2ProServerManager


def test_skips_when_engine_not_s2pro():
    cfg = DubbingConfig(tts_engine="kokoro")
    m = S2ProServerManager(cfg)
    with patch("subprocess.Popen") as popen:
        m.start()
        popen.assert_not_called()
    assert m.process is None
    assert m.is_ready() is False


def test_skips_when_binary_missing(tmp_path: Path, monkeypatch):
    cfg = DubbingConfig(tts_engine="s2pro")
    monkeypatch.setattr(
        "dubbing_generator.tts.s2pro_server_manager._S2_BINARY",
        tmp_path / "missing",
    )
    m = S2ProServerManager(cfg)
    with patch("subprocess.Popen") as popen:
        m.start()
        popen.assert_not_called()
    assert m.is_ready() is False


def test_skips_when_gguf_missing(tmp_path: Path, monkeypatch):
    binary = tmp_path / "s2"
    binary.write_text("")
    binary.chmod(0o755)
    monkeypatch.setattr(
        "dubbing_generator.tts.s2pro_server_manager._S2_BINARY", binary,
    )
    cfg = DubbingConfig(
        tts_engine="s2pro",
        s2_gguf_path=str(tmp_path / "missing.gguf"),
    )
    m = S2ProServerManager(cfg)
    with patch("subprocess.Popen") as popen:
        m.start()
        popen.assert_not_called()


def test_start_returns_fast_does_not_block(tmp_path: Path, monkeypatch):
    """start() must return without waiting for health."""
    binary = tmp_path / "s2"; binary.write_text(""); binary.chmod(0o755)
    monkeypatch.setattr(
        "dubbing_generator.tts.s2pro_server_manager._S2_BINARY", binary,
    )
    (tmp_path / "model.gguf").write_text("")
    (tmp_path / "tokenizer.json").write_text("")
    cfg = DubbingConfig(
        tts_engine="s2pro",
        s2_gguf_path=str(tmp_path / "model.gguf"),
        s2_tokenizer_path=str(tmp_path / "tokenizer.json"),
        s2_health_timeout_s=10.0,
    )

    proc = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline.return_value = b""
    proc.poll.return_value = None

    started = threading.Event()

    def slow_health(*a, **kw):
        started.set()
        return False  # never becomes ready in this test

    with patch("subprocess.Popen", return_value=proc), \
         patch("dubbing_generator.tts.s2pro_server_manager._wait_for_health",
               side_effect=slow_health):
        m = S2ProServerManager(cfg)
        m.start()
        # start() must have returned immediately even though health probe
        # would take 10 s. Confirm the readiness thread is the one polling.
        assert started.wait(timeout=2.0), "readiness probe never started"
        assert m.is_ready() is False  # health returned False


def test_start_marks_ready_when_health_ok(tmp_path: Path, monkeypatch):
    binary = tmp_path / "s2"; binary.write_text(""); binary.chmod(0o755)
    monkeypatch.setattr(
        "dubbing_generator.tts.s2pro_server_manager._S2_BINARY", binary,
    )
    (tmp_path / "model.gguf").write_text("")
    (tmp_path / "tokenizer.json").write_text("")
    cfg = DubbingConfig(
        tts_engine="s2pro",
        s2_gguf_path=str(tmp_path / "model.gguf"),
        s2_tokenizer_path=str(tmp_path / "tokenizer.json"),
    )

    proc = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline.return_value = b""
    proc.poll.return_value = None

    with patch("subprocess.Popen", return_value=proc), \
         patch("dubbing_generator.tts.s2pro_server_manager._wait_for_health",
               return_value=True):
        m = S2ProServerManager(cfg)
        m.start()
        assert m.wait_until_ready(timeout=2.0) is True
        assert m.is_ready() is True


def test_stop_terminates_process(monkeypatch):
    cfg = DubbingConfig(tts_engine="s2pro")
    m = S2ProServerManager(cfg)
    proc = MagicMock(); proc.poll.return_value = None
    m._process = proc
    m.stop()
    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()


def test_stop_kills_when_terminate_times_out(monkeypatch):
    import subprocess as sp
    cfg = DubbingConfig(tts_engine="s2pro")
    m = S2ProServerManager(cfg)
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.side_effect = [sp.TimeoutExpired("s2", 5.0), None]
    m._process = proc
    m.stop()
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


def test_start_is_idempotent_when_already_running(tmp_path: Path, monkeypatch):
    """In process_directory the manager.start() is called once per
    episode. We want to amortize the GGUF mmap across the batch by
    reusing the running server, not spawning a new one each time."""
    binary = tmp_path / "s2"; binary.write_text(""); binary.chmod(0o755)
    monkeypatch.setattr(
        "dubbing_generator.tts.s2pro_server_manager._S2_BINARY", binary,
    )
    (tmp_path / "model.gguf").write_text("")
    (tmp_path / "tokenizer.json").write_text("")
    cfg = DubbingConfig(
        tts_engine="s2pro",
        s2_gguf_path=str(tmp_path / "model.gguf"),
        s2_tokenizer_path=str(tmp_path / "tokenizer.json"),
    )
    m = S2ProServerManager(cfg)
    # Simulate first start having spawned a live process.
    alive_proc = MagicMock(); alive_proc.poll.return_value = None
    m._process = alive_proc

    with patch("subprocess.Popen") as popen:
        m.start()
        popen.assert_not_called()
    # The original process is still the one we have.
    assert m._process is alive_proc
