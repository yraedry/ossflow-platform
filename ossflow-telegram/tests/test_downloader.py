"""Tests for Downloader — mocks client.download_media, validates cancel cleanup."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from telegram_fetcher.downloader import Downloader
from telegram_fetcher.errors import MediaUnavailableError


pytestmark = pytest.mark.asyncio


class FakeMsg:
    def __init__(self, has_media: bool = True) -> None:
        self.media = object() if has_media else None


class FakeClient:
    def __init__(self, msg: FakeMsg, behavior: str = "ok", payload: bytes = b"xx") -> None:
        self.msg = msg
        self.behavior = behavior
        self.payload = payload

    async def get_messages(self, channel_id: str, *, ids: int) -> FakeMsg:
        return self.msg

    async def download_media(self, msg: Any, *, file: str, progress_callback: Any) -> str:
        if self.behavior == "ok":
            Path(file).write_bytes(self.payload)
            if progress_callback is not None:
                progress_callback(len(self.payload), len(self.payload))
            return file
        if self.behavior == "cancel_midway":
            # Write partial then cancel.
            Path(file).write_bytes(b"partial")
            if progress_callback is not None:
                progress_callback(7, 100)
            raise asyncio.CancelledError()
        if self.behavior == "raise":
            Path(file).write_bytes(b"garbage")
            raise RuntimeError("boom")
        raise AssertionError("unknown behavior")


async def test_download_success_returns_path_and_size(tmp_path: Path) -> None:
    dest = tmp_path / "out.mp4"
    client = FakeClient(FakeMsg(True), behavior="ok", payload=b"hello world")
    dl = Downloader()
    progress = []
    path, size = await dl.download(
        client, "C1", 42, str(dest), progress_cb=lambda c, t: progress.append((c, t))
    )
    assert Path(path).is_file()
    assert size == len(b"hello world")
    assert progress and progress[-1][0] == size


async def test_download_cancel_removes_partial(tmp_path: Path) -> None:
    dest = tmp_path / "part.mp4"
    client = FakeClient(FakeMsg(True), behavior="cancel_midway")
    dl = Downloader()
    with pytest.raises(asyncio.CancelledError):
        await dl.download(client, "C1", 1, str(dest))
    assert not dest.exists(), "partial file must be deleted on cancellation"


async def test_download_no_media_raises(tmp_path: Path) -> None:
    client = FakeClient(FakeMsg(False))
    dl = Downloader()
    with pytest.raises(MediaUnavailableError):
        await dl.download(client, "C1", 1, str(tmp_path / "x.mp4"))


async def test_download_other_error_cleans_partial(tmp_path: Path) -> None:
    dest = tmp_path / "broken.mp4"
    client = FakeClient(FakeMsg(True), behavior="raise")
    dl = Downloader()
    from telegram_fetcher.errors import TelegramError
    with pytest.raises(TelegramError):
        await dl.download(client, "C1", 1, str(dest))
    assert not dest.exists()
