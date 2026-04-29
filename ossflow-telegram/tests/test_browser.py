"""Tests for ChannelBrowser — uses a fake client, no network."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, AsyncIterator, List

import pytest

from telegram_fetcher.browser import ChannelBrowser


pytestmark = pytest.mark.asyncio


class FakeAttrFilename:
    __name__ = "DocumentAttributeFilename"

    def __init__(self, file_name: str) -> None:
        self.file_name = file_name


class FakeAttrVideo:
    __name__ = "DocumentAttributeVideo"

    def __init__(self, duration: float, w: int = 1920, h: int = 1080) -> None:
        self.duration = duration
        self.w = w
        self.h = h


# Force type(a).__name__ to match what extractor expects — use real classes.
class DocumentAttributeFilename(FakeAttrFilename):
    pass


class DocumentAttributeVideo(FakeAttrVideo):
    pass


class FakeDocument:
    def __init__(self, mime: str, size: int, attrs: list) -> None:
        self.mime_type = mime
        self.size = size
        self.attributes = attrs


class FakeMediaDoc:
    def __init__(self, doc: FakeDocument) -> None:
        self.document = doc


class FakeMessage:
    def __init__(self, mid: int, media: Any, message: str = "", date: Any = None) -> None:
        self.id = mid
        self.media = media
        self.message = message
        self.text = message
        self.date = date or datetime(2026, 4, 1, tzinfo=timezone.utc)


class FakeEntity:
    def __init__(self, cid: int, username: str, title: str) -> None:
        self.id = cid
        self.username = username
        self.title = title


class FakeClient:
    def __init__(self, messages: List[FakeMessage]) -> None:
        self._messages = messages
        self.last_filter: Any = "UNSET"
        self.last_limit: Any = None

    async def get_entity(self, username: str) -> FakeEntity:
        return FakeEntity(12345, username.lstrip("@"), "BJJ Channel")

    def iter_messages(self, entity: Any, *, limit: int | None = None, filter: Any = None) -> AsyncIterator:
        self.last_filter = filter
        self.last_limit = limit
        msgs = self._messages

        class _Iter:
            def __init__(self) -> None:
                self._i = 0

            def __aiter__(self) -> "_Iter":
                return self

            async def __anext__(self) -> Any:
                if self._i >= len(msgs):
                    raise StopAsyncIteration
                m = msgs[self._i]
                self._i += 1
                return m

        return _Iter()


def _video_msg(mid: int, filename: str, size: int = 1000) -> FakeMessage:
    doc = FakeDocument(
        mime="video/mp4",
        size=size,
        attrs=[DocumentAttributeFilename(filename), DocumentAttributeVideo(300.5)],
    )
    return FakeMessage(mid, FakeMediaDoc(doc), message=f"caption {mid}")


def _non_video_msg(mid: int) -> FakeMessage:
    doc = FakeDocument(mime="image/jpeg", size=100, attrs=[DocumentAttributeFilename("x.jpg")])
    return FakeMessage(mid, FakeMediaDoc(doc))


def _text_only_msg(mid: int) -> FakeMessage:
    return FakeMessage(mid, None, message="hi")


async def test_iter_channel_media_filters_and_extracts() -> None:
    client = FakeClient([
        _video_msg(1, "John Danaher - Leglocks 1.mp4", size=5000),
        _text_only_msg(2),
        _non_video_msg(3),
        _video_msg(4, "John Danaher - Leglocks 2.mp4", size=6000),
    ])
    browser = ChannelBrowser()
    collected = []
    async for item in browser.iter_channel_media(client, "@bjjinstructionalsmma", limit=50):
        collected.append(item)
    assert len(collected) == 2
    assert collected[0]["message_id"] == 1
    assert collected[0]["filename"] == "John Danaher - Leglocks 1.mp4"
    assert collected[0]["mime_type"] == "video/mp4"
    assert collected[0]["size_bytes"] == 5000
    assert collected[0]["duration"] == 300.5
    assert collected[0]["channel_id"] == "12345"
    assert collected[0]["caption"] == "caption 1"
    assert client.last_limit == 50


async def test_resolve_channel_missing_raises() -> None:
    class BadClient:
        async def get_entity(self, username: str) -> Any:
            raise RuntimeError("not found")

    browser = ChannelBrowser()
    from telegram_fetcher.errors import ChannelNotFoundError
    with pytest.raises(ChannelNotFoundError):
        await browser.resolve_channel(BadClient(), "@nope")
