"""Tests for Config (credentials polling + hot reload)."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from telegram_fetcher.config import Config


pytestmark = pytest.mark.asyncio


class _FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def get(self, url: str):
        self.calls += 1
        if not self._responses:
            raise RuntimeError("boom")
        return self._responses.pop(0)


def _factory(client: _FakeClient):
    return lambda: client


class TestFetchOnce:
    async def test_returns_creds_when_settings_include_them(self) -> None:
        client = _FakeClient([
            _FakeResponse(200, {"telegram_api_id": "42", "telegram_api_hash": "abc"})
        ])
        cfg = Config("http://p", http_client_factory=_factory(client))
        res = await cfg.fetch_once()
        assert res == (42, "abc")

    async def test_returns_none_on_missing_fields(self) -> None:
        client = _FakeClient([_FakeResponse(200, {"library_path": "/media"})])
        cfg = Config("http://p", http_client_factory=_factory(client))
        assert await cfg.fetch_once() is None

    async def test_returns_none_on_http_error(self) -> None:
        client = _FakeClient([_FakeResponse(503, {})])
        cfg = Config("http://p", http_client_factory=_factory(client))
        assert await cfg.fetch_once() is None

    async def test_returns_none_on_transport_error(self) -> None:
        client = _FakeClient([])  # first .get() will raise
        cfg = Config("http://p", http_client_factory=_factory(client))
        assert await cfg.fetch_once() is None


class TestReloadAndListeners:
    async def test_reload_notifies_listeners_once_per_change(self) -> None:
        client = _FakeClient([
            _FakeResponse(200, {"telegram_api_id": "1", "telegram_api_hash": "h1"}),
            _FakeResponse(200, {"telegram_api_id": "1", "telegram_api_hash": "h1"}),
            _FakeResponse(200, {"telegram_api_id": "2", "telegram_api_hash": "h2"}),
        ])
        cfg = Config("http://p", http_client_factory=_factory(client))
        events: list[tuple[int, str]] = []

        async def _listener(api_id: int, api_hash: str) -> None:
            events.append((api_id, api_hash))

        cfg.on_change(_listener)
        assert await cfg.reload() is True
        assert await cfg.reload() is False  # same creds, no notify
        assert await cfg.reload() is True
        assert events == [(1, "h1"), (2, "h2")]
        assert cfg.api_id == 2
        assert cfg.api_hash == "h2"

    async def test_wait_for_credentials_resolves_after_fetch(self) -> None:
        client = _FakeClient([
            _FakeResponse(200, {"telegram_api_id": "7", "telegram_api_hash": "h"}),
        ])
        cfg = Config("http://p", http_client_factory=_factory(client))
        t = asyncio.create_task(cfg.wait_for_credentials(timeout=1.0))
        await cfg.reload()
        assert await t is True


class TestPollLoopBackoff:
    async def test_loop_retries_with_backoff_until_creds(self) -> None:
        client = _FakeClient([
            _FakeResponse(503, {}),
            _FakeResponse(503, {}),
            _FakeResponse(200, {"telegram_api_id": "1", "telegram_api_hash": "h"}),
        ])
        cfg = Config(
            "http://p",
            min_backoff_s=0.01,
            max_backoff_s=0.02,
            http_client_factory=_factory(client),
        )
        task = asyncio.create_task(cfg.poll_credentials_loop())
        got = await cfg.wait_for_credentials(timeout=2.0)
        cfg.stop()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
        assert got is True
        assert cfg.have_credentials()
        assert client.calls >= 3
