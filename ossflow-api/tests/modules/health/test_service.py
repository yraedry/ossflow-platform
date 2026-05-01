"""Tests unitarios para HealthService."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from ossflow_api.modules.health.service import HealthService


@pytest.mark.asyncio
@respx.mock
async def test_ping_one_returns_up_when_backend_responds_200() -> None:
    backends = {"foo": "http://foo.test:9000"}
    respx.get("http://foo.test:9000/health").mock(
        return_value=Response(200, json={"ok": True})
    )
    svc = HealthService(backends)

    result = await svc.ping_one("foo")

    assert result == {"service": "foo", "status": "up", "body": {"ok": True}}


@pytest.mark.asyncio
@respx.mock
async def test_ping_one_returns_down_on_http_error() -> None:
    backends = {"foo": "http://foo.test:9000"}
    respx.get("http://foo.test:9000/health").mock(return_value=Response(500))
    svc = HealthService(backends)

    result = await svc.ping_one("foo")

    assert result["service"] == "foo"
    assert result["status"] == "down"
    assert "HTTP 500" in result["error"]


@pytest.mark.asyncio
async def test_ping_one_returns_unknown_for_unregistered_service() -> None:
    svc = HealthService({"foo": "http://foo.test"})

    result = await svc.ping_one("missing")

    assert result == {"service": "missing", "status": "unknown"}


@pytest.mark.asyncio
@respx.mock
async def test_ping_all_aggregates_responses() -> None:
    backends = {
        "foo": "http://foo.test",
        "bar": "http://bar.test",
    }
    respx.get("http://foo.test/health").mock(return_value=Response(200, json={}))
    respx.get("http://bar.test/health").mock(return_value=Response(503))
    svc = HealthService(backends)

    result = await svc.ping_all()

    assert {r["service"] for r in result["services"]} == {"foo", "bar"}
    statuses = {r["service"]: r["status"] for r in result["services"]}
    assert statuses == {"foo": "up", "bar": "down"}


@pytest.mark.asyncio
@respx.mock
async def test_ping_one_uses_api_tags_for_ollama() -> None:
    """Ollama no expone /health, usa /api/tags como liveness probe."""
    backends = {"ollama": "http://ollama.test"}
    respx.get("http://ollama.test/api/tags").mock(
        return_value=Response(200, json={"models": []})
    )
    svc = HealthService(backends)

    result = await svc.ping_one("ollama")

    assert result["status"] == "up"


def test_is_known_returns_true_for_registered() -> None:
    svc = HealthService({"foo": "http://foo"})
    assert svc.is_known("foo") is True
    assert svc.is_known("bar") is False
