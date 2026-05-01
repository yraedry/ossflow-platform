"""Tests del router del módulo health vía FastAPI TestClient."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ossflow_api.modules.health import health_router
from ossflow_api.modules.health.dependencies import get_health_service
from ossflow_api.modules.health.service import HealthService


class _StubService(HealthService):
    """HealthService con respuestas precocinadas para tests del router."""

    def __init__(self, *, all_payload: dict, one_payload: dict, known: bool) -> None:
        self._all_payload = all_payload
        self._one_payload = one_payload
        self._known = known

    async def ping_all(self) -> dict:
        return self._all_payload

    async def ping_one(self, service: str) -> dict:
        return self._one_payload

    def is_known(self, service: str) -> bool:
        return self._known


def _build_app(stub: _StubService) -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    app.dependency_overrides[get_health_service] = lambda: stub
    return app


def test_get_backends_returns_aggregated_payload() -> None:
    payload = {"services": [{"service": "foo", "status": "up"}]}
    stub = _StubService(all_payload=payload, one_payload={}, known=False)
    client = TestClient(_build_app(stub))

    response = client.get("/api/health/backends")

    assert response.status_code == 200
    assert response.json() == payload


def test_get_unknown_service_returns_404() -> None:
    stub = _StubService(all_payload={}, one_payload={}, known=False)
    client = TestClient(_build_app(stub))

    response = client.get("/api/health/missing")

    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


def test_get_known_service_returns_payload() -> None:
    payload = {"service": "foo", "status": "up"}
    stub = _StubService(all_payload={}, one_payload=payload, known=True)
    client = TestClient(_build_app(stub))

    response = client.get("/api/health/foo")

    assert response.status_code == 200
    assert response.json() == payload
