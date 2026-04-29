"""Tests for /api/oracle/* router (proxy + sidecar persistence)."""

from __future__ import annotations

import json
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("SPLITTER_URL", "http://chapter-splitter:8001")

    import importlib

    import api.settings as settings_mod
    importlib.reload(settings_mod)
    settings_mod.save_settings({**settings_mod.load_settings(), "library_path": str(library_dir)})

    import api.oracle as oracle_mod
    importlib.reload(oracle_mod)

    app = FastAPI()
    app.include_router(oracle_mod.router)

    instructional = library_dir / "John Danaher - Tripod Passing"
    instructional.mkdir()

    return {
        "client": TestClient(app),
        "library": library_dir,
        "instructional": instructional,
        "oracle_mod": oracle_mod,
        "encoded_path": quote(str(instructional), safe=""),
    }


def _patch_httpx(monkeypatch, oracle_mod, handler):
    """Replace httpx.AsyncClient with a transport-backed client."""
    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(oracle_mod.httpx, "AsyncClient", _Client)


# ---------------------------------------------------------------------------
# GET /providers
# ---------------------------------------------------------------------------

def test_providers_happy(env, monkeypatch):
    body = [{"id": "bjjfanatics", "display_name": "BJJ Fanatics", "domains": ["bjjfanatics.com"]}]

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/oracle/providers"
        return httpx.Response(200, json=body)

    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].get("/api/oracle/providers")
    assert r.status_code == 200
    assert r.json() == body


def test_providers_backend_500(env, monkeypatch):
    def handler(request):
        return httpx.Response(500, text="boom")
    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].get("/api/oracle/providers")
    assert r.status_code == 502


def test_providers_invalid_json(env, monkeypatch):
    def handler(request):
        return httpx.Response(200, text="<html>not json</html>")
    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].get("/api/oracle/providers")
    assert r.status_code == 502


# ---------------------------------------------------------------------------
# GET /{path}
# ---------------------------------------------------------------------------

def test_get_oracle_404_when_no_meta(env):
    r = env["client"].get(f"/api/oracle/{env['encoded_path']}")
    assert r.status_code == 404


def test_get_oracle_returns_cached(env):
    oracle = {"product_url": "https://x", "scraped_at": "2026-04-13T00:00:00Z", "volumes": []}
    (env["instructional"] / ".bjj-meta.json").write_text(json.dumps({"oracle": oracle}))
    r = env["client"].get(f"/api/oracle/{env['encoded_path']}")
    assert r.status_code == 200
    assert r.json() == oracle


def test_get_oracle_path_traversal_denied(env):
    bad = quote("/etc/passwd", safe="")
    r = env["client"].get(f"/api/oracle/{bad}")
    assert r.status_code in (403, 404)


# ---------------------------------------------------------------------------
# POST /{path}/resolve
# ---------------------------------------------------------------------------

def test_resolve_proxies_with_derived_title_author(env, monkeypatch):
    captured = {}

    def handler(request):
        assert request.url.path == "/oracle/search"
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=[{"url": "https://x", "score": 0.9}])

    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].post(
        f"/api/oracle/{env['encoded_path']}/resolve",
        json={"provider_id": "bjjfanatics"},
    )
    assert r.status_code == 200
    assert r.json() == [{"url": "https://x", "score": 0.9}]
    assert captured["body"]["provider_id"] == "bjjfanatics"
    # Derived from "John Danaher - Tripod Passing"
    assert captured["body"]["author"] == "John Danaher"
    assert captured["body"]["title"] == "Tripod Passing"


def test_resolve_uses_meta_when_available(env, monkeypatch):
    (env["instructional"] / ".bjj-meta.json").write_text(
        json.dumps({"instructor": "Gordon", "topic": "Leglocks"})
    )
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].post(f"/api/oracle/{env['encoded_path']}/resolve", json={})
    assert r.status_code == 200
    assert captured["body"]["author"] == "Gordon"
    assert captured["body"]["title"] == "Leglocks"
    assert "provider_id" not in captured["body"]


def test_resolve_backend_error(env, monkeypatch):
    def handler(request):
        return httpx.Response(500, text="x")
    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].post(f"/api/oracle/{env['encoded_path']}/resolve", json={})
    assert r.status_code == 502


# ---------------------------------------------------------------------------
# POST /{path}/scrape
# ---------------------------------------------------------------------------

VALID_ORACLE = {
    "product_url": "https://bjjfanatics.com/products/foo",
    "scraped_at": "2026-04-13T00:00:00Z",
    "volumes": [
        {
            "number": 1,
            "total_duration_s": 600.0,
            "chapters": [
                {"title": "Intro", "start_s": 0.0, "end_s": 60.0},
                {"title": "Drill", "start_s": 60.0, "end_s": 600.0},
            ],
        }
    ],
}


def test_scrape_persists_to_meta(env, monkeypatch):
    def handler(request):
        assert request.url.path == "/oracle/scrape"
        assert json.loads(request.content) == {"url": "https://bjjfanatics.com/products/foo"}
        return httpx.Response(200, json=VALID_ORACLE)

    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].post(
        f"/api/oracle/{env['encoded_path']}/scrape",
        json={"url": "https://bjjfanatics.com/products/foo"},
    )
    assert r.status_code == 200
    assert r.json() == VALID_ORACLE

    meta = json.loads((env["instructional"] / ".bjj-meta.json").read_text())
    assert meta["oracle"] == VALID_ORACLE
    assert meta["url_bjjfanatics"] == "https://bjjfanatics.com/products/foo"


def test_scrape_missing_url(env):
    r = env["client"].post(f"/api/oracle/{env['encoded_path']}/scrape", json={})
    assert r.status_code == 422


def test_scrape_backend_404(env, monkeypatch):
    def handler(request):
        return httpx.Response(404, text="not found")
    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].post(
        f"/api/oracle/{env['encoded_path']}/scrape",
        json={"url": "https://x"},
    )
    assert r.status_code == 502


def test_scrape_backend_invalid_json(env, monkeypatch):
    def handler(request):
        return httpx.Response(200, text="oops")
    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].post(
        f"/api/oracle/{env['encoded_path']}/scrape",
        json={"url": "https://x"},
    )
    assert r.status_code == 502


def test_scrape_invalid_oracle_shape(env, monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"product_url": 123, "volumes": []})
    _patch_httpx(monkeypatch, env["oracle_mod"], handler)
    r = env["client"].post(
        f"/api/oracle/{env['encoded_path']}/scrape",
        json={"url": "https://x"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# PUT /{path}
# ---------------------------------------------------------------------------

def test_put_persists_oracle(env):
    r = env["client"].put(f"/api/oracle/{env['encoded_path']}", json=VALID_ORACLE)
    assert r.status_code == 200
    meta = json.loads((env["instructional"] / ".bjj-meta.json").read_text())
    assert meta["oracle"]["volumes"][0]["chapters"][0]["title"] == "Intro"
    assert meta["url_bjjfanatics"] == VALID_ORACLE["product_url"]


def test_put_rejects_invalid_payload(env):
    r = env["client"].put(
        f"/api/oracle/{env['encoded_path']}",
        json={"product_url": "x", "scraped_at": "y", "volumes": [{"number": "bad", "chapters": []}]},
    )
    assert r.status_code == 422


def test_put_invalid_json_body(env):
    r = env["client"].put(
        f"/api/oracle/{env['encoded_path']}",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /{path}
# ---------------------------------------------------------------------------

def test_delete_preserves_other_fields(env):
    (env["instructional"] / ".bjj-meta.json").write_text(json.dumps({
        "instructor": "John",
        "topic": "Tripod",
        "tags": ["bjj"],
        "oracle": VALID_ORACLE,
        "url_bjjfanatics": "https://x",
    }))
    r = env["client"].delete(f"/api/oracle/{env['encoded_path']}")
    assert r.status_code == 200
    meta = json.loads((env["instructional"] / ".bjj-meta.json").read_text())
    assert "oracle" not in meta
    assert meta["instructor"] == "John"
    assert meta["topic"] == "Tripod"
    assert meta["tags"] == ["bjj"]
    # url_bjjfanatics intentionally preserved (only "oracle" removed)
    assert meta["url_bjjfanatics"] == "https://x"


def test_delete_when_no_meta_is_ok(env):
    r = env["client"].delete(f"/api/oracle/{env['encoded_path']}")
    assert r.status_code == 200
    assert not (env["instructional"] / ".bjj-meta.json").exists()


def test_delete_unknown_instructional_404(env):
    bad = quote(str(env["library"] / "does-not-exist"), safe="")
    r = env["client"].delete(f"/api/oracle/{bad}")
    assert r.status_code == 404
