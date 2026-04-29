from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from chapter_splitter.oracle.errors import HTMLChangedError
from chapter_splitter.oracle.providers.bjjfanatics import (
    BJJFanaticsProvider,
    _parse_range,
    _parse_time,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- helpers
class _FakeResponse:
    def __init__(
        self,
        text: str = "",
        status_code: int = 200,
        json_data: object | None = None,
    ) -> None:
        self.text = text
        self.status_code = status_code
        self._json = json_data

    def json(self) -> object:
        if self._json is not None:
            return self._json
        return json.loads(self.text)


def _patch_get(monkeypatch, response: _FakeResponse) -> None:
    def fake_get(self, url, *args, **kwargs):  # noqa: ARG001
        return response

    monkeypatch.setattr(httpx.Client, "get", fake_get)


# ---------------------------------------------------------------- tests
def test_parse_time_variants() -> None:
    assert _parse_time("0") == 0
    assert _parse_time("1:35") == 95
    assert _parse_time("25:10") == 1510
    assert _parse_time("1:23:45") == 5025


def test_parse_range_variants() -> None:
    assert _parse_range("0 - (1:35)") == (0, 95)
    assert _parse_range("25:10 - (29:45)") == (1510, 1785)


def test_scrape_tripod(monkeypatch) -> None:
    html = (FIXTURES / "bjjfanatics_tripod.html").read_text(encoding="utf-8")
    _patch_get(monkeypatch, _FakeResponse(text=html, status_code=200))

    provider = BJJFanaticsProvider()
    result = provider.scrape(
        "https://bjjfanatics.com/products/tripod-passing"
    )

    assert result.provider_id == "bjjfanatics"
    assert len(result.volumes) >= 3

    v1 = result.volume(1)
    assert v1 is not None
    assert len(v1.chapters) == 12

    first = v1.chapters[0]
    assert first.title == "Phases Of Engagement"
    assert first.start_s == 0

    second = v1.chapters[1]
    assert second.title == "Prerequisites To Pass And How The Tripod Fits In"
    assert second.start_s == 95


def test_scrape_missing_course_content_raises(monkeypatch) -> None:
    html = (
        "<html><body>"
        "<h1 class='product-title'>Some Product</h1>"
        "<p>No course content here.</p>"
        "</body></html>"
    )
    _patch_get(monkeypatch, _FakeResponse(text=html, status_code=200))

    provider = BJJFanaticsProvider()
    with pytest.raises(HTMLChangedError):
        provider.scrape("https://bjjfanatics.com/products/x")


def test_search_returns_scored_candidates(monkeypatch) -> None:
    payload = {
        "products": [
            {
                "title": "Tripod Passing: Beating Inside Position by Jozef Chen",
                "vendor": "Jozef Chen",
                "handle": "tripod-passing-beating-inside-position-by-jozef-chen",
            },
            {
                "title": "Some Other Instructional",
                "vendor": "Other Author",
                "handle": "other-instructional",
            },
        ]
    }
    _patch_get(
        monkeypatch,
        _FakeResponse(status_code=200, json_data=payload),
    )

    provider = BJJFanaticsProvider()
    candidates = provider.search("Tripod Passing", "Jozef Chen")

    assert candidates, "expected at least one candidate"
    top = candidates[0]
    assert top.score > 0.7
    assert top.url == (
        "https://bjjfanatics.com/products/"
        "tripod-passing-beating-inside-position-by-jozef-chen"
    )
    assert top.provider_id == "bjjfanatics"
    # Sorted desc by score
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
