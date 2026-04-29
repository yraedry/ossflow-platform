import pytest
from pydantic import ValidationError

from chapter_splitter.oracle import (
    Candidate,
    OracleChapter,
    OracleResult,
    OracleVolume,
    ProviderNotFoundError,
    ProviderRegistry,
)


def _chapter(t: str, s: float, e: float) -> OracleChapter:
    return OracleChapter(title=t, start_s=s, end_s=e)


def test_chapter_rejects_non_positive_duration():
    with pytest.raises(ValidationError):
        OracleChapter(title="x", start_s=10.0, end_s=10.0)
    with pytest.raises(ValidationError):
        OracleChapter(title="x", start_s=10.0, end_s=5.0)


def test_volume_requires_monotonic_starts():
    with pytest.raises(ValidationError):
        OracleVolume(
            number=1,
            total_duration_s=100,
            chapters=[_chapter("a", 10, 20), _chapter("b", 5, 15)],
        )


def test_volume_number_bounds():
    with pytest.raises(ValidationError):
        OracleVolume(number=0, total_duration_s=10, chapters=[_chapter("a", 0, 5)])
    with pytest.raises(ValidationError):
        OracleVolume(number=51, total_duration_s=10, chapters=[_chapter("a", 0, 5)])


def test_result_rejects_duplicate_volume_numbers():
    v1 = OracleVolume(number=1, total_duration_s=100, chapters=[_chapter("a", 0, 100)])
    v2 = OracleVolume(number=1, total_duration_s=200, chapters=[_chapter("b", 0, 200)])
    with pytest.raises(ValidationError):
        OracleResult(
            product_url="https://x/", provider_id="bjjfanatics", volumes=[v1, v2]
        )


def test_result_volume_lookup():
    v1 = OracleVolume(number=1, total_duration_s=100, chapters=[_chapter("a", 0, 100)])
    v2 = OracleVolume(number=2, total_duration_s=200, chapters=[_chapter("b", 0, 200)])
    r = OracleResult(
        product_url="https://x/", provider_id="bjjfanatics", volumes=[v1, v2]
    )
    assert r.volume(1) is v1
    assert r.volume(3) is None


def test_candidate_score_range():
    Candidate(url="https://x/", title="t", score=0.5, provider_id="p")
    with pytest.raises(ValidationError):
        Candidate(url="https://x/", title="t", score=1.1, provider_id="p")


class _FakeProvider:
    id = "fake"
    display_name = "Fake"
    domains = ["example.com", "example.org"]

    def search(self, title, author=None):  # pragma: no cover
        return []

    def scrape(self, url):  # pragma: no cover
        raise NotImplementedError


def test_registry_get_and_resolve_by_url():
    reg = ProviderRegistry()
    p = _FakeProvider()
    reg.register(p)

    assert reg.get("fake") is p
    assert reg.resolve_by_url("https://example.com/x") is p
    assert reg.resolve_by_url("https://sub.example.org/y") is p

    with pytest.raises(ProviderNotFoundError):
        reg.get("nope")
    with pytest.raises(ProviderNotFoundError):
        reg.resolve_by_url("https://other.com/z")
