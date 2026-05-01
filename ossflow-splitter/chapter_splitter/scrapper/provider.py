from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import Candidate, ScrapeResult


@runtime_checkable
class ScrapeProvider(Protocol):
    """Contract for a site-specific scrapper source.

    Implementations live in chapter_splitter/scrapper/providers/*.py and are
    auto-registered at import time via ProviderRegistry.
    """

    id: str
    display_name: str
    domains: list[str]

    def search(self, title: str, author: str | None = None) -> list[Candidate]:
        """Return product URL candidates ranked by score (desc)."""
        ...

    def scrape(self, url: str) -> ScrapeResult:
        """Fetch the product page and return structured chapters/volumes."""
        ...
