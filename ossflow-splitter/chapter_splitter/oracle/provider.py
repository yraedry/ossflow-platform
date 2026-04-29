from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import Candidate, OracleResult


@runtime_checkable
class OracleProvider(Protocol):
    """Contract for a site-specific oracle source.

    Implementations live in chapter_splitter/oracle/providers/*.py and are
    auto-registered at import time via ProviderRegistry.
    """

    id: str
    display_name: str
    domains: list[str]

    def search(self, title: str, author: str | None = None) -> list[Candidate]:
        """Return product URL candidates ranked by score (desc)."""
        ...

    def scrape(self, url: str) -> OracleResult:
        """Fetch the product page and return structured chapters/volumes."""
        ...
