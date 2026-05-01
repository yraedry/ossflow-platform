from .models import ScrapeChapter, ScrapeVolume, ScrapeResult, Candidate
from .provider import ScrapeProvider
from .errors import (
    ScraperError,
    ProviderNotFoundError,
    ProviderSearchError,
    ProviderScrapeError,
    ProviderTimeoutError,
    HTMLChangedError,
    ScraperValidationError,
)
from .registry import ProviderRegistry, registry


def discover() -> None:
    """Import all provider modules so they self-register.

    Opt-in: callers (CLI / FastAPI startup) invoke this once. We don't auto-run
    on package import to avoid side effects (network clients, heavy deps) in
    contexts that only need the models/types.
    """
    registry.discover()

__all__ = [
    "ScrapeChapter",
    "ScrapeVolume",
    "ScrapeResult",
    "Candidate",
    "ScrapeProvider",
    "ScraperError",
    "ProviderNotFoundError",
    "ProviderSearchError",
    "ProviderScrapeError",
    "ProviderTimeoutError",
    "HTMLChangedError",
    "ScraperValidationError",
    "ProviderRegistry",
    "registry",
    "discover",
]
