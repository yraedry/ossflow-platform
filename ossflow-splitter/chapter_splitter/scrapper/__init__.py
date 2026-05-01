from .models import OracleChapter, OracleVolume, OracleResult, Candidate
from .provider import OracleProvider
from .errors import (
    OracleError,
    ProviderNotFoundError,
    ProviderSearchError,
    ProviderScrapeError,
    ProviderTimeoutError,
    HTMLChangedError,
    OracleValidationError,
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
    "OracleChapter",
    "OracleVolume",
    "OracleResult",
    "Candidate",
    "OracleProvider",
    "OracleError",
    "ProviderNotFoundError",
    "ProviderSearchError",
    "ProviderScrapeError",
    "ProviderTimeoutError",
    "HTMLChangedError",
    "OracleValidationError",
    "ProviderRegistry",
    "registry",
    "discover",
]
