class OracleError(Exception):
    """Base class for all oracle-related errors."""


class ProviderNotFoundError(OracleError):
    """No provider matches the given id or URL domain."""


class ProviderSearchError(OracleError):
    """search() failed (network, parse, no results with min score)."""


class ProviderScrapeError(OracleError):
    """scrape() failed (network, HTML structure unexpected)."""


class ProviderTimeoutError(OracleError):
    """HTTP timeout during search or scrape."""


class HTMLChangedError(ProviderScrapeError):
    """Expected selectors not found — site structure likely changed."""


class OracleValidationError(OracleError):
    """Scraped data failed pydantic validation."""
