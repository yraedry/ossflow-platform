class ScraperError(Exception):
    """Base class for all scrapper-related errors."""


class ProviderNotFoundError(ScraperError):
    """No provider matches the given id or URL domain."""


class ProviderSearchError(ScraperError):
    """search() failed (network, parse, no results with min score)."""


class ProviderScrapeError(ScraperError):
    """scrape() failed (network, HTML structure unexpected)."""


class ProviderTimeoutError(ScraperError):
    """HTTP timeout during search or scrape."""


class HTMLChangedError(ProviderScrapeError):
    """Expected selectors not found - site structure likely changed."""


class ScraperValidationError(ScraperError):
    """Scraped data failed pydantic validation."""
