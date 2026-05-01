from __future__ import annotations

import importlib
import logging
import pkgutil
from urllib.parse import urlparse

from .errors import ProviderNotFoundError
from .provider import OracleProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, OracleProvider] = {}

    def register(self, provider: OracleProvider) -> None:
        if provider.id in self._providers:
            logger.debug("provider %s already registered, replacing", provider.id)
        self._providers[provider.id] = provider

    def get(self, provider_id: str) -> OracleProvider:
        if provider_id not in self._providers:
            raise ProviderNotFoundError(f"unknown provider: {provider_id}")
        return self._providers[provider_id]

    def resolve_by_url(self, url: str) -> OracleProvider:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise ProviderNotFoundError(f"cannot parse host from url: {url}")
        for p in self._providers.values():
            for d in p.domains:
                if host == d.lower() or host.endswith("." + d.lower()):
                    return p
        raise ProviderNotFoundError(f"no provider registered for host: {host}")

    def all(self) -> list[OracleProvider]:
        return list(self._providers.values())

    def discover(self) -> None:
        """Import all modules under oracle.providers so they self-register."""
        try:
            from . import providers as _providers_pkg
        except ImportError:
            logger.debug("oracle.providers package not present yet")
            return
        for mod_info in pkgutil.iter_modules(_providers_pkg.__path__):
            name = f"{_providers_pkg.__name__}.{mod_info.name}"
            try:
                importlib.import_module(name)
            except Exception:
                logger.exception("failed importing provider module %s", name)


registry = ProviderRegistry()
