"""Compat shim. Lógica movida a ``ossflow_api.clients.{base,splitter,subtitle,dubbing}``.

Mantiene la API pública legacy para módulos que aún no han migrado.
"""

from ossflow_api.clients.base import BackendClient, BackendError, _parse_sse_block  # noqa: F401
from ossflow_api.clients.dubbing import dubbing_client
from ossflow_api.clients.dubbing import reset as _reset_dubbing
from ossflow_api.clients.splitter import reset as _reset_splitter
from ossflow_api.clients.splitter import splitter_client
from ossflow_api.clients.subtitle import reset as _reset_subs
from ossflow_api.clients.subtitle import subs_client

__all__ = [
    "BackendClient",
    "BackendError",
    "splitter_client",
    "subs_client",
    "dubbing_client",
    "reset_clients",
]


def reset_clients() -> None:
    """Test helper legacy: limpia las cachés de los tres clientes."""
    _reset_splitter()
    _reset_subs()
    _reset_dubbing()
