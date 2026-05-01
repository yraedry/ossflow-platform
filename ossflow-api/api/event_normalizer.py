"""Compat shim. La lógica vive en ``ossflow_api.shared.events``."""

from ossflow_api.shared.events import (  # noqa: F401
    NormalizedEvent,
    is_terminal,
    normalize,
)
