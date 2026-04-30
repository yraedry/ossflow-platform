"""Compat shim. La lógica vive en ``ossflow_api.shared.paths``.

Este shim mantiene la firma pública (``from api.paths import to_container_path``)
hasta que todos los importadores migren al nuevo paquete.
"""

from ossflow_api.shared.paths import (  # noqa: F401
    from_container_path,
    to_container_path,
)
