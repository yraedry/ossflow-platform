"""Adaptador de la capa de datos (SQLAlchemy 2.0 sobre SQLite).

Re-exporta las primitivas del kit compartido y expone una dependencia
``get_session`` lista para FastAPI ``Depends()``.
"""

from __future__ import annotations

from typing import Iterator

from ossflow_service_kit.db import init_db, session_scope

__all__ = ["init_db", "session_scope", "get_session"]


def get_session() -> Iterator:
    """Dependencia FastAPI que entrega una sesión SQLAlchemy con scope de request.

    Uso::

        def endpoint(session = Depends(get_session)):
            ...
    """
    with session_scope() as session:
        yield session
