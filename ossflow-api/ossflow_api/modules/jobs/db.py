"""Modelo SQLAlchemy ``legacy_jobs`` local al módulo ``jobs``.

**Decisión arquitectónica (anexo del spec, §1.4):** la tabla vive aquí en
lugar de en ``ossflow_service_kit.db.models`` para no acoplar el refactor
de ``ossflow-api`` a otro repo (``ossflow-core``). Es deuda explícita y
gestionada — promover a ``service_kit`` en la consolidación post-refactor.

Hereda de la misma ``Base`` declarativa del kit, así ``init_db()`` la
recoge automáticamente vía ``Base.metadata.create_all()`` siempre que
este módulo se haya importado antes (lo hace ``LegacyJobsRepository`` en
T19.5).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ossflow_service_kit.db.models import Base


class LegacyJobRow(Base):
    """Fila de la tabla ``legacy_jobs``.

    Persistencia 1:1 de ``LegacyJob`` (dataclass de dominio en ``models.py``).
    El nombre de columnas mantiene el JSON shape externo: ``job_id`` y
    ``job_type`` no se renombran a ``id``/``type`` aunque sea más SQL-idiomático.
    """

    __tablename__ = "legacy_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    job_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    video_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    progress: Mapped[Optional[float]] = mapped_column(Float)
    message: Mapped[Optional[str]] = mapped_column(Text)
    result: Mapped[Optional[str]] = mapped_column(Text)  # JSON serializado
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    __table_args__ = (
        Index("idx_legacy_jobs_status_created", "status", "created_at"),
    )
