"""Servicio del módulo voices.

Envuelve ``voice_profiles.manager.VoiceProfileManager`` exponiendo las
tres operaciones que consume el frontend (list/create/delete). El
manager se inyecta vía factory para que los tests puedan reemplazarlo
sin tocar el filesystem real (``PROFILES_DIR`` / ``REGISTRY_FILE``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from ossflow_api.shared.exceptions import ApiError, NotFoundError, ValidationError

log = logging.getLogger(__name__)


# Factory: callable sin argumentos que devuelva una instancia con la
# misma interfaz pública que ``VoiceProfileManager``.
ManagerFactory = Callable[[], Any]


class VoicesService:
    """Operaciones sobre perfiles de voz."""

    def __init__(self, *, manager_factory: ManagerFactory) -> None:
        self._manager_factory = manager_factory

    # ------------------------------------------------------------------
    # Listado
    # ------------------------------------------------------------------

    def list_profiles(self) -> dict[str, Any]:
        """Devuelve ``{"profiles": [...]}``. Errores propagan como ApiError."""
        try:
            mgr = self._manager_factory()
            profiles = mgr.list_profiles()
            return {"profiles": [p.to_dict() for p in profiles]}
        except Exception as exc:  # noqa: BLE001
            log.error("list_voice_profiles failed: %s", exc)
            raise ApiError(str(exc), status_code=500) from exc

    # ------------------------------------------------------------------
    # Creación
    # ------------------------------------------------------------------

    def create_profile(
        self,
        *,
        video_path: str,
        instructor: str,
        start_sec: float,
        duration: float,
    ) -> dict[str, Any]:
        """Extrae y guarda el sample de voz para ``instructor``.

        Lanza ``ValidationError`` (campos faltantes, video inexistente),
        ``NotFoundError`` (extracción ffmpeg sin fichero), ``ApiError``
        para fallos genéricos.
        """
        if not video_path or not instructor:
            raise ValidationError("Missing 'video_path' or 'instructor'")
        if not Path(video_path).exists():
            raise NotFoundError("Video file not found")

        try:
            mgr = self._manager_factory()
            profile = mgr.extract_sample(
                Path(video_path), instructor,
                start_sec=start_sec, duration=duration,
            )
            return {"ok": True, "profile": profile.to_dict()}
        except FileNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError(str(exc), status_code=500) from exc
        except Exception as exc:  # noqa: BLE001
            log.error("create_voice_profile failed: %s", exc)
            raise ApiError(str(exc), status_code=500) from exc

    # ------------------------------------------------------------------
    # Borrado
    # ------------------------------------------------------------------

    def delete_profile(self, instructor: str) -> dict[str, Any]:
        """Borra el perfil; lanza ``NotFoundError`` si no existe."""
        try:
            mgr = self._manager_factory()
            deleted = mgr.delete_profile(instructor)
        except Exception as exc:  # noqa: BLE001
            log.error("delete_voice_profile failed: %s", exc)
            raise ApiError(str(exc), status_code=500) from exc
        if not deleted:
            raise NotFoundError(f"No profile found for '{instructor}'")
        return {"ok": True, "message": f"Deleted profile for '{instructor}'"}
