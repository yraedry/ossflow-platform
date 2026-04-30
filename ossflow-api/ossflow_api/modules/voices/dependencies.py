"""Dependencias FastAPI del módulo voices."""

from __future__ import annotations

from .service import VoicesService


def _default_manager_factory():
    """Crea un ``VoiceProfileManager`` por request.

    Import diferido — la carpeta ``voice_profiles/`` vive en root del
    repo (no se mueve en este task).
    """
    from voice_profiles.manager import VoiceProfileManager

    return VoiceProfileManager()


def get_voices_service() -> VoicesService:
    return VoicesService(manager_factory=_default_manager_factory)
