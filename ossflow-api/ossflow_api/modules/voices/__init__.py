"""Módulo voices: gestión de perfiles de voz por instructor.

Wrapper sobre ``voice_profiles.manager.VoiceProfileManager`` (carpeta
auxiliar en root del repo) — sin moverlo todavía.
"""

from .router import router as voices_router

__all__ = ["voices_router"]
