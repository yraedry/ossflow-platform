"""Módulo preflight: checks previos al pipeline (path, disco, ffmpeg, GPU, backends)."""

from .router import router as preflight_router
from .service import PreflightService

__all__ = ["preflight_router", "PreflightService"]
