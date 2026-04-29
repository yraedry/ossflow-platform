"""Dubbing Generator -- Doblaje al castellano con Coqui XTTS v2."""

__version__ = "2.0.0"

from .config import DubbingConfig
from .pipeline import DubbingPipeline

__all__ = ["DubbingConfig", "DubbingPipeline", "__version__"]
