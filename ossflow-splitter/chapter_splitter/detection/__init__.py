"""Detection subpackage: chapter detection with background memory and stability verification."""

from .background_memory import BackgroundMemory
from .detector import ChapterDetector
from .stability import StabilityVerifier

__all__ = ["BackgroundMemory", "ChapterDetector", "StabilityVerifier"]
