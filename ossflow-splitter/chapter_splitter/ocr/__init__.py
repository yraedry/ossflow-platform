"""OCR subpackage: frame preprocessing and text reading."""

from .preprocessor import FramePreprocessor
from .reader import OcrReader

__all__ = ["FramePreprocessor", "OcrReader"]
