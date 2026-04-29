"""WhisperX Subtitle Generator package for BJJ instructional videos.

Transcribes English audio using WhisperX (large-v3) with forced word-level
alignment, applies hallucination filtering, timestamp correction, and outputs
Netflix-standard .srt subtitle files (42 chars/line, 2 lines max).

Usage:
    python -m subtitle_generator "Z:\\path\\to\\videos"
    python -m subtitle_generator "Z:\\path" --model large-v3 --batch-size 4
    python -m subtitle_generator "Z:\\path" --verbose
"""

from .config import TranscriptionConfig, SubtitleConfig
from .pipeline import SubtitlePipeline

__all__ = ["SubtitlePipeline", "TranscriptionConfig", "SubtitleConfig"]
