"""Shared fixtures for the bjj-processor test suite."""

import pytest

from subtitle_generator.config import SubtitleConfig, TranscriptionConfig

# chapter_splitter is an optional sibling package -- only load its fixtures
# when the package is actually installed (avoids ImportError when running
# subtitle-generator tests in isolation).
try:
    from chapter_splitter.config import Config as ChapterConfig
    _HAS_CHAPTER_SPLITTER = True
except ImportError:
    _HAS_CHAPTER_SPLITTER = False


# ---------------------------------------------------------------------------
# chapter_splitter fixtures (only registered when available)
# ---------------------------------------------------------------------------

if _HAS_CHAPTER_SPLITTER:
    @pytest.fixture
    def chapter_config():
        """Return a default ChapterConfig for testing."""
        return ChapterConfig()

    @pytest.fixture
    def chapter_config_custom():
        """Return a ChapterConfig with non-default values for override tests."""
        return ChapterConfig(
            root_dir="/tmp/test",
            dry_run=True,
            verbose=True,
            voice_threshold=0.5,
            ocr_confidence_min=0.6,
            roi_top_fraction=0.20,
            roi_bottom_fraction=0.20,
            background_max_entries=10,
            stability_frames=5,
        )


# ---------------------------------------------------------------------------
# subtitle_generator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def subtitle_config():
    """Return a default SubtitleConfig for testing."""
    return SubtitleConfig()


@pytest.fixture
def transcription_config():
    """Return a default TranscriptionConfig for testing."""
    return TranscriptionConfig()


@pytest.fixture
def sample_segments():
    """A list of well-formed subtitle segments."""
    return [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Hello and welcome to this instructional.",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.95},
                {"word": "and", "start": 0.5, "end": 0.6, "score": 0.90},
                {"word": "welcome", "start": 0.7, "end": 1.1, "score": 0.92},
                {"word": "to", "start": 1.2, "end": 1.3, "score": 0.88},
                {"word": "this", "start": 1.4, "end": 1.6, "score": 0.91},
                {"word": "instructional.", "start": 1.7, "end": 2.5, "score": 0.93},
            ],
        },
        {
            "start": 3.0,
            "end": 5.5,
            "text": "Today we will look at the arm drag system.",
            "words": [
                {"word": "Today", "start": 3.0, "end": 3.3, "score": 0.94},
                {"word": "we", "start": 3.4, "end": 3.5, "score": 0.89},
                {"word": "will", "start": 3.6, "end": 3.7, "score": 0.91},
                {"word": "look", "start": 3.8, "end": 4.0, "score": 0.90},
                {"word": "at", "start": 4.1, "end": 4.2, "score": 0.85},
                {"word": "the", "start": 4.3, "end": 4.4, "score": 0.87},
                {"word": "arm", "start": 4.5, "end": 4.7, "score": 0.92},
                {"word": "drag", "start": 4.8, "end": 5.0, "score": 0.93},
                {"word": "system.", "start": 5.1, "end": 5.5, "score": 0.91},
            ],
        },
        {
            "start": 6.0,
            "end": 8.0,
            "text": "The first technique is the wrist roll.",
            "words": [
                {"word": "The", "start": 6.0, "end": 6.1, "score": 0.88},
                {"word": "first", "start": 6.2, "end": 6.4, "score": 0.90},
                {"word": "technique", "start": 6.5, "end": 6.9, "score": 0.92},
                {"word": "is", "start": 7.0, "end": 7.1, "score": 0.86},
                {"word": "the", "start": 7.2, "end": 7.3, "score": 0.87},
                {"word": "wrist", "start": 7.4, "end": 7.6, "score": 0.93},
                {"word": "roll.", "start": 7.7, "end": 8.0, "score": 0.91},
            ],
        },
    ]


@pytest.fixture
def sample_words():
    """A flat list of word dicts with timestamps and scores."""
    return [
        {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.95},
        {"word": "and", "start": 0.5, "end": 0.6, "score": 0.90},
        {"word": "welcome", "start": 0.7, "end": 1.1, "score": 0.92},
        {"word": "to", "start": 1.2, "end": 1.3, "score": 0.88},
        {"word": "this", "start": 1.4, "end": 1.6, "score": 0.91},
        {"word": "instructional.", "start": 1.7, "end": 2.5, "score": 0.93},
    ]
