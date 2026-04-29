"""Tests for audio stretcher (trim_silence, stretch_audio)."""

from unittest.mock import patch, MagicMock

from pydub import AudioSegment
from pydub.generators import Sine

from dubbing_generator.audio.stretcher import trim_silence, stretch_audio


def _make_tone(duration_ms: int = 1000, freq: int = 440) -> AudioSegment:
    """Create a short sine wave for testing."""
    return Sine(freq).to_audio_segment(duration=duration_ms).set_frame_rate(24000)


def _make_silent(duration_ms: int = 500) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms)


class TestTrimSilence:
    def test_trims_leading_silence(self):
        audio = _make_silent(200) + _make_tone(500)
        trimmed = trim_silence(audio, threshold_db=-40.0, chunk_ms=10)
        # Trimmed should be shorter (leading silence removed)
        assert len(trimmed) < len(audio)

    def test_trims_trailing_silence(self):
        audio = _make_tone(500) + _make_silent(200)
        trimmed = trim_silence(audio, threshold_db=-40.0, chunk_ms=10)
        assert len(trimmed) < len(audio)

    def test_returns_original_if_all_silent(self):
        audio = _make_silent(300)
        trimmed = trim_silence(audio, threshold_db=-20.0, chunk_ms=10)
        # Should return original, not empty
        assert len(trimmed) == len(audio)

    def test_no_change_if_no_silence(self):
        audio = _make_tone(500)
        trimmed = trim_silence(audio, threshold_db=-60.0, chunk_ms=10)
        # With a very low threshold, nothing should be trimmed
        assert len(trimmed) == len(audio)


class TestStretchAudio:
    def test_returns_as_is_if_fits(self):
        audio = _make_tone(500)
        result = stretch_audio(audio, target_duration_ms=1000, max_ratio=1.5)
        # Audio is 500ms and target is 1000ms -- no compression needed
        assert len(result) <= 1000

    def test_max_ratio_clamped_at_1_5(self):
        """Even if audio is 3x too long, max ratio should be 1.5."""
        audio = _make_tone(3000)  # 3 seconds
        with patch(
            "dubbing_generator.audio.stretcher._atempo_ffmpeg"
        ) as mock_atempo:
            # Make atempo return a shorter segment
            def fake_atempo(inp, out, ratio):
                _make_tone(2000).export(out, format="wav")

            mock_atempo.side_effect = fake_atempo
            result = stretch_audio(audio, target_duration_ms=1000, max_ratio=1.5)

            # Verify atempo was called with ratio clamped to 1.5
            mock_atempo.assert_called_once()
            actual_ratio = mock_atempo.call_args[0][2]
            assert actual_ratio == 1.5

    def test_empty_audio_returns_empty(self):
        audio = AudioSegment.empty()
        result = stretch_audio(audio, target_duration_ms=1000)
        assert len(result) == 0
