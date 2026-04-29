"""Tests for voice detection hysteresis in AudioAnalyzer."""

import numpy as np
import pytest

from chapter_splitter.config import Config
from chapter_splitter.audio.analyzer import AudioAnalyzer


class TestTransitionPeaks:
    """Unit tests for is_transition_window / _compute_transition_stats."""

    def _build(self, voice_map):
        cfg = Config(voice_window_ms=100)
        obj = object.__new__(AudioAnalyzer)
        obj.cfg = cfg
        obj.device = "cpu"
        obj.separator = None
        obj._voice_map = voice_map
        obj._voice_state = {}
        obj._sample_rate = 44100
        obj._window_samples = int(44100 * 0.1)
        obj._smoothing_windows = 1
        obj._rms_mean = 0.0
        obj._rms_std = 0.0
        obj._transition_peaks = []
        return obj

    def test_empty_map_no_peaks(self):
        an = self._build({})
        an._compute_transition_stats()
        assert an._transition_peaks == []
        assert an.is_transition_window(10.0) is False

    def test_detects_single_peak_above_mean_plus_2sigma(self):
        # 99 baseline windows at 0.1, one at 1.0 -> clear outlier.
        voice_map = {i: 0.1 for i in range(99)}
        voice_map[50] = 1.0
        an = self._build(voice_map)
        an._compute_transition_stats()
        # Peak at index 50 corresponds to 50 * 0.1s = 5.0s
        assert any(abs(p - 5.0) < 1e-6 for p in an._transition_peaks)

    def test_is_transition_window_matches_within_window(self):
        an = self._build({})
        an._transition_peaks = [5.0, 12.0]
        assert an.is_transition_window(5.5, window_s=2.0) is True
        assert an.is_transition_window(7.5, window_s=2.0) is False
        assert an.is_transition_window(12.1, window_s=2.0) is True

    def test_ignores_unknown_sentinels(self):
        voice_map = {0: -1.0, 1: -1.0, 2: 0.5}
        an = self._build(voice_map)
        an._compute_transition_stats()
        # With a single valid sample, std=0; peak threshold equals the sample.
        # That's fine -- just verify we didn't crash on -1.0 sentinels.
        assert isinstance(an._transition_peaks, list)


class TestHysteresisLogic:
    """Test the hysteresis state machine directly by populating internal maps."""

    @pytest.fixture
    def analyzer(self):
        """Create an AudioAnalyzer without loading Demucs (mock-free).

        We test the hysteresis logic by directly calling _apply_hysteresis
        and checking results on _voice_map / _voice_state.
        """
        cfg = Config(
            voice_enter_threshold=0.30,
            voice_exit_threshold=0.20,
            voice_window_ms=100,
            voice_smoothing_ms=100,  # no smoothing for test clarity
        )
        # We cannot instantiate AudioAnalyzer directly because __init__
        # loads Demucs. Instead we build the object manually.
        obj = object.__new__(AudioAnalyzer)
        obj.cfg = cfg
        obj.device = "cpu"
        obj.separator = None
        obj._voice_map = {}
        obj._voice_state = {}
        obj._sample_rate = 44100
        obj._window_samples = int(44100 * 0.1)  # 100ms
        obj._smoothing_windows = 1
        return obj

    def test_above_enter_threshold_is_voice(self, analyzer):
        """RMS above enter_threshold (0.30) should be classified as voice."""
        rms = np.array([0.35, 0.40, 0.50])
        analyzer._apply_hysteresis(0, rms)
        assert analyzer._voice_state[0] is True
        assert analyzer._voice_state[1] is True
        assert analyzer._voice_state[2] is True

    def test_below_exit_threshold_is_silence(self, analyzer):
        """RMS below exit_threshold (0.20) should be classified as silence."""
        rms = np.array([0.10, 0.15, 0.05])
        analyzer._apply_hysteresis(0, rms)
        assert analyzer._voice_state[0] is False
        assert analyzer._voice_state[1] is False
        assert analyzer._voice_state[2] is False

    def test_hysteresis_zone_carries_forward(self, analyzer):
        """RMS in the hysteresis zone (0.20-0.30) should carry forward state."""
        # Start with voice, then enter hysteresis zone
        rms = np.array([0.35, 0.25, 0.25, 0.25])
        analyzer._apply_hysteresis(0, rms)
        assert analyzer._voice_state[0] is True   # above enter
        assert analyzer._voice_state[1] is True   # zone, carries True
        assert analyzer._voice_state[2] is True   # zone, carries True
        assert analyzer._voice_state[3] is True   # zone, carries True

    def test_hysteresis_zone_from_silence(self, analyzer):
        """Hysteresis zone should carry silence forward if coming from silence."""
        rms = np.array([0.10, 0.25, 0.25])
        analyzer._apply_hysteresis(0, rms)
        assert analyzer._voice_state[0] is False  # below exit
        assert analyzer._voice_state[1] is False  # zone, carries False
        assert analyzer._voice_state[2] is False  # zone, carries False

    def test_transition_silence_to_voice(self, analyzer):
        """Transition from silence through hysteresis to voice."""
        rms = np.array([0.10, 0.25, 0.35])
        analyzer._apply_hysteresis(0, rms)
        assert analyzer._voice_state[0] is False  # silence
        assert analyzer._voice_state[1] is False  # hysteresis, carries silence
        assert analyzer._voice_state[2] is True   # above enter = voice

    def test_transition_voice_to_silence(self, analyzer):
        """Transition from voice through hysteresis to silence."""
        rms = np.array([0.40, 0.25, 0.15])
        analyzer._apply_hysteresis(0, rms)
        assert analyzer._voice_state[0] is True   # voice
        assert analyzer._voice_state[1] is True   # hysteresis, carries voice
        assert analyzer._voice_state[2] is False  # below exit = silence

    def test_no_previous_state_in_zone_is_none(self, analyzer):
        """First window in hysteresis zone with no prior state -> None."""
        rms = np.array([0.25])
        analyzer._apply_hysteresis(0, rms)
        assert analyzer._voice_state[0] is None

    def test_voice_confidence_returns_float(self, analyzer):
        """get_voice_confidence should return a float in [0.0, 1.0]."""
        rms = np.array([0.10, 0.25, 0.40])
        analyzer._apply_hysteresis(0, rms)
        # Window 0 -> second 0 (at 100ms windows, 10 windows per second)
        # We need to populate enough windows for a full second
        # Populate windows 0-9 for second 0
        for i in range(10):
            analyzer._voice_map[i] = 0.10
            analyzer._voice_state[i] = False

        conf = analyzer.get_voice_confidence(0)
        assert 0.0 <= conf <= 1.0
        # Low RMS -> low confidence
        assert conf < 0.4

    def test_voice_confidence_high_rms(self, analyzer):
        """High RMS should give high confidence."""
        for i in range(10):
            analyzer._voice_map[i] = 0.50
            analyzer._voice_state[i] = True

        conf = analyzer.get_voice_confidence(0)
        assert conf > 0.7

    def test_voice_confidence_unknown(self, analyzer):
        """Unknown RMS should return 0.5 (uncertain)."""
        # No data populated
        conf = analyzer.get_voice_confidence(99)
        assert conf == pytest.approx(0.5)

    def test_is_silent_majority_vote(self, analyzer):
        """is_silent should use majority vote across windows in a second."""
        # 10 windows per second at 100ms
        # 6 silent, 4 voice -> majority silent
        for i in range(6):
            analyzer._voice_map[i] = 0.10
            analyzer._voice_state[i] = False
        for i in range(6, 10):
            analyzer._voice_map[i] = 0.40
            analyzer._voice_state[i] = True

        assert analyzer.is_silent(0) is True

    def test_is_uncertain_majority_unknown(self, analyzer):
        """is_uncertain should return True if majority of windows are None."""
        # 10 windows per second, 6 unknown
        for i in range(4):
            analyzer._voice_map[i] = 0.40
            analyzer._voice_state[i] = True
        for i in range(4, 10):
            analyzer._voice_map[i] = -1.0
            analyzer._voice_state[i] = None

        assert analyzer.is_uncertain(0) is True
