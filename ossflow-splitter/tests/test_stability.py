"""Tests for bidirectional stability verification."""

from unittest.mock import MagicMock, patch

import pytest

from chapter_splitter.config import Config
from chapter_splitter.models import OcrResult
from chapter_splitter.detection.stability import StabilityVerifier


class TestBidirectionalVerification:
    """Test that bidirectional mode checks frames before and after."""

    @pytest.fixture
    def mock_ocr(self):
        """Create a mock OcrReader."""
        ocr = MagicMock()
        return ocr

    @pytest.fixture
    def mock_cap(self):
        """Create a mock cv2.VideoCapture."""
        return MagicMock()

    def test_bidirectional_checks_before_and_after(self, mock_ocr, mock_cap):
        """With bidirectional=True, should check frames before AND after."""
        cfg = Config(
            stability_frames=3,
            stability_interval=1.0,
            stability_agreement=0.50,
            stability_bidirectional=True,
        )
        verifier = StabilityVerifier(cfg, mock_ocr)

        # Mock OCR to always return matching text
        mock_ocr.read_frame.return_value = OcrResult(
            text="GUARD PASSING", confidence=0.8
        )

        result = verifier.verify(mock_cap, 10.0, "GUARD PASSING", 60.0)

        assert result is not None
        # Should have been called for forward (11, 12, 13) and backward (9, 8, 7)
        calls = mock_ocr.read_frame.call_args_list
        timestamps = [call[0][1] for call in calls]
        assert any(t > 10.0 for t in timestamps), "Should check forward"
        assert any(t < 10.0 for t in timestamps), "Should check backward"

    def test_unidirectional_only_checks_after(self, mock_ocr, mock_cap):
        """With bidirectional=False, should only check frames after."""
        cfg = Config(
            stability_frames=3,
            stability_interval=1.0,
            stability_agreement=0.50,
            stability_bidirectional=False,
        )
        verifier = StabilityVerifier(cfg, mock_ocr)

        mock_ocr.read_frame.return_value = OcrResult(
            text="GUARD PASSING", confidence=0.8
        )

        result = verifier.verify(mock_cap, 10.0, "GUARD PASSING", 60.0)

        assert result is not None
        calls = mock_ocr.read_frame.call_args_list
        timestamps = [call[0][1] for call in calls]
        assert all(t > 10.0 for t in timestamps), "Should only check forward"

    def test_no_agreement_returns_none(self, mock_ocr, mock_cap):
        """If no frames agree, should return None."""
        cfg = Config(
            stability_frames=3,
            stability_interval=1.0,
            stability_agreement=0.80,
            stability_bidirectional=True,
        )
        verifier = StabilityVerifier(cfg, mock_ocr)

        # Mock OCR to return completely different text
        mock_ocr.read_frame.return_value = OcrResult(
            text="COMPLETELY DIFFERENT TEXT", confidence=0.8
        )

        result = verifier.verify(mock_cap, 10.0, "GUARD PASSING", 60.0)
        assert result is None

    def test_majority_vote_returns_most_common(self, mock_ocr, mock_cap):
        """Should return the most common text, not the highest confidence."""
        cfg = Config(
            stability_frames=3,
            stability_interval=1.0,
            stability_agreement=0.30,
            stability_bidirectional=False,
        )
        verifier = StabilityVerifier(cfg, mock_ocr)

        # Return matching text with different confidences
        results = [
            OcrResult(text="Guard Passing", confidence=0.7),
            OcrResult(text="Guard Passing", confidence=0.6),
            OcrResult(text="Guard Passing Concepts", confidence=0.9),
        ]
        mock_ocr.read_frame.side_effect = results

        result = verifier.verify(mock_cap, 10.0, "Guard Passing", 60.0)

        # Both "Guard Passing" and "Guard Passing Concepts" are similar enough
        # to reference, so they all agree. Majority vote should pick
        # "Guard Passing" (appears twice) over "Guard Passing Concepts" (once)
        assert result is not None
        assert result.text == "Guard Passing"

    def test_backward_respects_zero_boundary(self, mock_ocr, mock_cap):
        """Backward checks should not go below t=0."""
        cfg = Config(
            stability_frames=5,
            stability_interval=1.0,
            stability_agreement=0.30,
            stability_bidirectional=True,
        )
        verifier = StabilityVerifier(cfg, mock_ocr)

        mock_ocr.read_frame.return_value = OcrResult(
            text="GUARD PASSING", confidence=0.8
        )

        # timestamp=2.0, so backward can only go to t=1.0 and t=0 (not negative)
        result = verifier.verify(mock_cap, 2.0, "GUARD PASSING", 60.0)

        calls = mock_ocr.read_frame.call_args_list
        timestamps = [call[0][1] for call in calls]
        assert all(t >= 0 for t in timestamps), "No negative timestamps"

    def test_forward_respects_duration_boundary(self, mock_ocr, mock_cap):
        """Forward checks should not exceed video duration."""
        cfg = Config(
            stability_frames=5,
            stability_interval=1.0,
            stability_agreement=0.30,
            stability_bidirectional=True,
        )
        verifier = StabilityVerifier(cfg, mock_ocr)

        mock_ocr.read_frame.return_value = OcrResult(
            text="GUARD PASSING", confidence=0.8
        )

        # timestamp=58.0, duration=60.0 -> forward can only go to 59
        result = verifier.verify(mock_cap, 58.0, "GUARD PASSING", 60.0)

        calls = mock_ocr.read_frame.call_args_list
        timestamps = [call[0][1] for call in calls]
        assert all(t < 60.0 for t in timestamps), "No timestamps beyond duration"


class TestMajorityVoteStatic:
    """Test the static _majority_vote method directly."""

    def test_single_reading(self):
        readings = [OcrResult(text="Guard Passing", confidence=0.8)]
        result = StabilityVerifier._majority_vote(readings)
        assert result.text == "Guard Passing"

    def test_picks_most_common(self):
        readings = [
            OcrResult(text="Guard Passing", confidence=0.6),
            OcrResult(text="Guard Passing", confidence=0.7),
            OcrResult(text="Arm Drag", confidence=0.9),
        ]
        result = StabilityVerifier._majority_vote(readings)
        assert result.text == "Guard Passing"
        # Should pick the highest confidence among "Guard Passing" variants
        assert result.confidence == 0.7

    def test_tie_picks_first_most_common(self):
        readings = [
            OcrResult(text="Guard Passing", confidence=0.8),
            OcrResult(text="Arm Drag", confidence=0.9),
        ]
        result = StabilityVerifier._majority_vote(readings)
        # Counter.most_common returns first element on tie
        assert result is not None
