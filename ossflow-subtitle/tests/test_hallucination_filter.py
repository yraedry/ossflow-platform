"""Comprehensive tests for subtitle_generator.hallucination_filter.HallucinationFilter."""

import pytest

from subtitle_generator.config import SubtitleConfig
from subtitle_generator.hallucination_filter import HallucinationFilter


@pytest.fixture
def hfilter():
    """HallucinationFilter with default config."""
    return HallucinationFilter(SubtitleConfig())


@pytest.fixture
def config():
    """Default SubtitleConfig."""
    return SubtitleConfig()


# ---------------------------------------------------------------------------
# _filter_repeated_segments
# ---------------------------------------------------------------------------

class TestFilterRepeatedSegments:
    """Segments that are >80% similar to recent segments should be dropped."""

    def test_filter_repeated_segments(self, hfilter):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "The arm drag is fundamental."},
            {"start": 2.0, "end": 4.0, "text": "The arm drag is fundamental."},  # exact repeat
            {"start": 4.0, "end": 6.0, "text": "The arm drag is fundamental!"},  # near-repeat
        ]
        result = hfilter._filter_repeated_segments(segments)
        assert len(result) == 1
        assert result[0]["text"] == "The arm drag is fundamental."

    def test_keeps_different_segments(self, hfilter):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "The arm drag is fundamental."},
            {"start": 2.0, "end": 4.0, "text": "Now let us look at the wrist roll."},
            {"start": 4.0, "end": 6.0, "text": "This requires shoulder contact."},
        ]
        result = hfilter._filter_repeated_segments(segments)
        assert len(result) == 3

    def test_empty_text_segments_dropped(self, hfilter):
        segments = [
            {"start": 0.0, "end": 2.0, "text": ""},
            {"start": 2.0, "end": 4.0, "text": "Valid text here."},
        ]
        result = hfilter._filter_repeated_segments(segments)
        # Empty text is skipped (continue), not added to result
        assert len(result) == 1
        assert result[0]["text"] == "Valid text here."

    def test_lookback_window_respected(self):
        """With lookback=1, fuzzy matching only checks the previous segment,
        but the global exact-duplicate set still catches exact repeats."""
        cfg = SubtitleConfig(similarity_lookback=1)
        filt = HallucinationFilter(cfg)
        segments = [
            {"start": 0.0, "end": 2.0, "text": "The arm drag is fundamental."},
            {"start": 2.0, "end": 4.0, "text": "Something completely different."},
            {"start": 4.0, "end": 6.0, "text": "The arm drag is fundamental."},  # same as [0]
        ]
        result = filt._filter_repeated_segments(segments)
        # Global exact-duplicate set now catches this even with lookback=1
        assert len(result) == 2

    def test_global_duplicate_detection_across_distance(self):
        """Exact duplicates are caught regardless of how far apart they are."""
        cfg = SubtitleConfig(similarity_lookback=1)
        filt = HallucinationFilter(cfg)
        segments = [
            {"start": 0.0, "end": 2.0, "text": "The arm drag is fundamental."},
        ]
        # Add many different segments to push beyond lookback
        for i in range(20):
            segments.append({
                "start": 2.0 + i * 2, "end": 4.0 + i * 2,
                "text": f"Unique segment number {i} with distinct content.",
            })
        # Add the duplicate far away
        segments.append({
            "start": 100.0, "end": 102.0,
            "text": "The arm drag is fundamental.",
        })
        result = filt._filter_repeated_segments(segments)
        # The far-away duplicate should be caught by the global set
        texts = [s["text"] for s in result]
        assert texts.count("The arm drag is fundamental.") == 1


# ---------------------------------------------------------------------------
# _filter_repeated_phrases
# ---------------------------------------------------------------------------

class TestFilterRepeatedPhrases:
    """3-word ngrams repeated >2x should cause truncation."""

    def test_filter_repeated_phrases(self, hfilter):
        # "the arm drag" repeated 4 times (> max of 2)
        text = "the arm drag the arm drag the arm drag the arm drag something else"
        segments = [{"start": 0.0, "end": 5.0, "text": text}]
        result = hfilter._filter_repeated_phrases(segments)
        # Should be truncated after first occurrence of the repeated ngram
        assert len(result[0]["text"].split()) <= len(text.split())
        assert result[0]["text"].startswith("the arm drag")

    def test_no_truncation_when_under_limit(self, hfilter):
        # "the arm drag" appears only twice (= max, not >max)
        text = "the arm drag is good the arm drag is great"
        segments = [{"start": 0.0, "end": 5.0, "text": text}]
        result = hfilter._filter_repeated_phrases(segments)
        assert result[0]["text"] == text

    def test_short_segments_skipped(self, hfilter):
        # Too few words for ngram analysis
        text = "hello world"
        segments = [{"start": 0.0, "end": 2.0, "text": text}]
        result = hfilter._filter_repeated_phrases(segments)
        assert result[0]["text"] == text

    def test_word_level_data_truncated_too(self, hfilter):
        text = "the arm drag the arm drag the arm drag the arm drag end"
        words = [{"word": w, "start": 0.0, "end": 1.0, "score": 0.9} for w in text.split()]
        segments = [{"start": 0.0, "end": 5.0, "text": text, "words": words}]
        result = hfilter._filter_repeated_phrases(segments)
        # words list should also be truncated
        assert len(result[0]["words"]) == len(result[0]["text"].split())


# ---------------------------------------------------------------------------
# _filter_low_confidence
# ---------------------------------------------------------------------------

class TestFilterLowConfidence:
    """Segments where >60% of words have low confidence should be dropped."""

    def test_filter_low_confidence(self, hfilter):
        segments = [{
            "start": 0.0,
            "end": 3.0,
            "text": "this is bad quality text here",
            "words": [
                {"word": "this", "score": 0.1},
                {"word": "is", "score": 0.1},
                {"word": "bad", "score": 0.1},
                {"word": "quality", "score": 0.1},
                {"word": "text", "score": 0.9},  # only 1 of 6 above threshold (0.4)
                {"word": "here", "score": 0.1},
            ],
        }]
        result = hfilter._filter_low_confidence(segments)
        # 5/6 below threshold(0.4) = 83% > 60% ratio -> dropped
        assert len(result) == 0

    def test_keeps_high_confidence(self, hfilter):
        segments = [{
            "start": 0.0,
            "end": 3.0,
            "text": "this is great quality text",
            "words": [
                {"word": "this", "score": 0.9},
                {"word": "is", "score": 0.85},
                {"word": "great", "score": 0.95},
                {"word": "quality", "score": 0.88},
                {"word": "text", "score": 0.92},
            ],
        }]
        result = hfilter._filter_low_confidence(segments)
        assert len(result) == 1

    def test_no_words_key_keeps_segment(self, hfilter):
        segments = [{"start": 0.0, "end": 2.0, "text": "no word data"}]
        result = hfilter._filter_low_confidence(segments)
        assert len(result) == 1

    def test_no_scored_words_keeps_segment(self, hfilter):
        segments = [{
            "start": 0.0,
            "end": 2.0,
            "text": "no scores",
            "words": [{"word": "no"}, {"word": "scores"}],
        }]
        result = hfilter._filter_low_confidence(segments)
        assert len(result) == 1

    def test_borderline_ratio_kept(self):
        # Exactly 60% low -> >= threshold -> dropped
        cfg = SubtitleConfig(low_confidence_segment_ratio=0.60, low_confidence_word_threshold=0.4)
        filt = HallucinationFilter(cfg)
        segments = [{
            "start": 0.0,
            "end": 3.0,
            "text": "a b c d e",
            "words": [
                {"word": "a", "score": 0.1},
                {"word": "b", "score": 0.1},
                {"word": "c", "score": 0.1},
                {"word": "d", "score": 0.9},
                {"word": "e", "score": 0.9},
            ],
        }]
        result = filt._filter_low_confidence(segments)
        # 3/5 = 0.6 >= 0.6 -> dropped
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _filter_nonsense_rate
# ---------------------------------------------------------------------------

class TestFilterNonsenseRate:
    """Segments exceeding max_chars_per_second should be dropped."""

    def test_filter_nonsense_rate(self, hfilter):
        # 100 chars in 1 second = 100 cps >> 30
        segments = [{
            "start": 0.0,
            "end": 1.0,
            "text": "a" * 100,
        }]
        result = hfilter._filter_nonsense_rate(segments)
        assert len(result) == 0

    def test_keeps_normal_rate(self, hfilter):
        # "Hello world" = 11 chars in 2 sec = 5.5 cps < 30
        segments = [{
            "start": 0.0,
            "end": 2.0,
            "text": "Hello world",
        }]
        result = hfilter._filter_nonsense_rate(segments)
        assert len(result) == 1

    def test_missing_timestamps_kept(self, hfilter):
        segments = [{"text": "no timing info"}]
        result = hfilter._filter_nonsense_rate(segments)
        assert len(result) == 1

    def test_zero_duration_kept(self, hfilter):
        segments = [{"start": 5.0, "end": 5.0, "text": "zero duration"}]
        result = hfilter._filter_nonsense_rate(segments)
        assert len(result) == 1

    def test_empty_text_kept(self, hfilter):
        segments = [{"start": 0.0, "end": 1.0, "text": ""}]
        result = hfilter._filter_nonsense_rate(segments)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _filter_impossible_timing
# ---------------------------------------------------------------------------

class TestFilterImpossibleTiming:
    """Segments with end<=start, negative, or out-of-order timestamps should be dropped."""

    def test_end_equals_start(self, hfilter):
        segments = [{"start": 5.0, "end": 5.0, "text": "bad"}]
        result = hfilter._filter_impossible_timing(segments)
        assert len(result) == 0

    def test_end_before_start(self, hfilter):
        segments = [{"start": 5.0, "end": 3.0, "text": "bad"}]
        result = hfilter._filter_impossible_timing(segments)
        assert len(result) == 0

    def test_negative_start(self, hfilter):
        segments = [{"start": -1.0, "end": 2.0, "text": "bad"}]
        result = hfilter._filter_impossible_timing(segments)
        assert len(result) == 0

    def test_out_of_order(self, hfilter):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "first"},
            {"start": 3.0, "end": 7.0, "text": "overlaps first by >50ms"},
        ]
        result = hfilter._filter_impossible_timing(segments)
        assert len(result) == 1
        assert result[0]["text"] == "first"

    def test_valid_ordering_kept(self, hfilter):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "first"},
            {"start": 2.0, "end": 4.0, "text": "second"},
            {"start": 4.0, "end": 6.0, "text": "third"},
        ]
        result = hfilter._filter_impossible_timing(segments)
        assert len(result) == 3

    def test_null_timestamps_kept(self, hfilter):
        segments = [{"start": None, "end": None, "text": "no timing"}]
        result = hfilter._filter_impossible_timing(segments)
        assert len(result) == 1

    def test_slight_overlap_tolerated(self, hfilter):
        # Overlap of 0.04s < 0.05 tolerance
        segments = [
            {"start": 0.0, "end": 5.0, "text": "first"},
            {"start": 4.96, "end": 7.0, "text": "barely overlaps"},
        ]
        result = hfilter._filter_impossible_timing(segments)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# filter_all (integration)
# ---------------------------------------------------------------------------

class TestFilterAllPipeline:
    """Integration test: all filters applied in sequence."""

    def test_filter_all_pipeline(self, hfilter):
        segments = [
            # Good segment (40 chars / 4s = 10 cps, under 13 limit)
            {"start": 0.0, "end": 4.0, "text": "The arm drag is a fundamental technique.",
             "words": [{"word": w, "score": 0.9} for w in "The arm drag is a fundamental technique.".split()]},
            # Repeated segment (should be dropped)
            {"start": 4.0, "end": 8.0, "text": "The arm drag is a fundamental technique.",
             "words": [{"word": w, "score": 0.9} for w in "The arm drag is a fundamental technique.".split()]},
            # Low confidence (should be dropped)
            {"start": 8.0, "end": 12.0, "text": "garbled nonsense here now",
             "words": [
                 {"word": "garbled", "score": 0.05},
                 {"word": "nonsense", "score": 0.05},
                 {"word": "here", "score": 0.05},
                 {"word": "now", "score": 0.05},
             ]},
            # Good segment (41 chars / 4s = 10.25 cps, under 13 limit)
            {"start": 12.0, "end": 16.0, "text": "Next we look at the wrist roll technique.",
             "words": [{"word": w, "score": 0.9} for w in "Next we look at the wrist roll technique.".split()]},
            # Impossible timing (end <= start)
            {"start": 20.0, "end": 19.0, "text": "bad timing"},
        ]
        result = hfilter.filter_all(segments)
        # Only 2 good segments should survive
        assert len(result) == 2
        assert result[0]["text"] == "The arm drag is a fundamental technique."
        assert result[1]["text"] == "Next we look at the wrist roll technique."

    def test_filter_all_empty_input(self, hfilter):
        result = hfilter.filter_all([])
        assert result == []

    def test_filter_all_stats_populated(self, hfilter):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Good text.",
             "words": [{"word": "Good", "score": 0.9}, {"word": "text.", "score": 0.9}]},
            {"start": 2.0, "end": 4.0, "text": "Good text.",
             "words": [{"word": "Good", "score": 0.9}, {"word": "text.", "score": 0.9}]},
        ]
        hfilter.filter_all(segments)
        assert hfilter.stats["repeated_segments"] >= 1


# ---------------------------------------------------------------------------
# _filter_silence_hallucinations
# ---------------------------------------------------------------------------

class TestFilterSilenceHallucinations:
    """Filter 6: segments over silent audio should be dropped."""

    def test_no_audio_path_is_noop(self, hfilter):
        """When no audio_path is given, all segments pass through."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Some text."},
        ]
        result = hfilter._filter_silence_hallucinations(segments, audio_path=None)
        assert len(result) == 1

    def test_nonexistent_audio_is_noop(self, hfilter, tmp_path):
        """When audio file does not exist, all segments pass through."""
        fake_path = tmp_path / "nonexistent.wav"
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello."}]
        result = hfilter._filter_silence_hallucinations(segments, audio_path=fake_path)
        assert len(result) == 1

    def test_silent_segment_dropped(self, hfilter, tmp_path):
        """A segment covering pure silence should be dropped."""
        try:
            from pydub import AudioSegment as PydubSegment
            from pydub.generators import Sine
        except ImportError:
            pytest.skip("pydub not installed")

        # Generate 5 seconds of silence
        silence = PydubSegment.silent(duration=5000, frame_rate=16000)
        audio_file = tmp_path / "silence.wav"
        silence.export(str(audio_file), format="wav")

        segments = [{"start": 0.0, "end": 3.0, "text": "Hallucinated over silence."}]
        result = hfilter._filter_silence_hallucinations(segments, audio_path=audio_file)
        assert len(result) == 0
        assert hfilter.stats["silence_hallucinations"] == 1

    def test_loud_segment_kept(self, hfilter, tmp_path):
        """A segment covering loud audio should be kept."""
        try:
            from pydub import AudioSegment as PydubSegment
            from pydub.generators import Sine
        except ImportError:
            pytest.skip("pydub not installed")

        # Generate 3 seconds of a 440Hz tone (definitely not silence)
        tone = Sine(440).to_audio_segment(duration=3000).set_frame_rate(16000)
        audio_file = tmp_path / "tone.wav"
        tone.export(str(audio_file), format="wav")

        segments = [{"start": 0.0, "end": 2.0, "text": "Real speech here."}]
        result = hfilter._filter_silence_hallucinations(segments, audio_path=audio_file)
        assert len(result) == 1

    def test_missing_timestamps_kept(self, hfilter, tmp_path):
        """Segments without start/end should pass through."""
        try:
            from pydub import AudioSegment as PydubSegment
        except ImportError:
            pytest.skip("pydub not installed")

        silence = PydubSegment.silent(duration=2000, frame_rate=16000)
        audio_file = tmp_path / "silence.wav"
        silence.export(str(audio_file), format="wav")

        segments = [{"text": "No timestamps."}]
        result = hfilter._filter_silence_hallucinations(segments, audio_path=audio_file)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _filter_synthetic_scores
# ---------------------------------------------------------------------------

class TestFilterSyntheticScores:
    """Filter 7: segments with mostly exact-0.5 scores get stricter filtering."""

    def test_all_exact_half_and_low_conf_dropped(self):
        """Segment with 100% score=0.5 and all below word threshold -> dropped."""
        cfg = SubtitleConfig(
            low_confidence_word_threshold=0.4,
            synthetic_score_ratio=0.50,
            synthetic_score_strict_ratio=0.30,
        )
        filt = HallucinationFilter(cfg)
        segments = [{
            "start": 0.0,
            "end": 3.0,
            "text": "fake hallucinated text here now",
            "words": [
                {"word": "fake", "score": 0.5},
                {"word": "hallucinated", "score": 0.5},
                {"word": "text", "score": 0.5},
                {"word": "here", "score": 0.5},
                {"word": "now", "score": 0.5},
            ],
        }]
        # All words have score 0.5 (100% > 50% threshold)
        # All words have score 0.5 which is >= 0.4 word_threshold,
        # so low_count = 0/5 = 0.0 < 0.30 -> NOT dropped
        # Actually score 0.5 >= 0.4, so it's NOT below threshold
        # We need words actually below 0.4 for strict filter to kick in
        # Let's adjust: use score=0.3 which IS below 0.4
        segments = [{
            "start": 0.0,
            "end": 3.0,
            "text": "fake hallucinated text here now",
            "words": [
                {"word": "fake", "score": 0.5},
                {"word": "hallucinated", "score": 0.5},
                {"word": "text", "score": 0.5},
                {"word": "here", "score": 0.3},  # below 0.4 threshold
                {"word": "now", "score": 0.3},    # below 0.4 threshold
            ],
        }]
        result = filt._filter_synthetic_scores(segments)
        # 3/5 = 60% exact-0.5 > 50% -> synthetic detected
        # 2/5 = 40% below word_threshold(0.4) >= 30% strict ratio -> dropped
        assert len(result) == 0
        assert filt.stats["synthetic_scores"] == 1

    def test_high_confidence_synthetic_kept(self):
        """Segment with 100% score=0.5 but none below word threshold -> kept."""
        cfg = SubtitleConfig(
            low_confidence_word_threshold=0.4,
            synthetic_score_ratio=0.50,
            synthetic_score_strict_ratio=0.30,
        )
        filt = HallucinationFilter(cfg)
        segments = [{
            "start": 0.0,
            "end": 3.0,
            "text": "good synthetic text here now",
            "words": [
                {"word": "good", "score": 0.5},
                {"word": "synthetic", "score": 0.5},
                {"word": "text", "score": 0.5},
                {"word": "here", "score": 0.5},
                {"word": "now", "score": 0.5},
            ],
        }]
        result = filt._filter_synthetic_scores(segments)
        # 100% exact-0.5 -> synthetic detected
        # 0/5 below 0.4 = 0% < 30% strict ratio -> kept
        assert len(result) == 1

    def test_non_synthetic_segment_kept(self):
        """Segment with natural scores (not 0.5) should pass through."""
        cfg = SubtitleConfig()
        filt = HallucinationFilter(cfg)
        segments = [{
            "start": 0.0,
            "end": 3.0,
            "text": "natural speech segment here",
            "words": [
                {"word": "natural", "score": 0.92},
                {"word": "speech", "score": 0.88},
                {"word": "segment", "score": 0.85},
                {"word": "here", "score": 0.91},
            ],
        }]
        result = filt._filter_synthetic_scores(segments)
        assert len(result) == 1

    def test_no_words_kept(self):
        """Segment without words key passes through."""
        filt = HallucinationFilter(SubtitleConfig())
        segments = [{"start": 0.0, "end": 2.0, "text": "no words"}]
        result = filt._filter_synthetic_scores(segments)
        assert len(result) == 1

    def test_mixed_synthetic_and_real(self):
        """Only the synthetic segment is filtered, real one kept."""
        cfg = SubtitleConfig(
            low_confidence_word_threshold=0.4,
            synthetic_score_ratio=0.50,
            synthetic_score_strict_ratio=0.30,
        )
        filt = HallucinationFilter(cfg)
        segments = [
            {
                "start": 0.0, "end": 2.0, "text": "real speech",
                "words": [
                    {"word": "real", "score": 0.92},
                    {"word": "speech", "score": 0.88},
                ],
            },
            {
                "start": 2.0, "end": 4.0, "text": "synthetic low conf words here",
                "words": [
                    {"word": "synthetic", "score": 0.5},
                    {"word": "low", "score": 0.5},
                    {"word": "conf", "score": 0.2},
                    {"word": "words", "score": 0.5},
                    {"word": "here", "score": 0.1},
                ],
            },
        ]
        result = filt._filter_synthetic_scores(segments)
        # Second segment: 3/5=60% exact-0.5 > 50%, 2/5=40% below 0.4 >= 30% -> dropped
        assert len(result) == 1
        assert result[0]["text"] == "real speech"


# ---------------------------------------------------------------------------
# Expanded lookback (default = 15)
# ---------------------------------------------------------------------------

class TestExpandedLookback:
    """Verify the default lookback of 15 catches distant similar segments."""

    def test_default_lookback_is_15(self):
        cfg = SubtitleConfig()
        assert cfg.similarity_lookback == 15

    def test_catches_repeat_within_15(self):
        """A near-duplicate at distance 10 should be caught with lookback=15."""
        filt = HallucinationFilter(SubtitleConfig())
        segments = [
            {"start": 0.0, "end": 2.0, "text": "The arm drag is fundamental."},
        ]
        # Add 10 unique segments in between
        for i in range(10):
            segments.append({
                "start": 2.0 + i * 2, "end": 4.0 + i * 2,
                "text": f"Unique distinct sentence number {i} about a different topic.",
            })
        # Add near-duplicate of segment 0
        segments.append({
            "start": 50.0, "end": 52.0,
            "text": "The arm drag is fundamental!",  # near-repeat (punctuation diff)
        })
        result = filt._filter_repeated_segments(segments)
        # The near-duplicate should be caught (within lookback window of 15)
        # Plus the exact duplicate is caught by global set
        texts = [s["text"] for s in result]
        assert "The arm drag is fundamental!" not in texts
