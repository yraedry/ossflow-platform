"""Tests for subtitle_generator.config dataclass defaults."""

from subtitle_generator.config import (
    SubtitleConfig,
    TranscriptionConfig,
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_INITIAL_PROMPT_GENERIC,
    DEFAULT_INITIAL_PROMPT_TEMPLATE,
    DEFAULT_ROOT_DIR,
    EXTENSIONES,
    generate_prompt,
)


class TestTranscriptionConfigDefaults:
    """Verify TranscriptionConfig default values."""

    def test_model_name(self):
        cfg = TranscriptionConfig()
        assert cfg.model_name == "large-v3"

    def test_compute_type(self):
        cfg = TranscriptionConfig()
        assert cfg.compute_type == "float16"

    def test_device(self):
        cfg = TranscriptionConfig()
        assert cfg.device == "cuda"

    def test_language(self):
        cfg = TranscriptionConfig()
        assert cfg.language == "en"

    def test_batch_size(self):
        cfg = TranscriptionConfig()
        assert cfg.batch_size == 4

    def test_beam_size(self):
        cfg = TranscriptionConfig()
        assert cfg.beam_size == 5

    def test_initial_prompt(self):
        cfg = TranscriptionConfig()
        assert cfg.initial_prompt == DEFAULT_INITIAL_PROMPT
        assert "Brazilian Jiu-Jitsu" in cfg.initial_prompt

    def test_vad_onset(self):
        cfg = TranscriptionConfig()
        assert cfg.vad_onset == 0.350

    def test_vad_offset(self):
        cfg = TranscriptionConfig()
        assert cfg.vad_offset == 0.250

    def test_condition_on_previous_text(self):
        cfg = TranscriptionConfig()
        assert cfg.condition_on_previous_text is False

    def test_no_speech_threshold(self):
        cfg = TranscriptionConfig()
        assert cfg.no_speech_threshold == 0.7

    def test_initial_prompt_template(self):
        cfg = TranscriptionConfig()
        assert "{instructor}" in cfg.initial_prompt_template
        assert "{topic}" in cfg.initial_prompt_template


class TestSubtitleConfigDefaults:
    """Verify SubtitleConfig default values."""

    def test_max_chars_per_line(self):
        cfg = SubtitleConfig()
        assert cfg.max_chars_per_line == 42

    def test_max_lines(self):
        cfg = SubtitleConfig()
        assert cfg.max_lines == 2

    def test_min_duration(self):
        cfg = SubtitleConfig()
        assert cfg.min_duration == 0.5

    def test_max_duration(self):
        cfg = SubtitleConfig()
        assert cfg.max_duration == 7.0

    def test_gap_fill_threshold(self):
        cfg = SubtitleConfig()
        assert cfg.gap_fill_threshold == 0.100

    def test_gap_warn_threshold(self):
        cfg = SubtitleConfig()
        assert cfg.gap_warn_threshold == 5.0

    def test_similarity_threshold(self):
        cfg = SubtitleConfig()
        assert cfg.similarity_threshold == 0.80

    def test_similarity_lookback(self):
        cfg = SubtitleConfig()
        assert cfg.similarity_lookback == 15

    def test_low_confidence_word_threshold(self):
        cfg = SubtitleConfig()
        assert cfg.low_confidence_word_threshold == 0.4

    def test_max_chars_per_second(self):
        cfg = SubtitleConfig()
        assert cfg.max_chars_per_second == 30.0

    def test_silence_threshold_db(self):
        cfg = SubtitleConfig()
        assert cfg.silence_threshold_db == -50.0

    def test_synthetic_score_ratio(self):
        cfg = SubtitleConfig()
        assert cfg.synthetic_score_ratio == 0.50

    def test_synthetic_score_strict_ratio(self):
        cfg = SubtitleConfig()
        assert cfg.synthetic_score_strict_ratio == 0.60


class TestGeneratePrompt:
    """Tests for the dynamic prompt generation function."""

    def test_no_args_returns_generic(self):
        result = generate_prompt()
        assert result == DEFAULT_INITIAL_PROMPT_GENERIC
        assert "Brazilian Jiu-Jitsu" in result

    def test_instructor_only(self):
        result = generate_prompt(instructor="John Danaher")
        assert "John Danaher" in result
        assert "Brazilian Jiu-Jitsu" in result

    def test_topic_only(self):
        result = generate_prompt(topic="Arm Drags")
        assert "Arm Drags" in result

    def test_both_args(self):
        result = generate_prompt(instructor="John Danaher", topic="Arm Drags")
        assert "John Danaher" in result
        assert "Arm Drags" in result

    def test_custom_template(self):
        tpl = "Instructor: {instructor}, Subject: {topic}"
        result = generate_prompt(instructor="Gordon Ryan", topic="Pins", template=tpl)
        assert result == "Instructor: Gordon Ryan, Subject: Pins"

    def test_none_instructor_uses_placeholder(self):
        result = generate_prompt(topic="Escapes")
        assert "the instructor" in result
        assert "Escapes" in result

    def test_none_topic_uses_placeholder(self):
        result = generate_prompt(instructor="Marcelo Garcia")
        assert "Marcelo Garcia" in result
        assert "Brazilian Jiu-Jitsu techniques" in result

    def test_default_prompt_alias(self):
        """DEFAULT_INITIAL_PROMPT should be the generic prompt for backward compat."""
        assert DEFAULT_INITIAL_PROMPT == DEFAULT_INITIAL_PROMPT_GENERIC


class TestModuleConstants:
    """Verify module-level constants."""

    def test_extensiones(self):
        assert ".mp4" in EXTENSIONES
        assert ".mkv" in EXTENSIONES

    def test_default_root_dir_is_string(self):
        assert isinstance(DEFAULT_ROOT_DIR, str)
