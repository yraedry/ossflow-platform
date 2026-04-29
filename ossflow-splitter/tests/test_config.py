"""Tests for chapter_splitter.config.Config defaults and override behavior."""

from chapter_splitter.config import Config, DEFAULT_ROOT


class TestConfigDefaults:
    """Verify that Config has sensible defaults."""

    def test_default_root_dir(self):
        cfg = Config()
        assert cfg.root_dir == DEFAULT_ROOT

    def test_default_extensions(self):
        cfg = Config()
        assert ".mp4" in cfg.extensions
        assert ".mkv" in cfg.extensions
        assert ".avi" in cfg.extensions
        assert ".mov" in cfg.extensions

    def test_default_dry_run_false(self):
        cfg = Config()
        assert cfg.dry_run is False

    def test_default_verbose_false(self):
        cfg = Config()
        assert cfg.verbose is False

    def test_default_voice_threshold(self):
        cfg = Config()
        assert cfg.voice_threshold == 0.25

    def test_default_voice_hysteresis_thresholds(self):
        cfg = Config()
        assert cfg.voice_enter_threshold == 0.30
        assert cfg.voice_exit_threshold == 0.20

    def test_default_voice_window_and_smoothing(self):
        cfg = Config()
        assert cfg.voice_window_ms == 100
        assert cfg.voice_smoothing_ms == 500

    def test_default_ocr_confidence_min(self):
        cfg = Config()
        assert cfg.ocr_confidence_min == 0.55

    def test_default_ocr_min_text_length(self):
        cfg = Config()
        assert cfg.ocr_min_text_length == 5

    def test_default_stability_bidirectional(self):
        cfg = Config()
        assert cfg.stability_bidirectional is True

    def test_default_encoder(self):
        cfg = Config()
        assert cfg.encoder == "h264_nvenc"

    def test_default_output_format(self):
        cfg = Config()
        assert cfg.output_format == ".mkv"


class TestConfigOverride:
    """Verify that Config fields can be overridden at construction."""

    def test_override_root_dir(self):
        cfg = Config(root_dir="/custom/path")
        assert cfg.root_dir == "/custom/path"

    def test_override_dry_run(self):
        cfg = Config(dry_run=True)
        assert cfg.dry_run is True

    def test_override_voice_threshold(self):
        cfg = Config(voice_threshold=0.5)
        assert cfg.voice_threshold == 0.5

    def test_override_hysteresis_thresholds(self):
        cfg = Config(voice_enter_threshold=0.35, voice_exit_threshold=0.15)
        assert cfg.voice_enter_threshold == 0.35
        assert cfg.voice_exit_threshold == 0.15

    def test_override_voice_window(self):
        cfg = Config(voice_window_ms=200, voice_smoothing_ms=1000)
        assert cfg.voice_window_ms == 200
        assert cfg.voice_smoothing_ms == 1000

    def test_override_ocr_min_text_length(self):
        cfg = Config(ocr_min_text_length=3)
        assert cfg.ocr_min_text_length == 3

    def test_override_stability_bidirectional(self):
        cfg = Config(stability_bidirectional=False)
        assert cfg.stability_bidirectional is False

    def test_override_roi_fractions(self):
        cfg = Config(roi_top_fraction=0.25, roi_bottom_fraction=0.30)
        assert cfg.roi_top_fraction == 0.25
        assert cfg.roi_bottom_fraction == 0.30

    def test_override_encoder_and_preset(self):
        cfg = Config(encoder="libx264", preset="fast", cq=18)
        assert cfg.encoder == "libx264"
        assert cfg.preset == "fast"
        assert cfg.cq == 18

    def test_override_does_not_affect_other_fields(self):
        cfg = Config(dry_run=True)
        assert cfg.verbose is False
        assert cfg.voice_threshold == 0.25
