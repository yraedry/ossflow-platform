"""Tests for chapter_splitter.splitting.splitter.VideoSplitter._build_cmd."""

import unittest.mock as mock

import pytest

from chapter_splitter.config import Config
from chapter_splitter.models import Chapter

# Mock subprocess since splitter imports it at module level
with mock.patch.dict("sys.modules", {"subprocess": mock.MagicMock()}):
    from chapter_splitter.splitting.splitter import VideoSplitter


@pytest.fixture
def splitter():
    """VideoSplitter with default config."""
    return VideoSplitter(Config())


@pytest.fixture
def splitter_custom():
    """VideoSplitter with custom encoder settings."""
    return VideoSplitter(Config(encoder="libx264", preset="fast", cq=18, audio_codec="copy"))


class TestBuildCmd:
    """Test _build_cmd produces correct ffmpeg arguments."""

    def test_starts_with_ffmpeg(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, 120.0)
        assert cmd[0] == "ffmpeg"

    def test_contains_overwrite_flag(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, 120.0)
        assert "-y" in cmd

    def test_contains_hide_banner(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, 120.0)
        assert "-hide_banner" in cmd

    def test_contains_input_file(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, 120.0)
        i_idx = cmd.index("-i")
        assert cmd[i_idx + 1] == "input.mkv"

    def test_contains_output_file(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, 120.0)
        assert cmd[-1] == "output.mkv"

    def test_two_pass_seek_before_input(self, splitter):
        # start=60.0 -> fast_seek = max(0, 60-30) = 30.0
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, 120.0)
        # First -ss should appear before -i (fast seek)
        i_idx = cmd.index("-i")
        first_ss_idx = cmd.index("-ss")
        assert first_ss_idx < i_idx
        assert cmd[first_ss_idx + 1] == "30.000"

    def test_two_pass_seek_after_input(self, splitter):
        # start=60.0, fast_seek=30.0, precise_offset=30.0
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, 120.0)
        i_idx = cmd.index("-i")
        # Find second -ss after -i
        second_ss_idx = None
        for j in range(i_idx + 1, len(cmd)):
            if cmd[j] == "-ss":
                second_ss_idx = j
                break
        assert second_ss_idx is not None
        assert cmd[second_ss_idx + 1] == "30.000"

    def test_fast_seek_clamps_to_zero(self, splitter):
        # start=10.0 -> fast_seek = max(0, 10-30) = 0.0
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 10.0, 20.0)
        first_ss_idx = cmd.index("-ss")
        assert cmd[first_ss_idx + 1] == "0.000"

    def test_duration_flag_with_end_time(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, 120.0)
        t_idx = cmd.index("-t")
        # duration = 120 - 60 = 60
        assert cmd[t_idx + 1] == "60.000"

    def test_no_duration_flag_without_end_time(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 60.0, None)
        assert "-t" not in cmd

    def test_uses_configured_encoder(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 0.0, 10.0)
        cv_idx = cmd.index("-c:v")
        assert cmd[cv_idx + 1] == "h264_nvenc"

    def test_uses_configured_preset(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 0.0, 10.0)
        p_idx = cmd.index("-preset")
        assert cmd[p_idx + 1] == "p4"

    def test_uses_configured_cq(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 0.0, 10.0)
        cq_idx = cmd.index("-cq")
        assert cmd[cq_idx + 1] == "23"

    def test_uses_configured_audio_codec(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 0.0, 10.0)
        ca_idx = cmd.index("-c:a")
        assert cmd[ca_idx + 1] == "aac"

    def test_custom_encoder_settings(self, splitter_custom):
        cmd = splitter_custom._build_cmd("in.mkv", "out.mkv", 0.0, 5.0)
        cv_idx = cmd.index("-c:v")
        assert cmd[cv_idx + 1] == "libx264"
        p_idx = cmd.index("-preset")
        assert cmd[p_idx + 1] == "fast"
        cq_idx = cmd.index("-cq")
        assert cmd[cq_idx + 1] == "18"
        ca_idx = cmd.index("-c:a")
        assert cmd[ca_idx + 1] == "copy"

    def test_start_zero_no_negative_seek(self, splitter):
        cmd = splitter._build_cmd("input.mkv", "output.mkv", 0.0, 10.0)
        first_ss_idx = cmd.index("-ss")
        assert float(cmd[first_ss_idx + 1]) >= 0.0
