"""Tests for api.paths — host↔container translation."""

import pytest

from api.paths import from_container_path, to_container_path


class TestToContainer:
    def test_unc_with_unc_library(self):
        lib = r"\\10.10.100.6\multimedia\instruccionales"
        host = r"\\10.10.100.6\multimedia\instruccionales\Danaher\Season 01\vid.mkv"
        assert (
            to_container_path(host, lib)
            == "/media/Danaher/Season 01/vid.mkv"
        )

    def test_drive_letter(self):
        lib = r"Z:\instruccionales"
        host = r"Z:\instruccionales\Gordon\ep1.mkv"
        assert to_container_path(host, lib) == "/media/Gordon/ep1.mkv"

    def test_already_container_media(self):
        assert (
            to_container_path("/media/x/y.mkv", r"Z:\instruccionales")
            == "/media/x/y.mkv"
        )

    def test_already_container_library_legacy(self):
        # Legacy /library paths must pass through unchanged for backward compat.
        assert (
            to_container_path("/library/x/y.mkv", r"Z:\instruccionales")
            == "/library/x/y.mkv"
        )

    def test_idempotent(self):
        lib = r"\\srv\share\lib"
        host = r"\\srv\share\lib\a\b.mkv"
        once = to_container_path(host, lib)
        twice = to_container_path(once, lib)
        assert once == twice == "/media/a/b.mkv"

    def test_case_insensitive_windows(self):
        lib = r"Z:\Instruccionales"
        host = r"z:\INSTRUCCIONALES\X\y.mkv"
        assert to_container_path(host, lib) == "/media/X/y.mkv"

    def test_outside_library_raises(self):
        with pytest.raises(ValueError):
            to_container_path(r"C:\other\foo.mkv", r"Z:\instruccionales")

    def test_empty_host(self):
        with pytest.raises(ValueError):
            to_container_path("", r"Z:\lib")

    def test_empty_library(self):
        with pytest.raises(ValueError):
            to_container_path(r"Z:\lib\x.mkv", "")

    def test_custom_container_root(self):
        lib = r"Z:\lib"
        assert (
            to_container_path(r"Z:\lib\a.mkv", lib, container_root="/data")
            == "/data/a.mkv"
        )


class TestFromContainer:
    def test_roundtrip_unc(self):
        lib = r"\\srv\share\lib"
        host = r"\\srv\share\lib\a\b.mkv"
        cp = to_container_path(host, lib)
        back = from_container_path(cp, lib)
        assert back.lower().replace("\\", "/") == host.lower().replace("\\", "/")

    def test_root_only(self):
        assert (
            from_container_path("/media", r"Z:\lib").replace("\\", "/").lower()
            == "z:/lib"
        )

    def test_non_container_passthrough(self):
        assert from_container_path(r"Z:\lib\x.mkv", r"Z:\lib") == r"Z:\lib\x.mkv"
