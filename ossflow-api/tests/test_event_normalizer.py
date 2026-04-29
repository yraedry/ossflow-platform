"""Tests for api.event_normalizer.normalize."""

from __future__ import annotations

from api.event_normalizer import NormalizedEvent, is_terminal, normalize


def test_normalize_done_bjj_kit_shape():
    evt = normalize({"type": "done", "data": {"result": {"ok": True}}})
    assert evt.kind == "done"
    assert evt.status == "completed"
    assert evt.progress == 100.0
    assert evt.payload == {"result": {"ok": True}}
    assert is_terminal(evt)


def test_normalize_error_bjj_kit_shape():
    evt = normalize({"type": "error", "data": {"message": "boom"}})
    assert evt.kind == "error"
    assert evt.status == "failed"
    assert evt.message == "boom"
    assert is_terminal(evt)


def test_normalize_error_fallback_to_error_field():
    evt = normalize({"type": "error", "data": {"error": "bad thing"}})
    assert evt.kind == "error"
    assert evt.message == "bad thing"


def test_normalize_progress_percent_field():
    evt = normalize({"type": "progress", "data": {"percent": 0.42, "message": "half"}})
    assert evt.kind == "progress"
    assert evt.progress == 42.0
    assert evt.message == "half"
    assert not is_terminal(evt)


def test_normalize_progress_accepts_0_100_scale():
    evt = normalize({"type": "progress", "data": {"progress": 73}})
    assert evt.kind == "progress"
    assert evt.progress == 73.0


def test_normalize_log_line_field():
    evt = normalize({"type": "log", "data": {"line": "processing frame"}})
    assert evt.kind == "log"
    assert evt.message == "processing frame"


def test_normalize_log_weird_payload():
    evt = normalize({"type": "log", "data": {"foo": "bar"}})
    assert evt.kind == "log"
    assert evt.message is not None and "foo" in evt.message


def test_normalize_legacy_flat_running():
    evt = normalize({"status": "running", "progress": 0.5, "message": "m"})
    assert evt.kind == "progress"
    assert evt.progress == 50.0
    assert evt.message == "m"


def test_normalize_legacy_flat_done():
    evt = normalize({"status": "done", "progress": 1.0, "result": {"x": 1}})
    assert evt.kind == "done"
    assert evt.status == "completed"


def test_normalize_legacy_flat_failed():
    evt = normalize({"status": "failed", "message": "nope"})
    assert evt.kind == "error"
    assert evt.message == "nope"


def test_normalize_unknown_shape_doesnt_crash():
    evt = normalize({"garbage": 1})
    assert isinstance(evt, NormalizedEvent)
    assert evt.kind in ("log", "unknown")


def test_normalize_non_dict_input():
    evt = normalize("not-a-dict")  # type: ignore[arg-type]
    assert evt.kind == "unknown"
