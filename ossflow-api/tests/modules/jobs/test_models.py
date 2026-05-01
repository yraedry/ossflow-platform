"""Tests de las dataclasses de dominio."""

from __future__ import annotations

from ossflow_api.modules.jobs.models import (
    BackgroundJob,
    JobStatus,
    LegacyJob,
)


def test_job_status_string_values_match_legacy_contract():
    """Los valores string deben ser idénticos a los del sistema legacy."""
    assert JobStatus.QUEUED.value == "queued"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"


def test_background_job_default_status_is_queued():
    job = BackgroundJob(id="abc", type="cleanup_scan")
    assert job.status == "queued"
    assert job.params == {}
    assert job.result is None
    assert job.completed_at is None


def test_background_job_to_dict_round_trip():
    job = BackgroundJob(id="abc", type="cleanup_scan", progress=42.0, message="x")
    d = job.to_dict()
    assert d["id"] == "abc"
    assert d["type"] == "cleanup_scan"
    assert d["progress"] == 42.0
    assert d["message"] == "x"
    assert d["status"] == "queued"


def test_legacy_job_preserves_video_path_field():
    """El JSON shape externo del frontend depende de que ``video_path`` sea
    un campo de primer nivel — no anidado en ``params``."""
    job = LegacyJob(job_id="j1", job_type="dubbing", video_path="/media/x.mp4")
    d = job.to_dict()
    assert d["video_path"] == "/media/x.mp4"
    assert d["job_id"] == "j1"
    assert d["job_type"] == "dubbing"
    assert "params" not in d  # NO debe existir, el shape es plano


def test_legacy_job_distinct_field_names_from_background():
    """``id`` vs ``job_id`` y ``type`` vs ``job_type``: divergencia
    intencional para preservar JSON shapes externos heredados."""
    bg = BackgroundJob(id="x", type="cleanup_scan")
    legacy = LegacyJob(job_id="x", job_type="dubbing", video_path="/p.mp4")
    assert "id" in bg.to_dict() and "type" in bg.to_dict()
    assert "job_id" in legacy.to_dict() and "job_type" in legacy.to_dict()
