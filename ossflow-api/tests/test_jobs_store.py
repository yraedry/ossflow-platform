"""Tests for JobsStore persistence."""

from __future__ import annotations

from api.jobs_store import JobsStore


def test_load_missing_returns_empty(tmp_path):
    store = JobsStore(tmp_path / "jobs.json")
    assert store.load() == {}


def test_save_then_load_roundtrip(tmp_path):
    store = JobsStore(tmp_path / "jobs.json")
    jobs = {"j1": {"status": "done", "progress": 1.0}}
    store.save(jobs)
    assert store.load() == jobs


def test_upsert_preserves_existing(tmp_path):
    store = JobsStore(tmp_path / "jobs.json")
    store.upsert("j1", {"status": "running"})
    store.upsert("j2", {"status": "queued"})
    loaded = store.load()
    assert set(loaded) == {"j1", "j2"}
    assert loaded["j1"]["status"] == "running"


def test_save_creates_parent_dir(tmp_path):
    target = tmp_path / "nested" / "dir" / "jobs.json"
    store = JobsStore(target)
    store.save({"j": {"x": 1}})
    assert target.exists()


def test_load_recovers_from_corrupted_file(tmp_path):
    target = tmp_path / "jobs.json"
    target.write_text("{not json")
    store = JobsStore(target)
    assert store.load() == {}
