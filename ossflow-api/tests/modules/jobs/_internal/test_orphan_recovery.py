"""Tests de la utilidad ``mark_running_as_failed``."""

from __future__ import annotations

from dataclasses import dataclass

from ossflow_api.modules.jobs._internal.orphan_recovery import (
    ORPHAN_ERROR_MESSAGE,
    mark_running_as_failed,
)


@dataclass
class _FakeJob:
    """Mínimo objeto con atributo ``status`` que admite mutación."""
    status: str
    error: str | None = None
    completed_at: str | None = None


def _setter(job: _FakeJob, status: str, error: str, completed_at: str) -> None:
    job.status = status
    job.error = error
    job.completed_at = completed_at


def test_marks_running_as_failed():
    jobs = [_FakeJob(status="running"), _FakeJob(status="queued"), _FakeJob(status="completed")]
    count = mark_running_as_failed(jobs, _setter)
    assert count == 2
    assert jobs[0].status == "failed"
    assert jobs[0].error == ORPHAN_ERROR_MESSAGE
    assert jobs[0].completed_at is not None
    assert jobs[1].status == "failed"
    assert jobs[2].status == "completed"  # no tocado
    assert jobs[2].error is None


def test_returns_zero_when_no_orphans():
    jobs = [_FakeJob(status="completed"), _FakeJob(status="failed")]
    count = mark_running_as_failed(jobs, _setter)
    assert count == 0


def test_handles_empty_iterable():
    count = mark_running_as_failed([], _setter)
    assert count == 0
