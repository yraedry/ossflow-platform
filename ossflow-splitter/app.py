"""FastAPI entrypoint for chapter-splitter backend.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the shared kit (one level up) is importable whether this is run from
# the project directory or installed.
_KIT_PARENT = Path(__file__).resolve().parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))

from bjj_service_kit import JobEvent, RunRequest, create_app, emit_logs  # noqa: E402


SERVICE_NAME = "chapter-splitter"


def _resolve_root(path: Path, emit) -> Path:
    """If ``path`` is a file, promote to its parent directory.

    Emits a log event documenting the promotion. Raises FileNotFoundError if
    the resolved path does not exist or is not a directory.
    """
    if path.is_file():
        promoted = path.parent
        emit(JobEvent(type="log", data={"message": f"Promoted file to directory: {promoted}"}))
        path = promoted
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    return path


def _run_chapter_splitter(req: RunRequest, emit) -> None:
    """Bridge RunRequest -> chapter_splitter.pipeline.Pipeline."""
    root_dir = _resolve_root(Path(req.input_path), emit)

    with emit_logs(emit, level=logging.INFO):
        from chapter_splitter.config import Config  # type: ignore
        from chapter_splitter.pipeline import Pipeline  # type: ignore
        from chapter_splitter.utils import setup_logging  # type: ignore

        opts = req.options or {}
        setup_logging(bool(opts.get("verbose", False)))

        config = Config(
            root_dir=root_dir,
            dry_run=bool(opts.get("dry_run", False)),
            verbose=bool(opts.get("verbose", False)),
            voice_threshold=float(opts.get("voice_threshold", 0.25)),
            voice_enter_threshold=float(opts.get("voice_enter", 0.30)),
            voice_exit_threshold=float(opts.get("voice_exit", 0.20)),
            scan_step=float(opts.get("scan_step", 0.5)),
            ocr_confidence_min=float(opts.get("ocr_confidence", 0.55)),
        )

        emit(JobEvent(type="log", data={"message": f"starting chapter-splitter on {root_dir}"}))
        Pipeline(config).run()
        emit(JobEvent(type="progress", data={"pct": 100}))


app = create_app(service_name=SERVICE_NAME, task_fn=_run_chapter_splitter)


# ---------------------------------------------------------------------------
# Oracle HTTP endpoints (search / scrape / providers)
# Consumed by processor-api as proxy.
# ---------------------------------------------------------------------------

from fastapi import HTTPException as _HTTPException  # noqa: E402
from pydantic import BaseModel as _BaseModel  # noqa: E402

from chapter_splitter.oracle import (  # noqa: E402
    OracleError,
    ProviderNotFoundError,
    ProviderScrapeError,
    ProviderSearchError,
    ProviderTimeoutError,
    discover as _oracle_discover,
    registry as _oracle_registry,
)

_oracle_discover()


class _SearchReq(_BaseModel):
    title: str
    author: str | None = None
    provider_id: str | None = None


class _ScrapeReq(_BaseModel):
    url: str
    provider_id: str | None = None


@app.get("/oracle/providers")
def list_oracle_providers() -> list[dict]:
    return [
        {"id": p.id, "display_name": p.display_name, "domains": list(p.domains)}
        for p in _oracle_registry.all()
    ]


@app.post("/oracle/search")
def oracle_search(req: _SearchReq) -> list[dict]:
    try:
        if req.provider_id:
            providers = [_oracle_registry.get(req.provider_id)]
        else:
            providers = _oracle_registry.all()
        if not providers:
            raise _HTTPException(status_code=503, detail="no oracle providers registered")
        all_candidates: list = []
        for p in providers:
            try:
                all_candidates.extend(p.search(req.title, req.author))
            except OracleError as exc:
                logging.getLogger(__name__).warning(
                    "provider %s search failed: %s", p.id, exc
                )
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        return [c.model_dump() for c in all_candidates]
    except ProviderNotFoundError as exc:
        raise _HTTPException(status_code=404, detail=str(exc))
    except ProviderTimeoutError as exc:
        raise _HTTPException(status_code=504, detail=str(exc))
    except ProviderSearchError as exc:
        raise _HTTPException(status_code=502, detail=str(exc))


@app.post("/oracle/scrape")
def oracle_scrape(req: _ScrapeReq) -> dict:
    try:
        provider = (
            _oracle_registry.get(req.provider_id)
            if req.provider_id
            else _oracle_registry.resolve_by_url(req.url)
        )
        result = provider.scrape(req.url)
        return result.model_dump()
    except ProviderNotFoundError as exc:
        raise _HTTPException(status_code=404, detail=str(exc))
    except ProviderTimeoutError as exc:
        raise _HTTPException(status_code=504, detail=str(exc))
    except ProviderScrapeError as exc:
        raise _HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Oracle endpoint: cut by BJJFanatics-scraped timestamps
# ---------------------------------------------------------------------------
# WIRE_ORACLE_RUN_ENDPOINT  (added inline below; no central wiring required)

from typing import Any  # noqa: E402

from fastapi import HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from bjj_service_kit.events import sse_generator  # noqa: E402


class OracleRunRequest(BaseModel):
    path: str
    oracle: dict[str, Any]
    output_dir: str | None = None


def _run_oracle_task(req: OracleRunRequest, emit) -> None:
    from chapter_splitter.oracle.models import OracleResult  # type: ignore
    from chapter_splitter.splitting.oracle_splitter import (  # type: ignore
        OracleSplitter,
    )

    instructional_dir = Path(req.path)
    if instructional_dir.is_file():
        instructional_dir = instructional_dir.parent
    if not instructional_dir.exists() or not instructional_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {instructional_dir}")

    oracle = OracleResult.model_validate(req.oracle)
    output_dir = Path(req.output_dir) if req.output_dir else None

    def progress_cb(pct: float, message: str) -> None:
        emit(JobEvent(type="progress", data={"pct": pct, "message": message}))

    with emit_logs(emit, level=logging.INFO):
        emit(JobEvent(
            type="log",
            data={"message": f"oracle-split starting on {instructional_dir}"},
        ))
        splitter = OracleSplitter(
            instructional_dir=instructional_dir,
            oracle=oracle,
            output_dir=output_dir,
        )
        report = splitter.split(progress_cb=progress_cb)
        emit(JobEvent(type="progress", data={"pct": 100, "message": "done"}))
        # Emit a structured "done"-like payload as a log so the SSE consumer
        # gets the report. The runner appends its own canonical "done" event
        # after task_fn returns, so we surface the report explicitly here.
        emit(JobEvent(
            type="log",
            data={"message": "report", "report": report.to_dict()},
        ))


@app.post("/run-oracle")
def run_oracle(req: OracleRunRequest) -> dict[str, str]:
    """Submit an oracle-driven split job. Mirrors POST /run."""
    log = logging.getLogger(__name__)
    log.info("run-oracle request: path=%s, oracle_keys=%s, output_dir=%s",
             req.path, list(req.oracle.keys()) if req.oracle else None, req.output_dir)
    if not req.path:
        raise HTTPException(status_code=400, detail="path is required")
    p = Path(req.path)
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    if ".." in p.parts:
        raise HTTPException(
            status_code=400, detail="path traversal not allowed",
        )

    # Reuse the kit's job registry so /events/{job_id} works unchanged.
    registry = app.state.runner.registry
    job_id, q = registry.create()

    import threading

    def emit(evt: JobEvent) -> None:
        q.put(evt)

    def target() -> None:
        try:
            _run_oracle_task(req, emit)
            emit(JobEvent(type="done", data={"job_id": job_id}))
        except Exception as exc:  # noqa: BLE001
            log.exception("oracle job %s failed", job_id)
            emit(JobEvent(type="error", data={"message": str(exc)}))
        finally:
            q.close()

    threading.Thread(
        target=target, daemon=True, name=f"oracle-job-{job_id}",
    ).start()
    return {"job_id": job_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
