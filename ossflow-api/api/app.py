"""
BJJ Instructional Processor — Web UI

FastAPI application that provides a web interface for:
- Browsing the instructional video library
- Detecting and previewing chapters
- Generating subtitles
- Launching processing pipelines
- Monitoring progress in real-time via SSE

Run:
    cd web
    uvicorn app:app --reload --port 8000
    # or: python app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response

from api.event_normalizer import normalize
from api.paths import to_container_path
from api.settings import get_library_path
from api.backend_client import (
    BackendClient,
    BackendError,
    dubbing_client,
    splitter_client,
    subs_client,
)
from api.jobs_store import JobsStore
from api.settings import CONFIG_DIR as _CONFIG_DIR

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web")

BASE_DIR = Path(__file__).parent

# Add parent to path so we can import chapter_splitter / subtitle_generator
sys.path.insert(0, str(BASE_DIR.parent))

def _load_persisted_jobs() -> None:
    """Rehydrate _jobs from jobs.json at startup (Fase 3)."""
    try:
        saved = _jobs_store.load()
        for jid, data in saved.items():
            try:
                data = dict(data)
                data["status"] = JobStatus(data.get("status", "queued"))
                _jobs[jid] = JobInfo(**{
                    k: v for k, v in data.items()
                    if k in JobInfo.__dataclass_fields__
                })
            except Exception as exc:
                log.warning("Skipping bad persisted job %s: %s", jid, exc)
    except Exception as exc:
        log.warning("Failed to load persisted jobs: %s", exc)


def _auto_mount_on_startup() -> None:
    """Auto-mount NAS if mount config exists from previous session."""
    config_dir = Path(os.environ.get("CONFIG_DIR", "/data/config"))
    mount_cfg = config_dir / "mount.json"
    if not mount_cfg.exists():
        return

    import json, subprocess
    try:
        cfg = json.loads(mount_cfg.read_text())
        share = cfg.get("share", "")
        username = cfg.get("username", "guest")
        password = cfg.get("password", "")
        if not share:
            return

        media = Path(MEDIA_ROOT)
        media.mkdir(parents=True, exist_ok=True)

        # Skip if already mounted. Timeout defensivo: si el mount está "zombie"
        # (CIFS colgado), ``mountpoint`` se queda estancado llamando stat() y
        # bloqueaba todo el arranque del contenedor.
        result = subprocess.run(
            ["mountpoint", "-q", str(media)],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return

        opts = f"username={username},password={password},iocharset=utf8,noperm"
        subprocess.run(["mount", "-t", "cifs", share, str(media), "-o", opts],
                       capture_output=True, timeout=15)
    except Exception:
        pass  # Silent fail on startup — user can re-mount from the UI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _load_persisted_jobs()
    _auto_mount_on_startup()
    yield
    # Shutdown: cerrar httpx.AsyncClient compartido del módulo preflight.
    # Acoplamiento #5 roto: app.py ya no toca privados (aclose_shared_client),
    # invoca el método público del servicio. Cuando app.py migre a
    # ossflow_api/main.py este hook se registrará vía
    # infrastructure.lifespan.register_shutdown.
    try:
        from ossflow_api.modules.preflight.service import PreflightService
        await PreflightService.aclose()
    except Exception:
        pass


app = FastAPI(title="BJJ Instructional Processor", version="2.0.0", lifespan=lifespan)

# CORS — allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers (E7)
from ossflow_api.modules.settings import settings_router  # noqa: E402
from api.pipeline import router as pipeline_router   # noqa: E402
from ossflow_api.modules.preflight import preflight_router  # noqa: E402
from ossflow_api.modules.logs import (  # noqa: E402
    install_local_ring_buffer as _install_ring_buffer,
    logs_router,
)

# Instalar el ring buffer en import time mantiene el comportamiento previo.
# La gestión por lifespan se cableará cuando app.py migre a ossflow_api/main.py.
_install_ring_buffer()
from ossflow_api.modules.metrics import metrics_router  # noqa: E402
from ossflow_api.modules.metadata import metadata_router  # noqa: E402
from ossflow_api.modules.chapters import chapters_router as chapters_router  # noqa: E402
from ossflow_api.modules.cleanup import cleanup_router  # noqa: E402
from ossflow_api.modules.duplicates import duplicates_router  # noqa: E402
from api.background_jobs import router as bg_jobs_router  # noqa: E402
from ossflow_api.modules.health import health_router as health_proxy_router  # noqa: E402
from ossflow_api.modules.subtitles import subtitles_router as subtitles_router  # noqa: E402
from ossflow_api.modules.dubbing import (  # noqa: E402
    burn_subs_router,
    dubbing_router as dubbing_router,
)
from ossflow_api.modules.promote import promote_router as promote_router  # noqa: E402
from ossflow_api.modules.scrapper import scrapper_router  # noqa: E402
from ossflow_api.modules.voices import voices_router  # noqa: E402
from ossflow_api.modules.export import export_router  # noqa: E402
from ossflow_api.modules.library import library_router  # noqa: E402
# WIRE_TELEGRAM_ROUTER
from ossflow_api.modules.telegram import telegram_router  # noqa: E402

app.include_router(settings_router)
# IMPORTANTE: preflight_router comparte prefix "/api/pipeline" con pipeline_router
# y pipeline_router tiene una ruta catch-all `GET /{pipeline_id}`. Hay que
# registrar preflight ANTES para que `/preflight` no sea capturado como id.
app.include_router(preflight_router)
app.include_router(pipeline_router)
app.include_router(logs_router)
app.include_router(metrics_router)
app.include_router(metadata_router)
app.include_router(chapters_router)
app.include_router(cleanup_router)
app.include_router(duplicates_router)
app.include_router(bg_jobs_router)
app.include_router(burn_subs_router)
app.include_router(health_proxy_router)
app.include_router(subtitles_router)
app.include_router(dubbing_router)
app.include_router(promote_router)
app.include_router(voices_router)
app.include_router(export_router)
app.include_router(library_router)
app.include_router(scrapper_router)
# WIRE_TELEGRAM_ROUTER
app.include_router(telegram_router)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
SUBTITLE_EXTENSIONS = {".srt"}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobInfo:
    job_id: str
    job_type: str  # "chapters", "subtitles", "translate", "dubbing"
    video_path: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Optional[dict] = None


# In-memory job store + persistent mirror
_jobs: dict[str, JobInfo] = {}
_job_events: dict[str, asyncio.Queue] = {}
_jobs_store = JobsStore(_CONFIG_DIR / "jobs.json")


def _persist_job(job: JobInfo) -> None:
    """Mirror a job dataclass into the JSON store."""
    try:
        _jobs_store.upsert(job.job_id, asdict(job))
    except Exception as exc:  # pragma: no cover - never block the pipeline
        log.warning("Failed to persist job %s: %s", job.job_id, exc)


# ---------------------------------------------------------------------------
# Library scanning  → migrado a ``ossflow_api.modules.library.service`` (T23.3)
# ---------------------------------------------------------------------------


# T23.5: video metadata + thumbnail viven en modules/library/media.py.
# Compat shim para pipeline.py (import diferido evita ciclo).
from ossflow_api.modules.library.media import (  # noqa: E402,F401
    video_info as get_video_info,
    generate_thumbnail,
)


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def _infer_event_type(data: dict) -> str:
    """Infer SSE event type when not explicit.

    Precedence: explicit ``type`` > ``status`` transition > ``progress`` > ``log``.
    """
    if "type" in data and data["type"]:
        return data["type"]
    if "status" in data:
        return "status"
    if "progress" in data:
        return "progress"
    return "log"


async def _parse_json_body(request: Request) -> dict:
    """Parse a JSON body, raising 400 on invalid payloads.

    Motivación: ``await request.json()`` levanta ``JSONDecodeError`` si el
    cliente manda JSON mal formado, y FastAPI lo transforma en 500 opaco
    que no dice QUÉ se rompió. Además, si el body es un array o un
    literal (``"hello"``, ``42``), el código siguiente accede con
    ``body.get(...)`` y levanta ``AttributeError`` → también 500.
    Normalizamos a dict y devolvemos 400 con mensaje claro.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Expected JSON object, got {type(body).__name__}",
        )
    return body


async def _emit(job_id: str, data: dict):
    """Send an SSE event to the job's event queue."""
    # Ensure every event carries an explicit ``type``.
    if "type" not in data or not data.get("type"):
        data = {**data, "type": _infer_event_type(data)}
    q = _job_events.get(job_id)
    if q:
        await q.put(data)
    # Mirror status transitions to the persistent store
    job = _jobs.get(job_id)
    if job is not None and "status" in data:
        _persist_job(job)


async def _run_remote(
    client: BackendClient,
    payload: dict,
    job: JobInfo,
    *,
    label: str,
) -> None:
    """Generic HTTP runner: POST /run + SSE stream -> job events.

    Replaces the legacy ``asyncio.create_subprocess_exec`` loop. The
    backend microservice owns the processing logic; we just relay.
    """
    job.status = JobStatus.RUNNING
    await _emit(job.job_id, {"status": "running", "message": f"Starting {label}..."})

    try:
        remote_job_id = await client.run(payload)
        log.info("[%s] remote job_id=%s", label, remote_job_id)

        async for evt in client.stream(remote_job_id):
            # evt is a NormalizedEvent (see api.event_normalizer).
            # Be tolerant of test doubles that yield raw dicts.
            if isinstance(evt, dict):
                evt = normalize(evt)
            if evt.progress is not None:
                job.progress = evt.progress
            if evt.message is not None:
                job.message = evt.message

            if evt.kind == "error":
                job.status = JobStatus.FAILED
                job.message = evt.message or "backend error"
                job.completed_at = datetime.now().isoformat()
                await _emit(job.job_id, {"status": "failed", "message": job.message})
                return

            if evt.kind == "done":
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.completed_at = datetime.now().isoformat()
                job.result = evt.payload.get("result") or {}
                await _emit(job.job_id, {
                    "status": "completed",
                    "progress": 100,
                    "result": job.result,
                })
                return

            # Intermediate event (progress / log) -> flat SSE for frontend
            await _emit(job.job_id, {
                "progress": job.progress,
                "message": job.message,
                **({"status": evt.status} if evt.status else {}),
            })

        # Stream ended without explicit terminal event -> treat as completed
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.now().isoformat()
        await _emit(job.job_id, {"status": "completed", "progress": 100, "result": job.result or {}})

    except BackendError as e:
        job.status = JobStatus.FAILED
        job.message = f"backend error: {e}"
        await _emit(job.job_id, {"status": "failed", "message": job.message})
    except Exception as e:  # pragma: no cover - defensive
        job.status = JobStatus.FAILED
        job.message = str(e)
        await _emit(job.job_id, {"status": "failed", "message": str(e)})


def _translate_job_path(host_path: str) -> tuple[str, str]:
    """Return (container_input_path, container_output_dir) for a job.

    When no library_path is configured (e.g. in unit tests) the path is
    passed through unchanged, preserving legacy behaviour.
    """
    lib = get_library_path() or ""
    if not lib:
        return host_path, str(Path(host_path).parent)
    try:
        ci = to_container_path(host_path, lib)
    except ValueError:
        # Path is outside the configured library — fall back to raw path.
        return host_path, str(Path(host_path).parent)
    co = ci.rsplit("/", 1)[0] or "/library"
    return ci, co


async def run_chapter_detection(job: JobInfo):
    """Delegate chapter detection to the splitter microservice."""
    ci, co = _translate_job_path(job.video_path)
    payload = {
        "input_path": ci,
        "output_dir": co,
        "options": {"dry_run": True, "verbose": True},
    }
    await _run_remote(splitter_client(), payload, job, label="chapter detection")


async def run_subtitle_generation(job: JobInfo):
    """Delegate subtitle generation to the subs microservice."""
    ci, co = _translate_job_path(job.video_path)
    payload = {
        "input_path": ci,
        "output_dir": co,
        "options": {"verbose": True},
    }
    await _run_remote(subs_client(), payload, job, label="subtitle generation")
    _reindex_search_silent(job.video_path)


def _reindex_search_silent(video_path: str) -> None:
    """Rebuild the subtitle index over the instructional folder. Best-effort."""
    try:
        from search.indexer import SubtitleIndexer
        p = Path(video_path)
        root = p if p.is_dir() else p.parent
        # Climb to instructional root (parent of "Season XX" if present).
        if root.name.lower().startswith("season"):
            root = root.parent
        SubtitleIndexer().build_index(root)
    except Exception as exc:
        log.warning("reindex after subtitles failed: %s", exc)


async def run_translation(job: JobInfo):
    """Delegate EN->ES SRT translation to the subtitle microservice (OpenAI-aware)."""
    from api.settings import get_setting

    ci, co = _translate_job_path(job.video_path)

    provider = (get_setting("translation_provider") or "ollama").lower()
    fallback = (get_setting("translation_fallback_provider") or "").lower() or None
    model = get_setting("translation_model")

    topts: dict = {
        "translate_only": True,
        "verbose": True,
        "target_lang": "ES",
        "source_lang": "EN",
        "provider": provider,
    }
    if model:
        topts["model"] = model

    key = (
        get_setting("openai_api_key") if provider == "openai"
        else None  # ollama no necesita key
    )
    if key:
        topts["api_key"] = key

    if fallback and fallback != provider:
        fb_key = (
            get_setting("openai_api_key") if fallback == "openai"
            else None  # ollama no necesita key
        )
        topts["fallback_provider"] = fallback
        if fb_key:
            topts["fallback_api_key"] = fb_key

    payload = {
        "input_path": ci,
        "output_dir": co,
        "options": topts,
    }
    await _run_remote(subs_client(), payload, job, label="translation")


async def run_dubbing(job: JobInfo, voice_profile: Optional[str] = None, use_model_voice: bool = False):
    """Delegate dubbing to the dubbing microservice."""
    ci, co = _translate_job_path(job.video_path)
    opts: dict = {"skip_translation": True}
    if voice_profile:
        opts["voice_profile"] = voice_profile
    if use_model_voice:
        opts["use_model_voice"] = True
    payload = {
        "input_path": ci,
        "output_dir": co,
        "options": opts,
    }
    await _run_remote(dubbing_client(), payload, job, label="dubbing")


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "/media")





@app.post("/api/jobs")
async def api_create_job(request: Request):
    body = await _parse_json_body(request)
    job_type = body.get("type")  # "chapters", "subtitles", "translate", "dubbing"
    video_path = body.get("path")

    valid_types = {"chapters", "subtitles", "translate", "dubbing"}
    if not job_type or not video_path:
        return JSONResponse({"error": "Missing type or path"}, status_code=400)
    if job_type not in valid_types:
        return JSONResponse(
            {"error": f"Unknown job type: {job_type}. Valid: {sorted(valid_types)}"},
            status_code=400,
        )
    if not Path(video_path).exists():
        return JSONResponse({"error": "Video/path not found"}, status_code=404)

    job_id = str(uuid.uuid4())[:8]
    job = JobInfo(job_id=job_id, job_type=job_type, video_path=video_path)
    _jobs[job_id] = job
    _job_events[job_id] = asyncio.Queue()
    _persist_job(job)

    # Launch in background
    if job_type == "chapters":
        asyncio.create_task(run_chapter_detection(job))
    elif job_type == "subtitles":
        asyncio.create_task(run_subtitle_generation(job))
    elif job_type == "translate":
        asyncio.create_task(run_translation(job))
    elif job_type == "dubbing":
        voice_profile = body.get("voice_profile")
        use_model_voice = body.get("use_model_voice", False)
        asyncio.create_task(run_dubbing(job, voice_profile=voice_profile, use_model_voice=use_model_voice))

    return {"job_id": job_id, "status": job.status.value}


@app.get("/api/jobs")
async def api_list_jobs():
    return {"jobs": [asdict(j) for j in _jobs.values()]}


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return asdict(job)


@app.get("/api/jobs/{job_id}/events")
async def api_job_events(job_id: str):
    """SSE endpoint for real-time job progress."""
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    q = _job_events.get(job_id)
    if not q:
        q = asyncio.Queue()
        _job_events[job_id] = q

    async def event_stream():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    # SSE comment = keepalive, invisible to EventSource clients.
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") in ("completed", "failed"):
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/subtitles")
async def api_get_subtitles(path: str):
    """Read an SRT file and return parsed subtitles."""
    srt_path = Path(path)
    if not srt_path.exists():
        return JSONResponse({"error": "SRT not found"}, status_code=404)

    content = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = []
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n$|\Z)",
        re.DOTALL
    )
    for m in pattern.finditer(content):
        blocks.append({
            "index": int(m.group(1)),
            "start": m.group(2),
            "end": m.group(3),
            "text": m.group(4).strip(),
        })
    return {"subtitles": blocks, "total": len(blocks)}


# ---------------------------------------------------------------------------
# MKV Chapters
# ---------------------------------------------------------------------------

@app.post("/api/chapters/embed")
async def api_embed_chapters(request: Request):
    """Embed chapter metadata into an MKV file."""
    from chapter_tools.mkv_chapters import MkvChapterGenerator

    body = await _parse_json_body(request)
    video_path = body.get("path", "")
    chapters = body.get("chapters", [])

    if not video_path:
        return JSONResponse({"error": "Missing 'path'"}, status_code=400)
    if not Path(video_path).exists():
        return JSONResponse({"error": "Video file not found"}, status_code=404)
    if not chapters:
        return JSONResponse({"error": "Missing 'chapters' list"}, status_code=400)

    try:
        gen = MkvChapterGenerator()
        xml_path = Path(video_path).with_suffix(".chapters.xml")
        gen.generate_xml(chapters, xml_path)
        success = gen.embed_chapters(Path(video_path), xml_path)

        # Clean up temporary XML
        if xml_path.exists():
            xml_path.unlink()

        if success:
            return {"ok": True, "message": f"Embedded {len(chapters)} chapters into {video_path}"}
        return JSONResponse(
            {"error": "mkvmerge failed. Is MKVToolNix installed?"},
            status_code=500,
        )
    except Exception as exc:
        log.error("embed_chapters failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/chapters/extract")
async def api_extract_chapters(path: str):
    """Extract chapters from an MKV file."""
    from chapter_tools.mkv_chapters import MkvChapterGenerator

    if not Path(path).exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    try:
        gen = MkvChapterGenerator()
        chapters = gen.extract_chapters(Path(path))
        return {"chapters": chapters, "total": len(chapters)}
    except Exception as exc:
        log.error("extract_chapters failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


# Endpoints /api/voice-profiles migrados a ossflow_api/modules/voices (T25).


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.post("/api/search/build-index")
async def api_build_search_index(request: Request):
    """Build or rebuild the subtitle search index."""
    from search.indexer import SubtitleIndexer

    body = await _parse_json_body(request)
    root_dir = body.get("path", "")

    if not root_dir:
        return JSONResponse({"error": "Missing 'path'"}, status_code=400)
    if not Path(root_dir).exists():
        return JSONResponse({"error": "Path does not exist"}, status_code=404)

    try:
        indexer = SubtitleIndexer()
        count = indexer.build_index(Path(root_dir))
        stats = indexer.get_stats()
        return {"ok": True, "indexed": count, "stats": stats}
    except Exception as exc:
        log.error("build_search_index failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/search")
async def api_search(q: str, limit: int = 50):
    """Search across all indexed subtitles."""
    from search.indexer import SubtitleIndexer

    if not q:
        return JSONResponse({"error": "Missing query parameter 'q'"}, status_code=400)

    try:
        indexer = SubtitleIndexer()
        results = indexer.search(q, limit=limit)
        return {
            "query": q,
            "results": [r.to_dict() for r in results],
            "total": len(results),
        }
    except Exception as exc:
        log.error("search failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
