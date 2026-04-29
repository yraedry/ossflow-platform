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
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
from api.library_refresh import (
    refresh_instructional_flags,
    rediscover_instructional,
)
from api.scan_cache import (
    ScanCache,
    enrich_with_poster,
    find_poster,
    find_poster_cached,
    patch_poster_in_cache,
)
from api.settings import CONFIG_DIR as _CONFIG_DIR

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

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
    # Re-attach to ElevenLabs jobs that were mid-flight when we last
    # stopped. Done after _load_persisted_jobs so the _jobs registry is
    # populated; before yield so resume tasks start concurrently with
    # normal request handling.
    try:
        from api.elevenlabs_dubbing import resume_orphan_jobs
        resume_orphan_jobs()
    except Exception as exc:
        log.warning("elevenlabs resume_orphan_jobs failed: %s", exc)
    yield
    # Shutdown: cerrar httpx.AsyncClient compartido del módulo preflight
    try:
        from api.preflight import aclose_shared_client
        await aclose_shared_client()
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

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Register routers (E7)
from api.settings import router as settings_router  # noqa: E402
from api.pipeline import router as pipeline_router   # noqa: E402
from api.preflight import router as preflight_router  # noqa: E402
from api.logs_view import router as logs_router  # noqa: E402
from api.metrics import router as metrics_router  # noqa: E402
from api.metadata import router as metadata_router  # noqa: E402
from api.chapters import router as chapters_router  # noqa: E402
from api.cleanup import router as cleanup_router  # noqa: E402
from api.duplicates import router as duplicates_router  # noqa: E402
from api.background_jobs import router as bg_jobs_router  # noqa: E402
from api.burn_subs import router as burn_subs_router  # noqa: E402
from api.health_proxy import router as health_proxy_router  # noqa: E402
from api.subtitles import router as subtitles_router  # noqa: E402
from api.dubbing import router as dubbing_router  # noqa: E402
from api.elevenlabs_dubbing import router as elevenlabs_dubbing_router  # noqa: E402
from api.promote import router as promote_router  # noqa: E402
# WIRE_ORACLE_ROUTER
from api import oracle as oracle_module  # noqa: E402
# WIRE_TELEGRAM_ROUTER
from api import telegram as telegram_module  # noqa: E402

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
app.include_router(elevenlabs_dubbing_router)
app.include_router(promote_router)
# WIRE_ORACLE_ROUTER
app.include_router(oracle_module.router)
# WIRE_TELEGRAM_ROUTER
app.include_router(telegram_module.router)

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
_scan_cache = ScanCache(_CONFIG_DIR / "library.json")
_library_refresh_lock = asyncio.Lock()
_library_refresh_inflight = False


def _persist_job(job: JobInfo) -> None:
    """Mirror a job dataclass into the JSON store."""
    try:
        _jobs_store.upsert(job.job_id, asdict(job))
    except Exception as exc:  # pragma: no cover - never block the pipeline
        log.warning("Failed to persist job %s: %s", job.job_id, exc)


# ---------------------------------------------------------------------------
# Library scanning
# ---------------------------------------------------------------------------

def scan_library(root_path: str) -> list[dict]:
    """Scan a directory tree for instructional videos and their processing state."""
    root = Path(root_path)
    if not root.exists():
        return []

    instructionals: dict[str, dict] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip artefact subfolders — mutating ``dirnames`` in place
        # short-circuits ``os.walk`` so we never descend into them.
        # ``elevenlabs/`` holds Studio E2E MP4s and ``doblajes/`` holds
        # the flujo-B pipeline output. Both would otherwise show up as
        # bogus "Seasons" with duplicated episode names.
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in ("elevenlabs", "doblajes")
        ]

        dp = Path(dirpath)
        # Exclude dubbed outputs. Match both ``name_DOBLADO.ext`` and
        # ``name_DOBLADO_<suffix>.ext`` — the backup/comparison flow
        # used to produce ``_DOBLADO_B_v5.mkv`` etc., which slipped
        # through the old endswith("_DOBLADO") filter.
        videos = sorted(
            f for f in filenames
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS
            and "_DOBLADO" not in Path(f).stem
        )
        if not videos:
            continue

        # Determine instructional name + root folder (posters live at root, not inside Season XX).
        folder_name = dp.name
        if "season" in folder_name.lower():
            instr_name = dp.parent.name
            instr_path = dp.parent
        else:
            instr_name = folder_name
            instr_path = dp

        if instr_name not in instructionals:
            # Read author from sidecar if present
            sidecar = instr_path / ".bjj-meta.json"
            author = ""
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    author = meta.get("instructor", "") or ""
                except (OSError, ValueError):
                    pass
            instructionals[instr_name] = {
                "name": instr_name,
                "path": str(instr_path),
                "author": author,
                "videos": [],
                "total_videos": 0,
                "chapters_detected": 0,
                "subtitled": 0,
                "dubbed": 0,
            }

        for vf in videos:
            vpath = dp / vf
            base = vpath.stem

            # Check processing state
            has_srt = (dp / f"{base}.en.srt").exists() or (dp / f"{base}.srt").exists()
            has_es_srt = (
                (dp / f"{base}.es.srt").exists()
                or (dp / f"{base}.ES.srt").exists()
                or (dp / f"{base}_ES.srt").exists()
                or (dp / f"{base}_ESP_DUB.srt").exists()
            )
            # Dub present if either:
            #   - legacy: <name>_DOBLADO.mkv/mp4 next to source (old XTTS)
            #   - flujo B v5: <Season>/doblajes/<name>.mkv
            #   - Studio E2E: <Season>/elevenlabs/<name>.mp4
            has_dubbed = (
                any((dp / f"{base}{sfx}").exists() for sfx in ["_DOBLADO.mkv", "_DOBLADO.mp4"])
                or (dp / "doblajes" / f"{base}.mkv").exists()
                or (dp / "elevenlabs" / vf).exists()
            )
            is_chapter = bool(re.search(r"S\d{2}E\d{2}", vf))

            try:
                size_mb = round(vpath.stat().st_size / (1024 * 1024), 1)
            except OSError:
                size_mb = None
            instructionals[instr_name]["videos"].append({
                "filename": vf,
                "path": str(vpath),
                "size_mb": size_mb,
                "duration": None,  # resolved lazily on instructional open
                "has_subtitles_en": has_srt,
                "has_subtitles_es": has_es_srt,
                "has_dubbing": has_dubbed,
                "is_chapter": is_chapter,
            })
            instructionals[instr_name]["total_videos"] += 1
            if has_srt:
                instructionals[instr_name]["subtitled"] += 1
            if has_dubbed:
                instructionals[instr_name]["dubbed"] += 1
            if is_chapter:
                instructionals[instr_name]["chapters_detected"] += 1

    return list(instructionals.values())


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            video_stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {}
            )
            return {
                "duration": float(fmt.get("duration", 0)),
                "duration_formatted": _format_duration(float(fmt.get("duration", 0))),
                "size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 1),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "codec": video_stream.get("codec_name", "unknown"),
                "fps": _parse_fps(video_stream.get("r_frame_rate", "0/1")),
            }
    except Exception as e:
        log.error("ffprobe failed: %s", e)
    return {"duration": 0, "duration_formatted": "00:00", "size_mb": 0}


def generate_thumbnail(video_path: str, time_sec: float = 5.0) -> Optional[bytes]:
    """Generate a thumbnail from a video at the given timestamp."""
    try:
        cmd = [
            "ffmpeg", "-ss", str(time_sec), "-i", video_path,
            "-vframes", "1", "-vf", "scale=320:-1",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:
        pass
    return None


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _parse_fps(rate_str: str) -> float:
    try:
        num, den = rate_str.split("/")
        return round(int(num) / int(den), 2) if int(den) > 0 else 0
    except (ValueError, ZeroDivisionError):
        return 0


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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/scan")
async def api_scan(request: Request):
    body = await _parse_json_body(request)
    root_path = body.get("path", "")

    # E6: fall back to library_path from settings if no path provided
    if not root_path:
        from api.settings import get_library_path
        root_path = get_library_path()

    if not root_path:
        return JSONResponse(
            {"error": "Library path not configured. Provide 'path' in the request or configure it in Settings."},
            status_code=422,
        )

    # E5: validate path exists BEFORE scanning
    if not Path(root_path).exists():
        return JSONResponse(
            {"error": f"Path not accessible: {root_path}. Verify the path exists and the server has read permissions."},
            status_code=422,
        )

    import asyncio

    def _scan_sync():
        lib = scan_library(root_path)
        return enrich_with_poster(lib)

    library = await asyncio.get_event_loop().run_in_executor(None, _scan_sync)
    try:
        _scan_cache.save(library)
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to persist library cache: %s", exc)
    return {"instructionals": library}


def _kick_background_library_refresh() -> None:
    """Fire-and-forget rescan of the library.

    Never blocks the request. Coalesces concurrent refreshes via a module-level
    flag so a page reload storm doesn't queue N parallel scans.
    """
    global _library_refresh_inflight
    if _library_refresh_inflight:
        return
    from api.settings import get_library_path
    root_path = get_library_path()
    if not root_path or not Path(root_path).exists():
        return

    _library_refresh_inflight = True

    def _scan_sync():
        global _library_refresh_inflight
        try:
            lib = scan_library(root_path)
            lib = enrich_with_poster(lib)
            try:
                _scan_cache.save(lib)
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to persist library cache: %s", exc)
        except Exception as exc:  # pragma: no cover
            log.warning("background library refresh failed: %s", exc)
        finally:
            _library_refresh_inflight = False

    asyncio.get_event_loop().run_in_executor(None, _scan_sync)


@app.get("/api/library")
async def api_library(refresh: bool = False):
    """Return cached library from the last scan, or 204 if no cache yet.

    With ``refresh=1``: kicks a background rescan (fire-and-forget) and returns
    the current cache immediately. The next poll picks up the refreshed data.
    This avoids blocking the first page load on a NAS walk.
    """
    if refresh:
        _kick_background_library_refresh()

    data = _scan_cache.load()
    if data is None:
        # Cold cache: still kick a scan so the next poll has data.
        _kick_background_library_refresh()
        return {"instructionals": [], "refreshing": True}
    if isinstance(data, dict):
        data = {**data, "refreshing": _library_refresh_inflight}
    return data


@app.get("/api/library/{name}/poster")
async def api_library_poster(name: str, request: Request):
    """Serve poster image for an instructional by folder name.

    Uses the scan cache's ``poster_filename`` to avoid ``iterdir`` on NAS.
    Emits ``ETag`` based on mtime+size and honors ``If-None-Match`` (304).
    """
    from api.settings import get_library_path

    lib = get_library_path()
    if not lib:
        return JSONResponse({"error": "library_path not configured"}, status_code=404)

    base = Path(lib).resolve()
    try:
        target = (base / name).resolve()
    except OSError:
        return JSONResponse({"error": "invalid path"}, status_code=403)

    # Anti-traversal: target must be strictly under base (or equal is not useful)
    try:
        target.relative_to(base)
    except ValueError:
        return JSONResponse({"error": "path traversal denied"}, status_code=403)
    if target == base:
        return JSONResponse({"error": "invalid target"}, status_code=403)

    if not target.exists() or not target.is_dir():
        return JSONResponse({"error": "instructional not found"}, status_code=404)

    # Prefer cached poster_filename to avoid iterdir on NAS/CIFS.
    cached_poster_filename: Optional[str] = None
    try:
        cache_data = _scan_cache.load()
        if cache_data and isinstance(cache_data, dict):
            for item in cache_data.get("instructionals", []) or []:
                if item.get("name") == name:
                    cached_poster_filename = item.get("poster_filename")
                    break
    except Exception:
        cached_poster_filename = None

    poster = find_poster_cached(target, cached_poster_filename)
    if poster is None:
        return JSONResponse({"error": "poster not found"}, status_code=404)

    # ETag from mtime_ns + size — stable while file unchanged.
    try:
        st = poster.stat()
        etag = f'"{st.st_mtime_ns}-{st.st_size}"'
    except OSError:
        etag = None

    cache_control = "public, max-age=86400, stale-while-revalidate=604800"

    if etag is not None:
        inm = request.headers.get("if-none-match")
        if inm and etag in [v.strip() for v in inm.split(",")]:
            return Response(
                status_code=304,
                headers={"ETag": etag, "Cache-Control": cache_control},
            )

    ext = poster.suffix.lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "webp": "image/webp"}.get(ext, "application/octet-stream")
    headers = {"Cache-Control": cache_control}
    if etag is not None:
        headers["ETag"] = etag
    return FileResponse(
        path=str(poster),
        media_type=media_type,
        headers=headers,
    )


_ALLOWED_POSTER_EXT = {"jpg", "jpeg", "png", "webp"}


@app.post("/api/library/{name}/poster")
async def api_library_poster_upload(name: str, file: UploadFile = File(...)):
    """Upload a custom poster for an instructional. Stored as poster.<ext> in
    the instructional folder. Replaces any existing canonical poster."""
    from api.settings import get_library_path

    lib = get_library_path()
    if not lib:
        raise HTTPException(status_code=404, detail="library_path not configured")

    base = Path(lib).resolve()
    try:
        target = (base / name).resolve()
        target.relative_to(base)
    except (OSError, ValueError):
        raise HTTPException(status_code=403, detail="path traversal denied")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="instructional not found")

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in _ALLOWED_POSTER_EXT:
        raise HTTPException(status_code=415, detail=f"unsupported extension: {ext}")

    # Remove any existing canonical poster (different ext) to avoid stale ones
    for stem in ("poster", "cover"):
        for old_ext in _ALLOWED_POSTER_EXT:
            old = target / f"{stem}.{old_ext}"
            if old.exists():
                try: old.unlink()
                except OSError: pass

    dest = target / f"poster.{ext}"
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large (max 10MB)")
    dest.write_bytes(contents)
    patch_poster_in_cache(_scan_cache, name, dest.name)
    return {"saved": dest.name, "size": len(contents)}


@app.post("/api/library/{name}/poster/redownload")
async def api_library_poster_redownload(name: str):
    """Re-download poster from the BJJFanatics URL stored in oracle metadata.

    Preserves existing oracle sidecar; only replaces the local poster file.
    Returns 404 if no oracle.poster_url is present.
    """
    from api.settings import get_library_path
    from api.oracle import _download_poster_if_missing, SIDECAR_NAME
    import json as _json

    lib = get_library_path()
    if not lib:
        raise HTTPException(status_code=404, detail="library_path not configured")

    base = Path(lib).resolve()
    try:
        target = (base / name).resolve()
        target.relative_to(base)
    except (OSError, ValueError):
        raise HTTPException(status_code=403, detail="path traversal denied")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="instructional not found")

    sidecar = target / SIDECAR_NAME
    if not sidecar.exists():
        raise HTTPException(status_code=404, detail="no oracle sidecar")
    try:
        meta = _json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"invalid sidecar: {exc}")

    poster_url = (meta.get("oracle") or {}).get("poster_url")
    if not poster_url:
        raise HTTPException(status_code=404, detail="no poster_url in oracle")

    saved = await _download_poster_if_missing(target, poster_url, force=True)
    if not saved:
        raise HTTPException(status_code=502, detail="poster download failed")
    patch_poster_in_cache(_scan_cache, name, saved)
    return {"saved": saved}


@app.post("/api/library/{name}/refresh")
async def api_library_refresh(name: str):
    """Re-discover videos + re-stat sidecars for one instructional.

    Cheaper than a full /api/scan: only walks the target folder, preserves
    cached per-video fields (duration, etc.) for videos that still exist.
    """
    data = _scan_cache.load()
    if data is None or not isinstance(data, dict):
        raise HTTPException(status_code=404, detail="no scan cache")
    items = data.get("instructionals") or []
    match = next((it for it in items if it.get("name") == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail="instructional not found")

    import asyncio

    def _rediscover_sync():
        rediscover_instructional(match)
        try:
            _scan_cache.save(items)
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to persist refreshed cache: %s", exc)

    await asyncio.get_event_loop().run_in_executor(None, _rediscover_sync)
    return {"ok": True, "videos": len(match.get("videos") or [])}


_SEASON_RE = re.compile(r"(?:Season|Volume|Vol)\s*(\d+)", re.IGNORECASE)
# Fallback: canonical episode code in the filename (e.g. "Show S01E03.mkv")
_EPISODE_RE = re.compile(r"\bS(\d{1,2})E\d{1,3}\b", re.IGNORECASE)


def _season_from_path(video_path: str, instructional_path: str) -> str:
    """Derive a season label from the video path relative to the instructional.

    Priority:
    1. Folder/path segment matching "Season N", "Volume N" or "Vol N".
    2. Episode code "SNNeMMM" in the filename.
    Returns "Sin temporada" if nothing matches.
    """
    if not video_path:
        return "Sin temporada"
    # Consider only the portion under the instructional if possible
    rel = video_path
    try:
        inst_norm = instructional_path.replace("\\", "/").rstrip("/")
        vid_norm = video_path.replace("\\", "/")
        if inst_norm and vid_norm.lower().startswith(inst_norm.lower() + "/"):
            rel = vid_norm[len(inst_norm) + 1 :]
    except Exception:  # pragma: no cover
        pass
    m = _SEASON_RE.search(rel)
    if m:
        return f"Season {int(m.group(1))}"
    m = _EPISODE_RE.search(rel)
    if m:
        return f"Season {int(m.group(1))}"
    return "Sin temporada"


@app.get("/api/library/{name}")
async def api_library_detail(name: str, refresh: bool = True):
    """Return one instructional with its videos grouped/annotated by season.

    By default re-stats sidecars (``refresh=true``) so subtitles/dubbing/poster
    flags reflect current disk state without a full rescan. Pass
    ``?refresh=false`` to force pure cache read.
    """
    data = _scan_cache.load()
    if data is None:
        return JSONResponse({"error": "no scan cache"}, status_code=404)

    items = data.get("instructionals", []) if isinstance(data, dict) else []
    match = next((it for it in items if it.get("name") == name), None)
    if match is None:
        return JSONResponse({"error": "instructional not found"}, status_code=404)

    if refresh:
        import asyncio
        from api.library_refresh import rediscover_instructional, ensure_duration

        def _refresh_and_backfill():
            rediscover_instructional(match)
            for v in (match.get("videos") or []):
                if isinstance(v, dict):
                    ensure_duration(v)
            try:
                _scan_cache.save(items)
            except Exception:
                pass

        # Fire-and-forget: refresh in background, respond immediately with cache
        asyncio.get_event_loop().run_in_executor(None, _refresh_and_backfill)

    inst_path = match.get("path", "")
    raw_videos = match.get("videos", []) or []
    videos = []
    for v in raw_videos:
        if not isinstance(v, dict):
            continue
        vp = v.get("path", "")
        videos.append({
            "path": vp,
            "filename": v.get("filename") or (Path(vp).name if vp else ""),
            "season": _season_from_path(vp, inst_path),
            "size": v.get("size"),
            "duration": v.get("duration"),
            "is_chapter": v.get("is_chapter", False),
            "has_subtitles_en": v.get("has_subtitles_en", False),
            "has_subtitles_es": v.get("has_subtitles_es", False),
            "has_dubbing": v.get("has_dubbing", False),
        })

    return {
        "name": match.get("name"),
        "path": inst_path,
        "has_poster": bool(match.get("has_poster")),
        "poster_filename": match.get("poster_filename"),
        "poster_mtime": match.get("poster_mtime"),
        "videos": videos,
    }


MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "/media")


@app.get("/api/fs/browse")
async def api_fs_browse(path: str = ""):
    """List subdirectories under MEDIA_ROOT for the library picker.

    Security: rejects any path that resolves outside MEDIA_ROOT.
    """
    root = Path(MEDIA_ROOT).resolve()
    if not root.exists():
        return JSONResponse(
            {"error": f"MEDIA_ROOT no accesible: {MEDIA_ROOT}"}, status_code=503
        )

    target = (root if not path else Path(path)).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return JSONResponse(
            {"error": "Path fuera de MEDIA_ROOT"}, status_code=400
        )

    if not target.exists() or not target.is_dir():
        return JSONResponse({"error": "Directorio no existe"}, status_code=404)

    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue
            entries.append({"name": child.name, "path": str(child)})
    except PermissionError:
        return JSONResponse({"error": "Sin permisos de lectura"}, status_code=403)

    parent = None
    if target != root:
        parent = str(target.parent)

    return {
        "root": str(root),
        "path": str(target),
        "parent": parent,
        "entries": entries,
    }


# ─── NAS / Network share mount ───────────────────────────────

@app.post("/api/mount")
async def api_mount(body: dict):
    """Mount a network share (SMB/CIFS) at MEDIA_ROOT.
    Body: { "share": "//10.10.100.6/multimedia/instruccionales",
            "username": "" (optional), "password": "" (optional) }
    """
    share = body.get("share", "").strip()
    if not share:
        return JSONResponse({"error": "Campo 'share' requerido (ej: //10.10.100.6/multimedia)"}, status_code=422)

    # Normalize backslashes to forward slashes for Linux mount
    share = share.replace("\\", "/")
    if not share.startswith("//"):
        share = "//" + share.lstrip("/")

    media = Path(MEDIA_ROOT)
    media.mkdir(parents=True, exist_ok=True)

    username = body.get("username", "guest")
    password = body.get("password", "")

    # Build mount command
    opts = f"username={username},password={password},vers=3.0,iocharset=utf8,noperm"
    cmd = ["mount", "-t", "cifs", share, str(media), "-o", opts]

    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            # Try without version specification
            opts_fallback = f"username={username},password={password},iocharset=utf8,noperm"
            cmd_fallback = ["mount", "-t", "cifs", share, str(media), "-o", opts_fallback]
            result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return JSONResponse(
                    {"error": f"No se pudo montar: {result.stderr.strip()}",
                     "hint": "Verifica la IP, la ruta compartida y las credenciales."},
                    status_code=500,
                )
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Timeout al montar. Verifica que el NAS es accesible."}, status_code=500)

    # Save mount config for auto-mount on restart
    config_dir = Path(os.environ.get("CONFIG_DIR", "/data/config"))
    config_dir.mkdir(parents=True, exist_ok=True)
    mount_cfg = config_dir / "mount.json"
    import json
    mount_cfg.write_text(json.dumps({"share": share, "username": username, "password": password}))

    # Count what we can see
    dirs = [d for d in media.iterdir() if d.is_dir()]
    return {"mounted": True, "share": share, "directories": len(dirs)}


@app.get("/api/mount")
async def api_mount_status():
    """Check if MEDIA_ROOT is mounted and has content."""
    media = Path(MEDIA_ROOT)
    if not media.exists():
        return {"mounted": False, "share": None}

    # Check if it's a mount point. Timeout defensivo contra mounts CIFS
    # zombies — sin esto el endpoint bloqueaba al worker entero.
    import subprocess
    try:
        result = subprocess.run(
            ["mountpoint", "-q", str(media)],
            capture_output=True, timeout=5,
        )
        is_mount = result.returncode == 0
    except subprocess.TimeoutExpired:
        is_mount = False

    dirs = [d.name for d in media.iterdir() if d.is_dir()] if media.exists() else []
    return {"mounted": is_mount, "path": str(media), "directories": len(dirs), "items": dirs[:20]}


def _is_subpath(child: Path, parent: Path) -> bool:
    """Check if child is equal to or under parent."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


@app.get("/api/browse")
async def api_browse(path: Optional[str] = None):
    """Browse directories and video files.

    Allows free navigation starting from library_path (settings) or MEDIA_ROOT.
    No hard security clamp — the user already has filesystem access via the NAS.
    """
    from api.settings import get_library_path
    library_path = get_library_path()

    media_root = Path(MEDIA_ROOT)

    # Default to library_path if set and accessible, otherwise MEDIA_ROOT
    if not path:
        if library_path and Path(library_path).exists():
            target = Path(library_path)
        else:
            target = media_root
    else:
        target = Path(path)

    # Resolve only if it exists (UNC paths on Windows don't always resolve well)
    try:
        target = target.resolve()
    except OSError:
        pass

    # Fallback to MEDIA_ROOT if target doesn't exist
    if not target.exists():
        if media_root.exists():
            target = media_root
        else:
            return JSONResponse(
                {"error": f"Ruta no encontrada: {target}"},
                status_code=404,
            )

    if not target.is_dir():
        return JSONResponse(
            {"error": f"No es un directorio: {target}"},
            status_code=404,
        )

    # Parent: allow navigating up (None only at filesystem root)
    parent_path = target.parent
    if parent_path == target:
        parent = None  # filesystem root
    else:
        parent = str(parent_path)

    directories = []
    files = []

    try:
        entries = list(target.iterdir())
    except PermissionError:
        return JSONResponse(
            {"error": f"Sin permisos para leer: {target}"},
            status_code=403,
        )
    except OSError as exc:
        return JSONResponse(
            {"error": f"Error al leer directorio: {exc}"},
            status_code=500,
        )

    for entry in entries:
        try:
            if entry.is_dir():
                directories.append({"name": entry.name, "path": str(entry)})
            elif entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
                files.append({
                    "name": entry.name,
                    "path": str(entry),
                    "size": entry.stat().st_size,
                })
        except (PermissionError, OSError):
            continue  # skip entries we can't access

    directories.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: f["name"].lower())

    return {
        "current": str(target),
        "parent": parent,
        "directories": directories,
        "files": files,
    }


@app.get("/api/video-info")
async def api_video_info(path: str):
    if not Path(path).exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    info = get_video_info(path)
    return info


@app.get("/api/thumbnail")
async def api_thumbnail(path: str, t: float = 5.0):
    # Translate host path → container path so ffmpeg (running in api container
    # which mounts library_path as /library) can access the file.
    try:
        lib = get_library_path() or ""
        container_path = to_container_path(path, lib) if lib else path
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not Path(container_path).exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    thumb = generate_thumbnail(container_path, t)
    if thumb:
        return StreamingResponse(
            iter([thumb]),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    return JSONResponse({"error": "Could not generate thumbnail"}, status_code=500)


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


# ---------------------------------------------------------------------------
# Voice Profiles
# ---------------------------------------------------------------------------

@app.get("/api/voice-profiles")
async def api_list_voice_profiles():
    """List all instructor voice profiles."""
    from voice_profiles.manager import VoiceProfileManager

    try:
        mgr = VoiceProfileManager()
        profiles = mgr.list_profiles()
        return {"profiles": [p.to_dict() for p in profiles]}
    except Exception as exc:
        log.error("list_voice_profiles failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/voice-profiles")
async def api_create_voice_profile(request: Request):
    """Extract and save a voice profile for an instructor."""
    from voice_profiles.manager import VoiceProfileManager

    body = await _parse_json_body(request)
    video_path = body.get("video_path", "")
    instructor = body.get("instructor", "")
    start_sec = float(body.get("start_sec", 60))
    duration = float(body.get("duration", 15))

    if not video_path or not instructor:
        return JSONResponse(
            {"error": "Missing 'video_path' or 'instructor'"},
            status_code=400,
        )
    if not Path(video_path).exists():
        return JSONResponse({"error": "Video file not found"}, status_code=404)

    try:
        mgr = VoiceProfileManager()
        profile = mgr.extract_sample(
            Path(video_path), instructor,
            start_sec=start_sec, duration=duration,
        )
        return {"ok": True, "profile": profile.to_dict()}
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except Exception as exc:
        log.error("create_voice_profile failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.delete("/api/voice-profiles/{instructor}")
async def api_delete_voice_profile(instructor: str):
    """Delete a voice profile for an instructor."""
    from voice_profiles.manager import VoiceProfileManager

    try:
        mgr = VoiceProfileManager()
        deleted = mgr.delete_profile(instructor)
        if deleted:
            return {"ok": True, "message": f"Deleted profile for '{instructor}'"}
        return JSONResponse(
            {"error": f"No profile found for '{instructor}'"},
            status_code=404,
        )
    except Exception as exc:
        log.error("delete_voice_profile failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


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
# Media streaming (video/subtitles with Range support)
# ---------------------------------------------------------------------------

_MEDIA_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".srt": "application/x-subrip",
    ".vtt": "text/vtt",
}


def _resolve_media_path(path: str) -> Optional[Path]:
    """Resolve a user-supplied path, ensuring it stays under MEDIA_ROOT."""
    if not path:
        return None
    try:
        root = Path(MEDIA_ROOT).resolve()
        target = Path(path).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        return None
    if not target.is_file():
        return None
    return target


@app.get("/api/media")
async def api_media(path: str, request: Request):
    """Serve a media file (video/subtitle) with HTTP Range support for seek."""
    target = _resolve_media_path(path)
    if target is None:
        return JSONResponse({"error": "not found or outside MEDIA_ROOT"}, status_code=404)

    ext = target.suffix.lower()
    mime = _MEDIA_MIME.get(ext, "application/octet-stream")
    size = target.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")

    # Subtitles: serve whole file, convert SRT → VTT on the fly when asked.
    if ext in (".srt", ".vtt"):
        if ext == ".srt" and request.query_params.get("as") == "vtt":
            raw = target.read_text(encoding="utf-8", errors="replace")
            vtt = "WEBVTT\n\n" + raw.replace(",", ".")
            return Response(content=vtt, media_type="text/vtt")
        return FileResponse(path=str(target), media_type=mime)

    # No Range → full file.
    if not range_header:
        return FileResponse(
            path=str(target),
            media_type=mime,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        )

    # Parse "bytes=start-end"
    try:
        units, _, rng = range_header.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        if start < 0 or end >= size or start > end:
            raise ValueError
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    chunk_size = 1024 * 1024
    length = end - start + 1

    def _iter():
        with open(target, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        _iter(),
        status_code=206,
        media_type=mime,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


# ---------------------------------------------------------------------------
# OssFlow Export
# ---------------------------------------------------------------------------

@app.post("/api/export/ossflow")
async def api_export_to_ossflow(request: Request):
    """Export an instructional to the OssFlow backend."""
    from ossflow_client.client import OssFlowClient, OssFlowConfig

    body = await _parse_json_body(request)
    root_dir = body.get("path", "")
    instructor = body.get("instructor", "")
    base_url = body.get("base_url", "http://localhost:8080")

    if not root_dir or not instructor:
        return JSONResponse(
            {"error": "Missing 'path' or 'instructor'"},
            status_code=400,
        )
    if not Path(root_dir).exists():
        return JSONResponse({"error": "Path does not exist"}, status_code=404)

    try:
        config = OssFlowConfig(base_url=base_url)
        client = OssFlowClient(config)
        summary = client.export_full_instructional(Path(root_dir), instructor)
        return {"ok": True, "summary": summary}
    except Exception as exc:
        log.error("export_to_ossflow failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/export/ossflow/status")
async def api_ossflow_status():
    """Check if the OssFlow backend is reachable."""
    from ossflow_client.client import OssFlowClient

    try:
        client = OssFlowClient()
        reachable = client.health_check()
        return {"reachable": reachable}
    except Exception as exc:
        log.error("ossflow_status failed: %s", exc)
        return {"reachable": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Plex Export
# ---------------------------------------------------------------------------

@app.post("/api/export/plex")
async def api_export_plex(request: Request):
    """Export processed videos in Plex-compatible format."""
    from chapter_tools.plex_exporter import PlexExporter

    body = await _parse_json_body(request)
    instructional_name = body.get("name", "")
    chapters = body.get("chapters", [])
    source_dir = body.get("source_dir", "")
    output_dir = body.get("output_dir", "")

    if not instructional_name or not chapters or not source_dir or not output_dir:
        return JSONResponse(
            {"error": "Missing required fields: name, chapters, source_dir, output_dir"},
            status_code=400,
        )
    if not Path(source_dir).exists():
        return JSONResponse({"error": "source_dir does not exist"}, status_code=404)

    try:
        exporter = PlexExporter()
        exporter.export(instructional_name, chapters, Path(source_dir), Path(output_dir))
        return {
            "ok": True,
            "message": f"Exported '{instructional_name}' to {output_dir}",
        }
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        log.error("export_plex failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
