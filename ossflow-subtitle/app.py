"""FastAPI entrypoint for subtitle-generator backend.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_KIT_PARENT = Path(__file__).resolve().parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))

from ossflow_service_kit import JobEvent, RunRequest, create_app, emit_logs  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402


SERVICE_NAME = "subtitle-generator"


class ValidateRequest(BaseModel):
    srt_path: str


class RegenerateRequest(BaseModel):
    srt_path: str
    segment_idx: int
    context_seconds: float = 1.0
    video_path: str | None = None
    model: str = "large-v3"
    language: str = "en"


class ApplyRequest(BaseModel):
    srt_path: str
    segment_idx: int
    text: str
    start: float | None = None
    end: float | None = None


class TranslateRequest(BaseModel):
    srt_path: str
    target_lang: str = "ES"
    source_lang: str = "EN"
    provider: str = "ollama"
    model: str | None = None
    formality: str | None = None
    api_key: str | None = None
    fallback_provider: str | None = None
    fallback_api_key: str | None = None
    out_path: str | None = None
    # Dubbing-adapted (iso-synchronous) output instead of literal. Writes
    # <stem>.dub.es.srt by default so it lives alongside the literal subs.
    dubbing_mode: bool = False
    dubbing_cps: float = 16.0


class AnalyzeRequest(BaseModel):
    video_path: str
    language: str = "en"
    model: str = "large-v3"


# Regenerator — migrado a subtitle_generator/core/regenerator.py (T31.2).
# El global ``_regenerator`` ha sido eliminado a favor de ``lru_cache`` por
# (model, language). Esto permite tests aislados (cache_clear) y evita el
# anti-patrón Singleton module-level.
from subtitle_generator.core.regenerator import (  # noqa: F401,E402
    get_regenerator as _get_regenerator,
)


# Path helpers — migrados a subtitle_generator/shared/paths.py (T31.1).
from subtitle_generator.shared.paths import (  # noqa: F401,E402
    clean_base_stem as _clean_base_stem,
    dub_srt_path_for as _dub_srt_path_for,
    literal_srt_path_for as _literal_srt_path_for,
    resolve_input as _resolve_input,
    words_json_for as _words_json_for,
)


# Transcriber bridge — migrado a subtitle_generator/core/transcriber.py (T31.5).
from subtitle_generator.core.transcriber import (  # noqa: E402,F401
    run_subtitle_generator as _run_subtitle_generator,
)


# Translate runner — migrado a subtitle_generator/core/translate_runner.py (T31.4).
from subtitle_generator.core.translate_runner import (  # noqa: E402,F401
    build_translator_with_fallback as _build_translator_with_fallback,
    run_translate_directory as _run_translate_directory,
    translate_for_dubbing as _translate_for_dubbing,
    translate_for_dubbing_nivel3 as _translate_for_dubbing_nivel3,
)


# HF cache helpers — migrados a subtitle_generator/shared/hf_cache.py (T31.5).
from subtitle_generator.shared.hf_cache import (  # noqa: E402,F401
    clear_hf_locks as _clear_hf_locks,
    hf_cache_root as _hf_cache_root,
)


# Best-effort cleanup at import/startup — frees locks left by a killed worker.
try:
    _startup_clean = _clear_hf_locks()
    if _startup_clean["removed"]:
        logging.getLogger(SERVICE_NAME).warning(
            "cleared %d stale HF lock(s) at startup", _startup_clean["removed"]
        )
except Exception as _exc:  # noqa: BLE001
    logging.getLogger(SERVICE_NAME).warning("HF lock cleanup at startup failed: %s", _exc)


app = create_app(service_name=SERVICE_NAME, task_fn=_run_subtitle_generator)


@app.post("/maintenance/clear-hf-locks")
def clear_hf_locks() -> dict:
    """Manually clear stale HuggingFace hub locks."""
    return _clear_hf_locks()


@app.post("/maintenance/restart")
def restart_service() -> dict:
    """Graceful self-restart: libera VRAM y reinicia el proceso.

    Docker reinicia el contenedor automáticamente (restart: unless-stopped).
    Útil para liberar VRAM tras un OOM o cancelación de job.
    """
    import threading, os, signal
    # Kill PID 1 (uvicorn reloader / entrypoint) so Docker restarts the container.
    # Killing only os.getpid() (the worker) leaves the reloader alive without a worker.
    threading.Timer(0.5, lambda: os.kill(1, signal.SIGTERM)).start()
    return {"ok": True, "message": "Reiniciando subtitle-generator…"}


@app.post("/validate")
def validate(req: ValidateRequest) -> dict:
    from pathlib import Path as _P
    from subtitle_generator.config import SubtitleConfig  # type: ignore
    from subtitle_generator.quality_checker import SubtitleQualityChecker  # type: ignore
    from subtitle_generator.segment_regen import find_sibling_video  # type: ignore
    from subtitle_generator.srt_io import parse_srt  # type: ignore

    srt = _P(req.srt_path)
    if not srt.exists():
        # Fallback: try swapping .srt <-> .en.srt (convention migration)
        alt = None
        if srt.name.endswith(".en.srt"):
            alt = srt.with_name(srt.name[:-len(".en.srt")] + ".srt")
        elif srt.suffix == ".srt":
            alt = srt.with_name(srt.stem + ".en.srt")
        if alt and alt.exists():
            srt = alt
        else:
            raise HTTPException(status_code=404, detail=f"SRT no existe: {req.srt_path}")
    subs = parse_srt(srt)
    report = SubtitleQualityChecker(SubtitleConfig()).check(subs)
    sibling = find_sibling_video(srt)
    report["video_path"] = str(sibling) if sibling else None
    report["srt_path"] = str(srt)
    return report


@app.post("/regenerate-segment")
def regenerate_segment(req: RegenerateRequest) -> dict:
    from pathlib import Path as _P
    from subtitle_generator.cuda_setup import setup_nvidia_dlls, setup_pytorch_safety  # type: ignore
    from subtitle_generator.segment_regen import find_sibling_video  # type: ignore
    from subtitle_generator.srt_io import parse_srt  # type: ignore

    srt = _P(req.srt_path)
    if not srt.exists():
        raise HTTPException(status_code=404, detail=f"SRT no existe: {req.srt_path}")
    subs = parse_srt(srt)
    if req.segment_idx < 1 or req.segment_idx > len(subs):
        raise HTTPException(status_code=400, detail="segment_idx fuera de rango")

    video = _P(req.video_path) if req.video_path else find_sibling_video(srt)
    if not video or not video.exists():
        raise HTTPException(
            status_code=400,
            detail="No se encontró video hermano del SRT (pasa video_path)",
        )

    target = subs[req.segment_idx - 1]
    setup_nvidia_dlls()
    setup_pytorch_safety()
    regen = _get_regenerator(req.model, req.language)
    try:
        result = regen.regenerate(
            video, target["start"], target["end"], req.context_seconds
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Regeneración falló: {exc}")

    # Clamp timestamps so they don't overlap with neighbours
    idx = req.segment_idx - 1
    prev_end = subs[idx - 1]["end"] if idx > 0 else 0.0
    next_start = subs[idx + 1]["start"] if idx + 1 < len(subs) else float("inf")
    result["start"] = max(result["start"], prev_end)
    result["end"] = min(result["end"], next_start)

    return {
        "segment_idx": req.segment_idx,
        "old": {"text": target["text"], "start": target["start"], "end": target["end"]},
        "new": result,
        "video_path": str(video),
    }


@app.post("/apply-segment")
def apply_segment(req: ApplyRequest) -> dict:
    from pathlib import Path as _P
    from subtitle_generator.srt_io import parse_srt, write_srt_with_backup  # type: ignore

    srt = _P(req.srt_path)
    if not srt.exists():
        raise HTTPException(status_code=404, detail=f"SRT no existe: {req.srt_path}")
    subs = parse_srt(srt)
    if req.segment_idx < 1 or req.segment_idx > len(subs):
        raise HTTPException(status_code=400, detail="segment_idx fuera de rango")

    target = subs[req.segment_idx - 1]
    target["text"] = req.text.strip()
    if req.start is not None:
        target["start"] = float(req.start)
    if req.end is not None:
        target["end"] = float(req.end)

    write_srt_with_backup(srt, subs)
    return {"ok": True, "segment_idx": req.segment_idx, "backup": str(srt) + ".bak"}


@app.post("/translate")
def translate(req: TranslateRequest) -> dict:
    from pathlib import Path as _P

    srt = _P(req.srt_path)
    if not srt.exists():
        raise HTTPException(status_code=404, detail=f"SRT no existe: {req.srt_path}")

    opts = {
        "provider": req.provider,
        "model": req.model,
        "api_key": req.api_key,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
        "formality": req.formality,
        "fallback_provider": req.fallback_provider,
        "fallback_api_key": req.fallback_api_key,
    }
    try:
        primary, fallback = _build_translator_with_fallback(opts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if req.dubbing_mode:
        # Budget-aware adaptation written to the literal .es.srt so subs and
        # dubbing share a single source of truth. Level 3 (speech-anchored)
        # when a .words.json exists, level 2 (SRT slots) otherwise. The level 3
        # timestamps re-align to real speech pauses, so the dub reads fluently
        # while staying within ~1-2 s of the original video cues.
        out = _P(req.out_path) if req.out_path else _literal_srt_path_for(srt)
        try:
            _translate_for_dubbing(primary, srt, out, req.dubbing_cps, force_slot_mode=False)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{req.provider} (dubbing): {exc}")
        return {
            "ok": True,
            "out_path": str(out),
            "source": str(srt),
            "provider": req.provider,
            "mode": "dubbing",
        }

    # Default out uses video-level stem so foo.en.srt -> foo.es.srt (no .en).
    out = _P(req.out_path) if req.out_path else _literal_srt_path_for(srt)
    used = req.provider
    try:
        written = primary.translate_srt(srt, out)
    except Exception as exc:
        if fallback is None:
            raise HTTPException(status_code=502, detail=f"{req.provider}: {exc}")
        try:
            written = fallback.translate_srt(srt, out)
            used = req.fallback_provider or "fallback"
        except Exception as exc2:
            raise HTTPException(
                status_code=502,
                detail=f"{req.provider}: {exc} | fallback {req.fallback_provider}: {exc2}",
            )
    return {"ok": True, "out_path": str(written), "source": str(srt), "provider": used}


@app.post("/analyze")
def analyze_video(req: AnalyzeRequest) -> dict:
    """Análisis diagnóstico profundo. Lógica migrada a core/analyzer (T31.3)."""
    from subtitle_generator.core.analyzer import analyze_video as _analyze_video
    return _analyze_video(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
