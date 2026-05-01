"""HTTP endpoints del servicio subtitle-generator.

Migrado de ``app.py`` en T31.7. El módulo expone una función
``register(app)`` que registra los endpoints en el ``FastAPI`` app
creado por ``ossflow_service_kit.create_app``.

Endpoints:

* ``POST /maintenance/clear-hf-locks`` — limpia locks HF cache.
* ``POST /maintenance/restart``         — self-restart graceful.
* ``POST /validate``                     — validate SRT quality.
* ``POST /regenerate-segment``           — re-transcribe un segmento.
* ``POST /apply-segment``                — escribe segmento + backup.
* ``POST /translate``                    — traduce un SRT.
* ``POST /analyze``                      — diagnóstico profundo.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


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
    # Output dubbing-adapted (iso-síncrono) en lugar de literal. Escribe
    # <stem>.dub.es.srt por default junto al literal.
    dubbing_mode: bool = False
    dubbing_cps: float = 16.0


class AnalyzeRequest(BaseModel):
    video_path: str
    language: str = "en"
    model: str = "large-v3"


def register(app: FastAPI) -> None:
    """Registra todos los endpoints en el FastAPI app."""

    @app.post("/maintenance/clear-hf-locks")
    def clear_hf_locks() -> dict:
        """Limpieza manual de locks de HuggingFace hub."""
        from subtitle_generator.shared.hf_cache import clear_hf_locks as _clear
        return _clear()

    @app.post("/maintenance/restart")
    def restart_service() -> dict:
        """Self-restart graceful: libera VRAM y reinicia el proceso.

        Docker reinicia el contenedor automáticamente
        (restart: unless-stopped). Útil para liberar VRAM tras un OOM
        o cancelación de job.
        """
        import os
        import signal
        import threading
        # Kill PID 1 (uvicorn reloader) para que Docker reinicie.
        # Killing solo os.getpid() (worker) deja al reloader sin worker.
        threading.Timer(0.5, lambda: os.kill(1, signal.SIGTERM)).start()
        return {"ok": True, "message": "Reiniciando subtitle-generator…"}

    @app.post("/validate")
    def validate(req: ValidateRequest) -> dict:
        from subtitle_generator.config import SubtitleConfig
        from subtitle_generator.quality_checker import SubtitleQualityChecker
        from subtitle_generator.segment_regen import find_sibling_video
        from subtitle_generator.srt_io import parse_srt

        srt = Path(req.srt_path)
        if not srt.exists():
            # Fallback: probar swapping .srt <-> .en.srt (migration de convención).
            alt = None
            if srt.name.endswith(".en.srt"):
                alt = srt.with_name(srt.name[: -len(".en.srt")] + ".srt")
            elif srt.suffix == ".srt":
                alt = srt.with_name(srt.stem + ".en.srt")
            if alt and alt.exists():
                srt = alt
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"SRT no existe: {req.srt_path}",
                )
        subs = parse_srt(srt)
        report = SubtitleQualityChecker(SubtitleConfig()).check(subs)
        sibling = find_sibling_video(srt)
        report["video_path"] = str(sibling) if sibling else None
        report["srt_path"] = str(srt)
        return report

    @app.post("/regenerate-segment")
    def regenerate_segment(req: RegenerateRequest) -> dict:
        from subtitle_generator.core.regenerator import get_regenerator
        from subtitle_generator.cuda_setup import (
            setup_nvidia_dlls,
            setup_pytorch_safety,
        )
        from subtitle_generator.segment_regen import find_sibling_video
        from subtitle_generator.srt_io import parse_srt

        srt = Path(req.srt_path)
        if not srt.exists():
            raise HTTPException(
                status_code=404, detail=f"SRT no existe: {req.srt_path}",
            )
        subs = parse_srt(srt)
        if req.segment_idx < 1 or req.segment_idx > len(subs):
            raise HTTPException(
                status_code=400, detail="segment_idx fuera de rango",
            )

        video = Path(req.video_path) if req.video_path else find_sibling_video(srt)
        if not video or not video.exists():
            raise HTTPException(
                status_code=400,
                detail="No se encontró video hermano del SRT (pasa video_path)",
            )

        target = subs[req.segment_idx - 1]
        setup_nvidia_dlls()
        setup_pytorch_safety()
        regen = get_regenerator(req.model, req.language)
        try:
            result = regen.regenerate(
                video, target["start"], target["end"], req.context_seconds,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Regeneración falló: {exc}",
            )

        # Clamp timestamps para que no solapen con vecinos.
        idx = req.segment_idx - 1
        prev_end = subs[idx - 1]["end"] if idx > 0 else 0.0
        next_start = subs[idx + 1]["start"] if idx + 1 < len(subs) else float("inf")
        result["start"] = max(result["start"], prev_end)
        result["end"] = min(result["end"], next_start)

        return {
            "segment_idx": req.segment_idx,
            "old": {
                "text": target["text"],
                "start": target["start"],
                "end": target["end"],
            },
            "new": result,
            "video_path": str(video),
        }

    @app.post("/apply-segment")
    def apply_segment(req: ApplyRequest) -> dict:
        from subtitle_generator.srt_io import parse_srt, write_srt_with_backup

        srt = Path(req.srt_path)
        if not srt.exists():
            raise HTTPException(
                status_code=404, detail=f"SRT no existe: {req.srt_path}",
            )
        subs = parse_srt(srt)
        if req.segment_idx < 1 or req.segment_idx > len(subs):
            raise HTTPException(
                status_code=400, detail="segment_idx fuera de rango",
            )

        target = subs[req.segment_idx - 1]
        target["text"] = req.text.strip()
        if req.start is not None:
            target["start"] = float(req.start)
        if req.end is not None:
            target["end"] = float(req.end)

        write_srt_with_backup(srt, subs)
        return {
            "ok": True,
            "segment_idx": req.segment_idx,
            "backup": str(srt) + ".bak",
        }

    @app.post("/translate")
    def translate(req: TranslateRequest) -> dict:
        from subtitle_generator.core.translate_runner import (
            build_translator_with_fallback,
            translate_for_dubbing,
        )
        from subtitle_generator.shared.paths import literal_srt_path_for

        srt = Path(req.srt_path)
        if not srt.exists():
            raise HTTPException(
                status_code=404, detail=f"SRT no existe: {req.srt_path}",
            )

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
            primary, fallback = build_translator_with_fallback(opts)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if req.dubbing_mode:
            # Budget-aware adaptation escrita al literal .es.srt — subs y
            # dubbing comparten una única source of truth. Nivel 3 (speech-
            # anchored) cuando hay .words.json, nivel 2 (SRT slots) si no.
            out = Path(req.out_path) if req.out_path else literal_srt_path_for(srt)
            try:
                translate_for_dubbing(
                    primary, srt, out, req.dubbing_cps, force_slot_mode=False,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"{req.provider} (dubbing): {exc}",
                )
            return {
                "ok": True,
                "out_path": str(out),
                "source": str(srt),
                "provider": req.provider,
                "mode": "dubbing",
            }

        # Default out usa video-stem: foo.en.srt -> foo.es.srt (sin .en).
        out = Path(req.out_path) if req.out_path else literal_srt_path_for(srt)
        used = req.provider
        try:
            written = primary.translate_srt(srt, out)
        except Exception as exc:
            if fallback is None:
                raise HTTPException(
                    status_code=502, detail=f"{req.provider}: {exc}",
                )
            try:
                written = fallback.translate_srt(srt, out)
                used = req.fallback_provider or "fallback"
            except Exception as exc2:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"{req.provider}: {exc} | "
                        f"fallback {req.fallback_provider}: {exc2}"
                    ),
                )
        return {
            "ok": True,
            "out_path": str(written),
            "source": str(srt),
            "provider": used,
        }

    @app.post("/analyze")
    def analyze_video(req: AnalyzeRequest) -> dict:
        """Análisis diagnóstico profundo (energía/VAD/transcripción/gaps)."""
        from subtitle_generator.core.analyzer import analyze_video as _analyze
        return _analyze(req)
