"""HTTP endpoints del servicio dubbing-generator.

Migrado de ``app.py`` en T32.7. El módulo expone ``register(app)``
que registra los endpoints en el FastAPI app creado por
``ossflow_service_kit.create_app``.

Endpoints:

* ``POST /maintenance/restart``           — self-restart graceful.
* ``GET  /s2pro/status``                  — estado del child process s2.cpp.
* ``GET  /voices``                        — lista de voice profiles disponibles.
* ``PUT  /voices/{filename}/transcript``  — guarda transcripción de referencia.
* ``POST /analyze``                       — diagnóstico del pipeline para un vídeo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


VOICES_DIR = Path("/voices")


def _transcript_path_for(voice_path: Path) -> Path:
    """Sidecar ``.txt`` con la transcripción de referencia (mismo stem)."""
    return voice_path.with_suffix(".txt")


def _read_transcript(voice_path: Path) -> str:
    sidecar = _transcript_path_for(voice_path)
    if not sidecar.exists():
        return ""
    try:
        return sidecar.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


class _TranscriptBody(BaseModel):
    transcript: str


class AnalyzeRequest(BaseModel):
    video_path: str
    srt_path: Optional[str] = None
    synthesize: bool = False
    max_phrases: Optional[int] = None
    voice_profile: Optional[str] = None
    model_voice_path: Optional[str] = None


def _resolve_srt_for(video_path: Path) -> Optional[Path]:
    """Localiza el SRT español para un vídeo dado (literal o dub)."""
    base = video_path.with_suffix("")
    stem = base.name
    for dub_sfx in ("_DOBLADO",):
        if stem.endswith(dub_sfx):
            stem = stem[: -len(dub_sfx)]
            base = base.parent / stem
            break
    for suffix in (".es.srt", ".ES.srt", "_ES.srt", "_ESP.srt", ".dub.es.srt"):
        candidate = base.parent / f"{base.name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def register(app: FastAPI) -> None:
    """Registra todos los endpoints en el FastAPI app."""

    @app.post("/maintenance/restart")
    def restart_service() -> dict:
        """Self-restart graceful: libera VRAM y reinicia el proceso.

        Docker reinicia el contenedor automáticamente
        (restart: unless-stopped). El s2.cpp server hijo sigue al PID 1
        al recibir SIGTERM.
        """
        import os
        import signal
        import threading
        threading.Timer(0.5, lambda: os.kill(1, signal.SIGTERM)).start()
        return {"ok": True, "message": "Reiniciando dubbing-generator…"}

    @app.get("/s2pro/status")
    def s2pro_status() -> dict:
        """Estado del subproceso S2-Pro.

        Con lazy-load el manager solo existe durante un job activo;
        idle GET devuelve ``{"running":false,"ready":false,"engine":"idle"}``.
        """
        manager = getattr(app.state, "s2pro_manager", None)
        if manager is None:
            return {"running": False, "ready": False, "engine": "idle"}
        proc = manager.process
        return {
            "running": proc is not None and proc.poll() is None,
            "ready": manager.is_ready(),
            "engine": manager.cfg.tts_engine,
        }

    @app.get("/voices")
    def list_voices() -> dict:
        """Lista WAVs bajo ``/voices`` como voice profiles seleccionables.

        Operadores dropean ficheros en ``dubbing-generator/voices/``
        (mounted a ``/voices`` en el container). Cada voz puede tener
        ``<stem>.txt`` con su transcripción de referencia.
        """
        voices: list[dict] = []
        if VOICES_DIR.exists():
            for p in sorted(VOICES_DIR.iterdir()):
                if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".mp3"):
                    voices.append({
                        "id": p.name,
                        "path": str(p),
                        "size_bytes": p.stat().st_size,
                        "transcript": _read_transcript(p),
                    })
        return {"voices": voices}

    @app.put("/voices/{filename}/transcript")
    def save_voice_transcript(filename: str, body: _TranscriptBody) -> dict:
        """Persiste la transcripción de referencia de una voz."""
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="invalid filename")
        voice_path = VOICES_DIR / filename
        if not voice_path.exists() or not voice_path.is_file():
            raise HTTPException(status_code=404, detail="voice not found")
        sidecar = _transcript_path_for(voice_path)
        try:
            sidecar.write_text(
                (body.transcript or "").strip() + "\n", encoding="utf-8",
            )
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Cannot write transcript to {sidecar}: {exc}. "
                    "Check that /voices is mounted read/write in docker-compose."
                ),
            ) from exc
        return {"ok": True, "voice": filename, "transcript_path": str(sidecar)}

    @app.post("/analyze")
    def analyze_dubbing(req: AnalyzeRequest) -> dict:
        """Análisis diagnóstico del pipeline de dubbing para un vídeo.

        Devuelve SRT blocks con duraciones y gaps, planned slots
        (aligner output con borrowed gap time), métricas de density,
        TTS fit report (estimado vs slot) y opcionalmente síntesis
        real con duraciones medidas (slow path).
        """
        from dubbing_generator.config import DubbingConfig
        from dubbing_generator.pipeline import parse_srt
        from dubbing_generator.sync.aligner import SyncAligner
        from dubbing_generator.sync.drift_corrector import DriftCorrector

        vp = Path(req.video_path)
        if not vp.exists():
            raise HTTPException(
                status_code=404, detail=f"Video not found: {req.video_path}",
            )

        srt_p = Path(req.srt_path) if req.srt_path else _resolve_srt_for(vp)
        if srt_p is None or not srt_p.exists():
            raise HTTPException(
                status_code=404, detail=f"SRT not found for {vp.name}",
            )

        model_voice_path = req.model_voice_path or ""
        if not model_voice_path and req.voice_profile:
            candidate = Path("/voices") / req.voice_profile
            if candidate.exists():
                model_voice_path = str(candidate)
        if not model_voice_path:
            model_voice_path = os.environ.get("DUBBING_MODEL_VOICE_PATH") or ""

        cfg = DubbingConfig(
            use_model_voice=bool(
                model_voice_path and Path(model_voice_path).exists()
            ),
            model_voice_path=model_voice_path,
        )
        blocks = parse_srt(srt_p)
        aligner = SyncAligner(cfg)
        planned = aligner.plan(blocks)
        drift = DriftCorrector(cfg)
        drift.reset()

        srt_blocks = [
            {
                "idx": b.index,
                "start_ms": b.start_ms,
                "end_ms": b.end_ms,
                "duration_ms": b.duration_ms,
                "chars": len(b.text),
                "text": b.text,
            }
            for b in blocks
        ]

        srt_gaps = []
        for i in range(len(blocks) - 1):
            gap = blocks[i + 1].start_ms - blocks[i].end_ms
            if gap > 0:
                srt_gaps.append({
                    "after_idx": blocks[i].index,
                    "gap_ms": gap,
                    "borrowed_by_previous": max(0, gap - cfg.inter_phrase_pad_ms),
                })

        planned_rows = []
        for i, p in enumerate(planned):
            density = len(p.text) / max(p.allocated_ms, 1)
            pressure = density / drift.DENSITY_BASE if drift.DENSITY_BASE > 0 else 1.0
            est_tts_ms = len(p.text) * cfg.avg_ms_per_char
            compression_needed = est_tts_ms / max(p.allocated_ms, 1)
            planned_rows.append({
                "idx": i,
                "text": p.text[:80],
                "target_start_ms": p.target_start_ms,
                "allocated_ms": p.allocated_ms,
                "chars": len(p.text),
                "density": round(density, 5),
                "pressure": round(pressure, 2),
                "est_tts_ms": int(est_tts_ms),
                "compression_needed": round(compression_needed, 2),
                "will_overflow": compression_needed > cfg.max_compression_ratio,
            })

        overflow_count = sum(1 for r in planned_rows if r["will_overflow"])
        compression_vals = [
            r["compression_needed"]
            for r in planned_rows
            if r["compression_needed"] > 0
        ]
        summary = {
            "total_phrases": len(planned),
            "total_chars": sum(len(p.text) for p in planned),
            "total_allocated_ms": sum(p.allocated_ms for p in planned),
            "srt_duration_ms": blocks[-1].end_ms if blocks else 0,
            "will_overflow_count": overflow_count,
            "max_compression": round(max(compression_vals), 2) if compression_vals else 0,
            "avg_compression": (
                round(sum(compression_vals) / len(compression_vals), 2)
                if compression_vals else 0
            ),
            "config": {
                "tts_speed": cfg.tts_speed,
                "max_compression_ratio": cfg.max_compression_ratio,
                "min_phrase_duration_ms": cfg.min_phrase_duration_ms,
                "max_overflow_ms": cfg.max_overflow_ms,
                "inter_phrase_pad_ms": cfg.inter_phrase_pad_ms,
                "speed_min": cfg.speed_min,
                "speed_max": cfg.speed_max,
                "ducking_bg_volume": cfg.ducking_bg_volume,
                "ducking_fg_volume": cfg.ducking_fg_volume,
            },
        }

        result = {
            "video_path": str(vp),
            "srt_path": str(srt_p),
            "summary": summary,
            "srt_blocks": srt_blocks,
            "srt_gaps": srt_gaps,
            "planned": planned_rows,
            "synthesis": None,
        }

        if req.synthesize:
            result["synthesis"] = _run_synthesis_probe(
                vp, planned, cfg, max_phrases=req.max_phrases,
            )

        return result


def _run_synthesis_probe(
    video_path: Path, planned: list, cfg, max_phrases=None,
) -> dict:
    """Corre TTS en N frases y reporta duraciones reales.

    Devuelve per-phrase actual tts_ms, post-stretch ms,
    overflow/underflow, final placement después de overlap resolution.
    """
    from dubbing_generator.audio.separator import AudioSeparator
    from dubbing_generator.audio.stretcher import stretch_audio
    from dubbing_generator.sync.drift_corrector import DriftCorrector
    from dubbing_generator.tts import build_synthesizer
    from dubbing_generator.tts.voice_cloner import VoiceCloner

    separator = AudioSeparator(cfg)
    cloner = VoiceCloner(cfg)
    synth = build_synthesizer(cfg)
    drift = DriftCorrector(cfg)
    drift.reset()

    separator.separate(video_path)
    vocals_stem = video_path.with_name(f"{video_path.stem}_VOCALS.wav")
    ref_wav = cloner.get_reference(
        video_path, vocals_stem if vocals_stem.exists() else None,
    )

    phrases = planned[:max_phrases] if max_phrases else planned
    rows = []

    for i, block in enumerate(phrases):
        if not block.text or len(block.text) < 2:
            continue
        density = len(block.text) / max(block.allocated_ms, 1)
        speed = drift.check_density(i, density)

        try:
            raw = synth.generate(block.text, ref_wav, speed=speed)
            raw_ms = len(raw)

            target_ms = block.allocated_ms + cfg.max_overflow_ms
            fitted = stretch_audio(
                raw, target_duration_ms=target_ms,
                max_ratio=cfg.max_compression_ratio,
                min_ratio=cfg.min_compression_ratio,
            )
            fitted_ms = len(fitted)

            rows.append({
                "idx": i,
                "text": block.text[:80],
                "start_ms": block.target_start_ms,
                "allocated_ms": block.allocated_ms,
                "raw_tts_ms": raw_ms,
                "fitted_ms": fitted_ms,
                "speed_used": round(speed, 3),
                "stretch_ratio": round(raw_ms / max(fitted_ms, 1), 3),
                "overflow_vs_slot_ms": fitted_ms - block.allocated_ms,
                "end_ms": block.target_start_ms + fitted_ms,
            })
        except Exception as exc:
            rows.append({
                "idx": i, "text": block.text[:80], "error": str(exc),
            })

    overlaps = []
    valid = [r for r in rows if "error" not in r]
    for i in range(len(valid) - 1):
        cur, nxt = valid[i], valid[i + 1]
        if cur["end_ms"] > nxt["start_ms"]:
            overlaps.append({
                "between": [cur["idx"], nxt["idx"]],
                "overlap_ms": cur["end_ms"] - nxt["start_ms"],
            })

    return {
        "phrases": rows,
        "overlaps": overlaps,
        "ref_wav": str(ref_wav),
    }
