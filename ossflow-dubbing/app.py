"""FastAPI entrypoint for dubbing-generator backend.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8003
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_KIT_PARENT = Path(__file__).resolve().parent.parent
if str(_KIT_PARENT) not in sys.path:
    sys.path.insert(0, str(_KIT_PARENT))

from fastapi import HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from bjj_service_kit import JobEvent, RunRequest, create_app, emit_logs  # noqa: E402


SERVICE_NAME = "dubbing-generator"


def _resolve_input(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _resolve_srt_for(video_path: Path) -> Optional[Path]:
    base = video_path.with_suffix("")
    # If caller passed the dubbed output (<name>_DOBLADO.mkv), walk back
    # to the source video stem — SRTs live next to the original, not the
    # remux. Lets the UI pass video.path directly for debug calls.
    stem = base.name
    for dub_sfx in ("_DOBLADO",):
        if stem.endswith(dub_sfx):
            stem = stem[: -len(dub_sfx)]
            base = base.parent / stem
            break
    # Híbrido: .es.srt literal (alineado con vídeo) preferido. El pipeline
    # hace merge de líneas consecutivas para prosodia continua manteniendo
    # anclaje al timestamp del vídeo. .dub.es.srt queda como fallback.
    for suffix in (".es.srt", ".ES.srt", "_ES.srt", "_ESP.srt", ".dub.es.srt"):
        candidate = base.parent / f"{base.name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _run_dubbing_generator(req: RunRequest, emit) -> None:
    """Bridge RunRequest -> dubbing_generator.pipeline.DubbingPipeline."""
    input_path = _resolve_input(Path(req.input_path))

    opts = req.options or {}
    level = logging.DEBUG if opts.get("verbose") else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    with emit_logs(emit, level=level):
        import os
        from dubbing_generator.config import DubbingConfig  # type: ignore
        from dubbing_generator.pipeline import DubbingPipeline  # type: ignore

        tts_engine = (
            opts.get("tts_engine")
            or os.environ.get("DUBBING_TTS_ENGINE")
            or "s2pro"
        ).strip().lower()

        # Voice profile resolution applies only to engines that actually
        # consume model_voice_path (XTTS-style cloning, ElevenLabs cloning).
        # S2-Pro reads its reference exclusively from cfg.s2_ref_audio_path
        # — running the resolver would emit "voice_profile not found" logs
        # for a value the synth never reads.
        #
        # Priority for non-s2pro engines:
        #   1. opts.model_voice_path — explicit absolute path from caller
        #   2. opts.voice_profile    — filename like "narrador_es.wav"
        #      resolved inside /voices (what the UI selector sends)
        #   3. DUBBING_MODEL_VOICE_PATH env var — global default
        # Falls back to cloning the instructor's own voice if none apply.
        model_voice_path = ""
        use_model_voice = False
        if tts_engine != "s2pro":
            model_voice_path = opts.get("model_voice_path") or ""
            voice_profile = opts.get("voice_profile") or ""
            if not model_voice_path and voice_profile:
                candidate = Path("/voices") / voice_profile
                if candidate.exists():
                    model_voice_path = str(candidate)
                else:
                    emit(JobEvent(type="log", data={
                        "message": f"voice_profile '{voice_profile}' not found in /voices — cloning instructor"
                    }))
            if not model_voice_path:
                model_voice_path = os.environ.get("DUBBING_MODEL_VOICE_PATH") or ""

            use_model_voice = bool(opts.get("use_model_voice"))
            if not use_model_voice and model_voice_path:
                use_model_voice = Path(model_voice_path).exists()
        if tts_engine not in ("s2pro", "elevenlabs", "piper", "kokoro"):
            emit(JobEvent(type="log", data={
                "message": f"unsupported tts_engine={tts_engine!r}, using 's2pro'"
            }))
            tts_engine = "s2pro"

        config_kwargs = dict(
            use_model_voice=use_model_voice,
            model_voice_path=model_voice_path,
            tts_engine=tts_engine,
        )
        if tts_engine == "elevenlabs":
            if voice_id := opts.get("elevenlabs_voice_id"):
                config_kwargs["elevenlabs_voice_id"] = str(voice_id)
            if model_id := opts.get("elevenlabs_model_id"):
                config_kwargs["elevenlabs_model_id"] = str(model_id)
            # ElevenLabs produces much cleaner ES than XTTS — no language
            # hallucination risk, stable prosody per render. Two tunings
            # that would be unsafe with XTTS pay off here:
            #   - merge larger blocks → fewer renders → more continuous
            #     prosody between what were separate SRT slots.
            #   - longer inter-phrase crossfades → masks the small
            #     timbre jumps that still exist between cloud renders.
            # Rationale for the specific values comes from the S01E01
            # QA: 5 hard_cuts + 5 warnings on 14 boundaries were caused
            # almost entirely by timbre/F0 jumps and short-gap cuts.
            # Wider merge caps. ElevenLabs handles 500-char prompts
            # fluently and the bigger caps mean fewer boundaries — the
            # S01E01 QA showed that most hard_cuts sat exactly at
            # boundaries the merger couldn't combine because of char or
            # gap limits. 1000 ms gap covers the natural pause at end
            # of sentence when the instructor takes a breath.
            config_kwargs.setdefault("merge_max_chars", 500)
            config_kwargs.setdefault("merge_max_gap_ms", 1000)
            config_kwargs.setdefault("inter_phrase_crossfade_ms", 250)
            config_kwargs.setdefault("inter_phrase_crossfade_max_gap_ms", 500)
            config_kwargs.setdefault("force_crossfade_ms", 420)
            # Tier 3: kept at the same value as force_crossfade_ms.
            # A first iteration set this to 550 ms — enough to mask a
            # 10 dB jump perceptually but the longer fade-out made the
            # next phrase's SRT-anchored start collide with leftover
            # audio, introducing a 1187 ms silence on a *different*
            # boundary. The pairwise RMS lift alone already cuts the
            # raw dB jump (10 → 8 on S01E01) and the standard force
            # crossfade is enough to hide that residual.
            config_kwargs.setdefault("rms_jump_crossfade_ms", 420)
        elif tts_engine == "piper":
            # Piper is local, deterministic, no hallucination risk. Cadence
            # is uniform per render so smaller merges are fine, and short
            # crossfades suffice (no timbre jumps to mask). Castellanize
            # already runs in the pipeline so EN BJJ terms reach Piper as
            # ES phonetic spelling.
            if model_id := opts.get("piper_model_path"):
                config_kwargs["piper_model_path"] = str(model_id)
            config_kwargs.setdefault("merge_max_chars", 300)
            config_kwargs.setdefault("merge_max_gap_ms", 600)
            config_kwargs.setdefault("inter_phrase_crossfade_ms", 80)
            config_kwargs.setdefault("force_crossfade_ms", 200)
            config_kwargs.setdefault("rms_jump_crossfade_ms", 0)
        elif tts_engine == "kokoro":
            # Kokoro StyleTTS2: prosodia más natural que Piper, voz preset
            # ES masculina (em_alex/em_santa). Output 24 kHz nativo. Cadencia
            # uniforme por render, sin alucinaciones cross-language.
            if voice := opts.get("kokoro_voice"):
                config_kwargs["kokoro_voice"] = str(voice)
            config_kwargs.setdefault("merge_max_chars", 300)
            config_kwargs.setdefault("merge_max_gap_ms", 600)
            config_kwargs.setdefault("inter_phrase_crossfade_ms", 100)
            config_kwargs.setdefault("force_crossfade_ms", 250)
        elif tts_engine == "s2pro":
            # S2-Pro local voice clone. Same model + same ref WAV across the
            # episode → timbre-stable across renders, so no long crossfade
            # masks needed. merge_max_chars=300 (same as Kokoro) keeps phrases
            # well under s2.cpp's 800-token degrade threshold while avoiding
            # fragmentation that would inflate wall-clock time per episode
            # (~120 s/phrase on the 2060).
            for k in ("s2_ref_audio_path", "s2_ref_text",
                      "s2_temperature", "s2_top_p", "s2_top_k",
                      "s2_max_tokens"):
                val = opts.get(k)
                if val is not None:
                    config_kwargs[k] = val
            config_kwargs.setdefault("merge_max_chars", 300)
            config_kwargs.setdefault("merge_max_gap_ms", 600)
            config_kwargs.setdefault("inter_phrase_crossfade_ms", 100)
            config_kwargs.setdefault("force_crossfade_ms", 250)
            config_kwargs.setdefault("rms_jump_crossfade_ms", 0)

        config = DubbingConfig(**config_kwargs)

        emit(JobEvent(type="log", data={"message": f"starting dubbing-generator on {input_path}"}))
        force = bool(opts.get("force"))

        def _remove_existing_output(video_path: Path) -> None:
            """Drop <Season>/doblajes/<name>.mkv so the pipeline regenerates it."""
            candidate = video_path.parent / "doblajes" / f"{video_path.stem}.mkv"
            if candidate.exists():
                try:
                    candidate.unlink()
                    emit(JobEvent(type="log", data={"message":
                        f"force overwrite: removed existing {candidate.name}"
                    }))
                except OSError as exc:
                    emit(JobEvent(type="log", data={"message":
                        f"ERROR removing {candidate.name}: {exc}"
                    }))

        # S2-Pro lazy-load: the s2.cpp server holds ~5 GB VRAM (Q6_K) once
        # the GGUF is mmap'd. On a 6 GB card (RTX 2060) that leaves no room
        # for Demucs (fase 0/6, ~2 GB pico) or the voice cloner. So we
        # ONLY boot it for the lifetime of this job, after Demucs has run
        # and right before the synthesis fase. Kept off otherwise — idle
        # VRAM stays near 0.
        s2pro_manager = None
        if tts_engine == "s2pro":
            from dubbing_generator.tts.s2pro_server_manager import S2ProServerManager
            s2pro_manager = S2ProServerManager(config)
            app.state.s2pro_manager = s2pro_manager

        pipeline = DubbingPipeline(config, s2pro_manager=s2pro_manager)
        try:
            if input_path.is_file():
                srt = _resolve_srt_for(input_path)
                if srt is None:
                    raise FileNotFoundError(f"No Spanish SRT found for {input_path.name}")
                emit(JobEvent(type="log", data={"message": f"using literal ES SRT: {srt.name}"}))
                if force:
                    _remove_existing_output(input_path)
                out = pipeline.process_file(input_path, srt)
                emit(JobEvent(type="progress", data={"pct": 100, "videos": 1}))
                emit(JobEvent(type="log", data={"message": f"dubbed: {out.name}"}))
            else:
                if force:
                    for vid in input_path.rglob("*"):
                        if vid.is_file() and vid.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi") \
                                and "_DOBLADO" not in vid.stem \
                                and vid.parent.name.lower() not in ("doblajes", "elevenlabs"):
                            _remove_existing_output(vid)
                results = pipeline.process_directory(input_path)
                emit(JobEvent(type="progress", data={"pct": 100, "videos": len(results)}))
        finally:
            # Always release VRAM held by s2.cpp, even on exception/abort.
            if s2pro_manager is not None:
                emit(JobEvent(type="log", data={"message": "Stopping s2.cpp server (VRAM release)"}))
                s2pro_manager.stop()
                app.state.s2pro_manager = None


app = create_app(service_name=SERVICE_NAME, task_fn=_run_dubbing_generator)


# ======================================================================
# S2-Pro server lifecycle
# ======================================================================
#
# Lazy-load model: NO startup hook. The s2.cpp server is booted by
# `_run_dubbing_generator` only for the duration of an s2pro job and
# stopped in its finally block. Rationale: the GGUF (Q6_K, ~5 GB)
# residing in VRAM permanently would starve Demucs (fase 0/6) on a
# 6 GB card (RTX 2060). Demucs and S2-Pro never coexist in the
# pipeline so serializing them is correct.
#
# The shutdown hook is kept as a safety net in case the process exits
# unexpectedly mid-job (uvicorn graceful shutdown still wants to
# terminate any subprocess we spawned).

def _stop_s2pro_server() -> None:
    manager = getattr(app.state, "s2pro_manager", None)
    if manager is not None:
        manager.stop()


app.router.on_shutdown.append(_stop_s2pro_server)


@app.post("/maintenance/restart")
def restart_service() -> dict:
    """Graceful self-restart: libera VRAM y reinicia el proceso.

    Docker reinicia el contenedor automáticamente (restart: unless-stopped).
    Útil para liberar VRAM tras un OOM o cancelación de job — el s2.cpp
    server hijo sigue al PID 1 al recibir SIGTERM.
    """
    import threading, os, signal
    # Kill PID 1 (entrypoint) so Docker restarts the container. Killing only
    # os.getpid() (worker) leaves the entrypoint alive without a worker.
    threading.Timer(0.5, lambda: os.kill(1, signal.SIGTERM)).start()
    return {"ok": True, "message": "Reiniciando dubbing-generator…"}


@app.get("/s2pro/status")
def s2pro_status() -> dict:
    """Report whether the S2-Pro subprocess is running and ready.

    With lazy-load the manager only exists during an active job; idle
    GET returns ``{"running":false,"ready":false,"engine":"idle"}``.
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


# ======================================================================
# Voice profile listing
# ======================================================================

VOICES_DIR = Path("/voices")


def _transcript_path_for(voice_path: Path) -> Path:
    """Return the sidecar .txt that holds the reference transcript.

    Stored next to the WAV with the same stem so moving voices around
    keeps the transcript paired with the audio without a database.
    """
    return voice_path.with_suffix(".txt")


def _read_transcript(voice_path: Path) -> str:
    sidecar = _transcript_path_for(voice_path)
    if not sidecar.exists():
        return ""
    try:
        return sidecar.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


@app.get("/voices")
def list_voices() -> dict:
    """List WAV files available under /voices as selectable voice profiles.

    Operators drop files in ``dubbing-generator/voices/`` (mounted to
    ``/voices`` inside the container). The UI lets users pick one per
    instructional; the chosen path is passed as ``model_voice_path``.
    Each voice may have a ``<stem>.txt`` sidecar with its reference
    transcription — when present we return it so the UI can pre-fill
    the s2_ref_text field instead of forcing the user to retype it
    every time they switch voices.
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


class _TranscriptBody(BaseModel):
    transcript: str


@app.put("/voices/{filename}/transcript")
def save_voice_transcript(filename: str, body: _TranscriptBody) -> dict:
    """Persist the reference transcription for a given voice file.

    The transcript is written to ``<voice_stem>.txt`` next to the WAV
    so it travels with the voice file. Path traversal is rejected.
    """
    # Reject any path component — only bare filenames inside /voices.
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="invalid filename")
    voice_path = VOICES_DIR / filename
    if not voice_path.exists() or not voice_path.is_file():
        raise HTTPException(status_code=404, detail="voice not found")
    sidecar = _transcript_path_for(voice_path)
    try:
        sidecar.write_text((body.transcript or "").strip() + "\n", encoding="utf-8")
    except OSError as exc:
        # Most common: the /voices bind-mount was declared :ro in compose.
        # Surface a useful message instead of FastAPI's generic 500.
        raise HTTPException(
            status_code=500,
            detail=(
                f"Cannot write transcript to {sidecar}: {exc}. "
                "Check that /voices is mounted read/write in docker-compose."
            ),
        ) from exc
    return {"ok": True, "voice": filename, "transcript_path": str(sidecar)}


# ======================================================================
# Debug / Analyze endpoint
# ======================================================================

class AnalyzeRequest(BaseModel):
    video_path: str
    srt_path: Optional[str] = None
    synthesize: bool = False      # if True, run TTS & measure real durations (slow)
    max_phrases: Optional[int] = None  # cap synthesis to N phrases (for speed)
    voice_profile: Optional[str] = None  # filename under /voices for ES ref
    model_voice_path: Optional[str] = None  # explicit absolute path override


@app.post("/analyze")
def analyze_dubbing(req: AnalyzeRequest) -> dict:
    """Diagnostic analysis of dubbing pipeline for a single video.

    Returns:
      - SRT blocks with durations + gaps
      - Planned slots (aligner output with borrowed gap time)
      - Density metrics (chars/ms → speed pressure per phrase)
      - TTS fit report (estimated vs slot, compression ratios) — static by default
      - If `synthesize=true`: actual TTS audio lengths, overflow/underflow,
        overlap resolution, final timeline.
    """
    from dubbing_generator.config import DubbingConfig
    from dubbing_generator.pipeline import parse_srt
    from dubbing_generator.sync.aligner import SyncAligner
    from dubbing_generator.sync.drift_corrector import DriftCorrector

    vp = Path(req.video_path)
    if not vp.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")

    srt_p = Path(req.srt_path) if req.srt_path else _resolve_srt_for(vp)
    if srt_p is None or not srt_p.exists():
        raise HTTPException(status_code=404, detail=f"SRT not found for {vp.name}")

    model_voice_path = req.model_voice_path or ""
    if not model_voice_path and req.voice_profile:
        candidate = Path("/voices") / req.voice_profile
        if candidate.exists():
            model_voice_path = str(candidate)
    if not model_voice_path:
        model_voice_path = os.environ.get("DUBBING_MODEL_VOICE_PATH") or ""

    cfg = DubbingConfig(
        use_model_voice=bool(model_voice_path and Path(model_voice_path).exists()),
        model_voice_path=model_voice_path,
    )
    blocks = parse_srt(srt_p)
    aligner = SyncAligner(cfg)
    planned = aligner.plan(blocks)
    drift = DriftCorrector(cfg)
    drift.reset()

    # 1. SRT blocks summary
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

    # 2. Gaps between SRT blocks
    srt_gaps = []
    for i in range(len(blocks) - 1):
        gap = blocks[i + 1].start_ms - blocks[i].end_ms
        if gap > 0:
            srt_gaps.append({
                "after_idx": blocks[i].index,
                "gap_ms": gap,
                "borrowed_by_previous": max(0, gap - cfg.inter_phrase_pad_ms),
            })

    # 3. Planned slots with density pressure
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

    # 4. Stats summary
    overflow_count = sum(1 for r in planned_rows if r["will_overflow"])
    compression_vals = [r["compression_needed"] for r in planned_rows if r["compression_needed"] > 0]
    summary = {
        "total_phrases": len(planned),
        "total_chars": sum(len(p.text) for p in planned),
        "total_allocated_ms": sum(p.allocated_ms for p in planned),
        "srt_duration_ms": blocks[-1].end_ms if blocks else 0,
        "will_overflow_count": overflow_count,
        "max_compression": round(max(compression_vals), 2) if compression_vals else 0,
        "avg_compression": round(sum(compression_vals) / len(compression_vals), 2) if compression_vals else 0,
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

    # 5. Optional real synthesis pass
    if req.synthesize:
        result["synthesis"] = _run_synthesis_probe(
            vp, planned, cfg, max_phrases=req.max_phrases,
        )

    return result


def _run_synthesis_probe(video_path: Path, planned: list, cfg, max_phrases=None) -> dict:
    """Actually run TTS on N phrases and report real durations.

    Returns per-phrase actual tts_ms, post-stretch ms, overflow/underflow,
    final placement after overlap resolution.
    """
    from dubbing_generator.audio.separator import AudioSeparator
    from dubbing_generator.audio.stretcher import stretch_audio
    from dubbing_generator.sync.drift_corrector import DriftCorrector
    from dubbing_generator.tts import build_synthesizer
    from dubbing_generator.tts.voice_cloner import VoiceCloner
    from pydub import AudioSegment

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
                "idx": i, "text": block.text[:80],
                "error": str(exc),
            })

    # Overlap detection
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
