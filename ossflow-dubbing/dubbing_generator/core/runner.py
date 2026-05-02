"""Bridge ``RunRequest`` → ``DubbingPipeline``.

Migrado de ``app.py`` en T32.7. Resuelve el SRT español adyacente al
vídeo (literal preferido, dub fallback), construye ``DubbingConfig``
desde las options del job, gestiona el ciclo de vida del s2.cpp
server (lazy-load + finally stop) y delega en ``DubbingPipeline``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ossflow_service_kit import JobEvent, RunRequest, emit_logs


def resolve_input(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def resolve_srt_for(video_path: Path) -> Optional[Path]:
    """Resuelve el SRT español adyacente a un vídeo."""
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


def run_dubbing_generator(req: RunRequest, emit, app_state=None) -> None:
    """Bridge ``RunRequest`` → ``DubbingPipeline``.

    ``app_state`` (opcional) permite al endpoint ``/s2pro/status``
    consultar el manager activo. Cuando se llama desde el task_fn
    de FastAPI, el caller pasa ``app.state``.
    """
    logger = logging.getLogger(__name__)
    input_path = resolve_input(Path(req.input_path))

    opts = req.options or {}
    level = logging.DEBUG if opts.get("verbose") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    with emit_logs(emit, level=level):
        from dubbing_generator.config import DubbingConfig
        from dubbing_generator.pipeline import DubbingPipeline

        tts_engine = (
            opts.get("tts_engine")
            or os.environ.get("DUBBING_TTS_ENGINE")
            or "s2pro"
        ).strip().lower()

        # Voice profile resolution solo aplica a engines que consumen
        # model_voice_path (XTTS-style cloning, ElevenLabs cloning).
        # S2-Pro lee su referencia exclusivamente de cfg.s2_ref_audio_path.
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
                        "message": (
                            f"voice_profile '{voice_profile}' not found "
                            "in /voices — cloning instructor"
                        ),
                    }))
            if not model_voice_path:
                model_voice_path = os.environ.get("DUBBING_MODEL_VOICE_PATH") or ""

            use_model_voice = bool(opts.get("use_model_voice"))
            if not use_model_voice and model_voice_path:
                use_model_voice = Path(model_voice_path).exists()

        # Tras T22.5 solo S2-Pro está soportado.
        if tts_engine != "s2pro":
            emit(JobEvent(type="log", data={
                "message": (
                    f"unsupported tts_engine={tts_engine!r}, "
                    "falling back to 's2pro'"
                ),
            }))
            tts_engine = "s2pro"

        config_kwargs = dict(
            use_model_voice=use_model_voice,
            model_voice_path=model_voice_path,
            tts_engine=tts_engine,
        )
        for k in (
            "s2_ref_audio_path", "s2_ref_text",
            "s2_temperature", "s2_top_p", "s2_top_k",
            "s2_max_tokens", "s2_gguf_path",
        ):
            val = opts.get(k)
            if val is not None:
                config_kwargs[k] = val
        config_kwargs.setdefault("merge_max_chars", 300)
        config_kwargs.setdefault("merge_max_gap_ms", 600)
        config_kwargs.setdefault("inter_phrase_crossfade_ms", 100)
        config_kwargs.setdefault("force_crossfade_ms", 250)
        config_kwargs.setdefault("rms_jump_crossfade_ms", 0)

        config = DubbingConfig(**config_kwargs)

        emit(JobEvent(type="log", data={
            "message": f"starting dubbing-generator on {input_path}",
        }))
        force = bool(opts.get("force"))

        def _remove_existing_output(video_path: Path) -> None:
            """Drop ``<Season>/doblajes/<name>.mkv`` para regeneración."""
            candidate = video_path.parent / "doblajes" / f"{video_path.stem}.mkv"
            if candidate.exists():
                try:
                    candidate.unlink()
                    emit(JobEvent(type="log", data={
                        "message": f"force overwrite: removed existing {candidate.name}",
                    }))
                except OSError as exc:
                    emit(JobEvent(type="log", data={
                        "message": f"ERROR removing {candidate.name}: {exc}",
                    }))

        # S2-Pro lazy-load: el s2.cpp server retiene ~5 GB VRAM (Q6_K)
        # una vez mmap'd. En 6 GB (RTX 2060) no caben Demucs + S2-Pro,
        # así que solo lo bootamos durante la fase de synthesis.
        s2pro_manager = None
        if tts_engine == "s2pro":
            from dubbing_generator.tts.s2pro_server_manager import (
                S2ProServerManager,
            )
            s2pro_manager = S2ProServerManager(config)
            if app_state is not None:
                app_state.s2pro_manager = s2pro_manager

        pipeline = DubbingPipeline(config, s2pro_manager=s2pro_manager)
        try:
            if input_path.is_file():
                srt = resolve_srt_for(input_path)
                if srt is None:
                    raise FileNotFoundError(
                        f"No Spanish SRT found for {input_path.name}",
                    )
                emit(JobEvent(type="log", data={
                    "message": f"using literal ES SRT: {srt.name}",
                }))
                if force:
                    _remove_existing_output(input_path)
                out = pipeline.process_file(input_path, srt)
                emit(JobEvent(type="progress", data={"pct": 100, "videos": 1}))
                emit(JobEvent(type="log", data={
                    "message": f"dubbed: {out.name}",
                }))
            else:
                if force:
                    for vid in input_path.rglob("*"):
                        if (
                            vid.is_file()
                            and vid.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi")
                            and "_DOBLADO" not in vid.stem
                            and vid.parent.name.lower() not in ("doblajes", "elevenlabs")
                        ):
                            _remove_existing_output(vid)
                results = pipeline.process_directory(input_path)
                # Surface the per-run report so the orchestrator can
                # detect partial completion (some chapters failed TTS or
                # had no ES SRT) and decide whether to retry. Permissive
                # mode (see ``DubbingPipeline.process_directory`` doc):
                # ``last_run_report`` is always populated post-run.
                report = getattr(pipeline, "last_run_report", None)
                if report:
                    emit(JobEvent(type="log", data={
                        "summary": report,
                        "message": (
                            f"Dubbing report: {report['dubbed']}/{report['total']} doblados, "
                            f"{len(report['missing_srt'])} sin SRT, "
                            f"{len(report['failed_tts'])} fallaron TTS"
                        ),
                    }))
                emit(JobEvent(type="progress", data={
                    "pct": 100, "videos": len(results),
                }))
        finally:
            if s2pro_manager is not None:
                emit(JobEvent(type="log", data={
                    "message": "Stopping s2.cpp server (VRAM release)",
                }))
                s2pro_manager.stop()
                if app_state is not None:
                    app_state.s2pro_manager = None
