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

from bjj_service_kit import JobEvent, RunRequest, create_app, emit_logs  # noqa: E402
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


_regenerator: object | None = None


def _get_regenerator(model: str, language: str):
    global _regenerator
    from subtitle_generator.config import TranscriptionConfig  # type: ignore
    from subtitle_generator.segment_regen import SegmentRegenerator  # type: ignore

    if _regenerator is None:
        _regenerator = SegmentRegenerator(
            TranscriptionConfig(model_name=model, language=language)
        )
    return _regenerator


def _resolve_input(path: Path) -> Path:
    """Validate input path exists. Returns the path as-is (file or directory)."""
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _run_subtitle_generator(req: RunRequest, emit) -> None:
    """Bridge RunRequest -> subtitle_generator.pipeline.SubtitlePipeline.

    When ``options.translate_only=True`` runs SRT translation (EN→ES via Ollama/OpenAI)
    instead of transcription — reusing the same job/SSE contract.
    """
    opts = req.options or {}

    if opts.get("translate_only"):
        _run_translate_directory(req, emit)
        return

    input_path = _resolve_input(Path(req.input_path))

    level = logging.DEBUG if opts.get("verbose") else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    with emit_logs(emit, level=level):
        from subtitle_generator.config import (  # type: ignore
            DEFAULT_INITIAL_PROMPT,
            SubtitleConfig,
            TranscriptionConfig,
            generate_prompt,
        )
        from subtitle_generator.cuda_setup import setup_nvidia_dlls, setup_pytorch_safety  # type: ignore
        from subtitle_generator.pipeline import SubtitlePipeline  # type: ignore

        setup_nvidia_dlls()
        setup_pytorch_safety()

        if opts.get("prompt") is not None:
            initial_prompt = opts["prompt"]
        elif opts.get("instructor") or opts.get("topic"):
            initial_prompt = generate_prompt(instructor=opts.get("instructor"), topic=opts.get("topic"))
        else:
            initial_prompt = DEFAULT_INITIAL_PROMPT

        t_config = TranscriptionConfig(
            model_name=opts.get("model", "large-v3"),
            language=opts.get("language", "en"),
            batch_size=int(opts.get("batch_size", 4)),
            initial_prompt=initial_prompt,
            postprocess_openai=bool(opts.get("postprocess_openai", False)),
            postprocess_model=str(opts.get("postprocess_model", "gpt-4o-mini")),
            postprocess_api_key=opts.get("postprocess_api_key") or os.environ.get("OPENAI_API_KEY"),
        )
        s_config = SubtitleConfig()

        force = bool(opts.get("force", False))
        emit(JobEvent(type="log", data={"message":
            f"starting subtitle-generator on {input_path}"
            + (" (force overwrite)" if force else "")
        }))
        import gc
        import torch

        pipeline = SubtitlePipeline(t_config, s_config)
        pipeline.load_models()
        try:
            if input_path.is_file():
                pipeline.process_file(input_path, force=force)
            else:
                pipeline.process_directory(input_path, force=force)
            emit(JobEvent(type="progress", data={"pct": 100}))
        finally:
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _build_translator_with_fallback(opts: dict):
    """Build (primary, fallback) translators from job options.

    Primary provider defaults to ``ollama``. Fallback is used only if the
    primary raises at translate time. Ollama no necesita api_key.
    """
    from subtitle_generator.translator import make_translator  # type: ignore

    provider = (opts.get("provider") or "ollama").lower()
    primary_key = opts.get("api_key")
    if not primary_key and provider == "openai":
        primary_key = os.environ.get("OPENAI_API_KEY")

    primary = make_translator(
        provider,
        api_key=primary_key,
        source_lang=opts.get("source_lang", "EN"),
        target_lang=opts.get("target_lang", "ES"),
        model=opts.get("model"),
        formality=opts.get("formality"),
    )

    fb_name = (opts.get("fallback_provider") or "").lower() or None
    fb = None
    if fb_name and fb_name != provider:
        fb_key = opts.get("fallback_api_key")
        if not fb_key and fb_name == "openai":
            fb_key = os.environ.get("OPENAI_API_KEY")
        try:
            fb = make_translator(
                fb_name,
                api_key=fb_key,
                source_lang=opts.get("source_lang", "EN"),
                target_lang=opts.get("target_lang", "ES"),
                model=opts.get("fallback_model"),
                formality=opts.get("formality"),
            )
        except ValueError:
            fb = None
    return primary, fb


def _clean_base_stem(srt_path: Path) -> str:
    """Return the video-level stem of an SRT path, stripping language tags.

    Examples:
      "S01E01 foo.en.srt"  -> "S01E01 foo"
      "S01E01 foo_EN.srt"  -> "S01E01 foo"
      "S01E01 foo.srt"     -> "S01E01 foo"
    """
    stem = srt_path.stem
    for tag in (".en", ".EN", "_en", "_EN"):
        if stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


def _literal_srt_path_for(srt_path: Path) -> Path:
    """Return <video-stem>.es.srt sidecar (literal subtitle)."""
    return srt_path.with_name(f"{_clean_base_stem(srt_path)}.es.srt")


def _dub_srt_path_for(srt_path: Path) -> Path:
    """Return <video-stem>.dub.es.srt sidecar (iso-sync dubbing script)."""
    return srt_path.with_name(f"{_clean_base_stem(srt_path)}.dub.es.srt")


def _words_json_for(srt_path: Path) -> Path:
    """Return <video>.words.json sibling for a given SRT.

    The SRT may be .en.srt, .srt or already an .es.srt — we always walk back
    to the base video stem.
    """
    return srt_path.with_name(f"{_clean_base_stem(srt_path)}.words.json")


def _translate_for_dubbing(
    primary, srt_path: Path, out_path: Path, cps: float,
    force_slot_mode: bool = False,
) -> None:
    """Dubbing translation — industry nivel 3 (speech-anchored) with fallback.

    Strategy:
      1. If ``<video>.words.json`` exists next to the SRT → re-segment speech
         by real pauses (dub_segmenter) and build the dub track from there.
         Timestamps mirror the speaker's breath, not the reading-oriented SRT.
      2. Otherwise fall back to the old slot-anchored mode (nivel 2): keep
         original SRT timestamps, just adapt text to its budget.

    ``force_slot_mode=True`` skips nivel 3 even when words.json exists. Use
    this when the output must preserve the original SRT timestamps (so the
    dub stays in sync with the on-screen video — e.g. when writing the
    literal ``.es.srt`` track used both by subtitles and by the dubbing
    pipeline).

    Only supported when ``primary`` is a chat-based translator (Ollama or OpenAI, budget prompt-based).
    """
    from subtitle_generator.srt_io import parse_srt, serialize_srt  # type: ignore
    from subtitle_generator.translator import _BaseChatTranslator  # type: ignore

    if not isinstance(primary, _BaseChatTranslator):
        raise RuntimeError(
            "dubbing_mode requires a chat-based translator (Ollama or OpenAI)"
        )

    words_path = _words_json_for(srt_path)
    if words_path.exists() and not force_slot_mode:
        _translate_for_dubbing_nivel3(primary, words_path, out_path, cps)
        return

    # Fallback nivel 2 — SRT slots as source of truth.
    subs = parse_srt(srt_path)
    if not subs:
        out_path.write_text("", encoding="utf-8")
        return
    items = [
        {
            "text": s["text"],
            "duration_ms": max(
                100,
                int((float(s.get("end", 0)) - float(s.get("start", 0))) * 1000),
            ),
        }
        for s in subs
    ]
    adapted = primary.translate_for_dubbing(items, cps=cps)
    if len(adapted) != len(subs):
        raise RuntimeError(
            f"dubbing translator returned {len(adapted)} items, expected {len(subs)}"
        )
    for sub, new_text in zip(subs, adapted):
        sub["text"] = new_text
    out_path.write_text(serialize_srt(subs), encoding="utf-8")


def _translate_for_dubbing_nivel3(
    primary, words_path: Path, out_path: Path, cps: float,
) -> None:
    """Speech-anchored dub translation.

    Re-segments speech by real pauses from word-timestamps, then adapts each
    segment to its own char budget. Writes an SRT whose timestamps mirror
    the speaker's real rhythm, eliminating the gaps inherited from reading
    subtitles.
    """
    from subtitle_generator.dub_segmenter import (  # type: ignore
        SegmenterConfig,
        load_words,
        segment_speech,
    )
    from subtitle_generator.srt_io import serialize_srt  # type: ignore

    words = load_words(words_path)
    segments = segment_speech(words, SegmenterConfig())
    if not segments:
        out_path.write_text("", encoding="utf-8")
        return

    items = [
        {"text": s["text"], "duration_ms": s["duration_ms"]}
        for s in segments
    ]
    # Speech-anchored: slots = real talk time, so ES must fill the budget
    # (not just respect it) or the TTS will leave silence inside speech.
    adapted = primary.translate_for_dubbing(items, cps=cps, fill_budget=True)
    if len(adapted) != len(segments):
        raise RuntimeError(
            f"dubbing translator returned {len(adapted)} items, expected {len(segments)}"
        )

    subs = [
        {"start": seg["start"], "end": seg["end"], "text": text}
        for seg, text in zip(segments, adapted)
    ]
    out_path.write_text(serialize_srt(subs), encoding="utf-8")


def _run_translate_directory(req: RunRequest, emit) -> None:
    """Translate every .srt under input_path to *_ES.srt."""
    opts = req.options or {}
    level = logging.DEBUG if opts.get("verbose") else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    root = Path(req.input_path)
    if root.is_file():
        if root.suffix.lower() == ".srt":
            srts = [root]
        else:
            # Video file — look for any sibling EN SRT. WhisperX writes
            # <stem>.en.srt; legacy outputs may be <stem>.srt.
            stem = root.stem
            srts = []
            for sfx in (".en.srt", ".EN.srt", ".srt"):
                candidate = root.with_name(f"{stem}{sfx}")
                if candidate.exists():
                    srts = [candidate]
                    break
        root = root.parent
    elif root.is_dir():
        srts = sorted(root.rglob("*.srt"))
    else:
        raise FileNotFoundError(f"input_path not found: {root}")

    # Exclude output subfolders (doblajes/, elevenlabs/) — their .srt files
    # come from already-processed videos and re-translating them creates
    # spurious .es.srt next to dubs.
    _EXCLUDED_DIR_NAMES = {"doblajes", "elevenlabs"}
    srts = [
        s for s in srts
        if not s.name.endswith(".es.srt")
        and not s.name.endswith(".dub.es.srt")
        and not s.stem.endswith("_ES")
        and not s.stem.endswith("_ESP_DUB")
        and not any(part.lower() in _EXCLUDED_DIR_NAMES for part in s.parts)
    ]

    with emit_logs(emit, level=level):
        primary, fallback = _build_translator_with_fallback(opts)
        provider_name = (opts.get("provider") or "ollama").lower()
        fb_name = (opts.get("fallback_provider") or "").lower() or None
        force = bool(opts.get("force"))

        total = len(srts)
        emit(JobEvent(type="log", data={"message":
            f"translating {total} SRT file(s) in {root} via {provider_name}"
            + (f" (fallback: {fb_name})" if fallback else "")
        }))
        if total == 0:
            emit(JobEvent(type="progress", data={"pct": 100}))
            return

        dubbing_mode = bool(opts.get("dubbing_mode"))
        cps = float(opts.get("dubbing_cps", 16.0))
        if dubbing_mode:
            emit(JobEvent(type="log", data={"message":
                f"dubbing_mode ON (cps={cps}): .es.srt will be compacted to fit "
                f"per-slot char budgets while keeping original SRT timestamps"
            }))

        # Strict accounting (since 2026-04-29): un SRT que no se traduce
        # NI por primary NI por fallback antes se loggeaba "ERROR" y el
        # job seguía como success. Eso dejaba la Season con M/N .es.srt y
        # el dubbing posterior dobla solo los M y reportaba completed →
        # el usuario veía "4/9 episodios doblados" sin notificación. Ahora
        # acumulamos errores y al final del bucle hacemos raise para que
        # el step termine FAILED.
        failed: list[tuple[str, str]] = []  # (srt_name, error)

        for i, srt in enumerate(srts, 1):
            # Single target — the literal .es.srt. In dubbing_mode the text is
            # compacted to each SRT slot's char budget (still anchored to the
            # original timestamps so the dub stays in sync with the on-screen
            # video). No .dub.es.srt sidecar is written.
            sub_out = _literal_srt_path_for(srt)

            if sub_out.exists() and force:
                try:
                    sub_out.unlink()
                    emit(JobEvent(type="log", data={"message":
                        f"force overwrite: removed existing {sub_out.name}"
                    }))
                except OSError as exc:
                    emit(JobEvent(type="log", data={"message":
                        f"ERROR removing {sub_out.name}: {exc}"
                    }))
            if sub_out.exists():
                emit(JobEvent(type="log", data={"message": f"skip subs (exists): {sub_out.name}"}))
            elif dubbing_mode:
                last_err: str | None = None
                ok = False
                try:
                    # force_slot_mode=False → use speech-anchored level 3 when
                    # a .words.json exists (real pause-based segmentation for
                    # fluent dub without mid-speech silence). Falls back to
                    # level 2 (SRT slots) automatically when no words index is
                    # present. Tolerance for visual drift: ~1-2 s per segment.
                    _translate_for_dubbing(primary, srt, sub_out, cps, force_slot_mode=False)
                    emit(JobEvent(type="log", data={"message":
                        f"translated subs (dub-compact): {srt.name} -> {sub_out.name}"
                    }))
                    ok = True
                except Exception as exc:
                    last_err = f"{provider_name}: {exc}"
                    emit(JobEvent(type="log", data={"message":
                        f"{provider_name} dub-compact failed on {srt.name}: {exc}"
                    }))
                    if fallback is not None:
                        from subtitle_generator.translator import _BaseChatTranslator  # type: ignore
                        try:
                            if isinstance(fallback, _BaseChatTranslator):
                                _translate_for_dubbing(fallback, srt, sub_out, cps, force_slot_mode=False)
                                emit(JobEvent(type="log", data={"message":
                                    f"translated subs via fallback {fb_name} (dub-aware): {srt.name}"
                                }))
                            else:
                                fallback.translate_srt(srt, sub_out)
                                emit(JobEvent(type="log", data={"message":
                                    f"translated subs via fallback {fb_name} (literal): {srt.name}"
                                }))
                            ok = True
                        except Exception as exc2:
                            last_err = f"{provider_name}: {exc} | {fb_name}: {exc2}"
                            emit(JobEvent(type="log", data={"message":
                                f"ERROR fallback {fb_name} also failed on {srt.name}: {exc2}"
                            }))
                if not ok:
                    failed.append((srt.name, last_err or "unknown"))
            else:
                last_err: str | None = None
                ok = False
                try:
                    primary.translate_srt(srt, sub_out)
                    emit(JobEvent(type="log", data={"message": f"translated subs: {srt.name} -> {sub_out.name}"}))
                    ok = True
                except Exception as exc:
                    last_err = f"{provider_name}: {exc}"
                    emit(JobEvent(type="log", data={"message":
                        f"{provider_name} failed on {srt.name}: {exc}"
                    }))
                    if fallback is not None:
                        try:
                            fallback.translate_srt(srt, sub_out)
                            emit(JobEvent(type="log", data={"message":
                                f"translated subs via fallback {fb_name}: {srt.name}"
                            }))
                            ok = True
                        except Exception as exc2:
                            last_err = f"{provider_name}: {exc} | {fb_name}: {exc2}"
                            emit(JobEvent(type="log", data={"message":
                                f"ERROR fallback {fb_name} also failed on {srt.name}: {exc2}"
                            }))
                if not ok:
                    failed.append((srt.name, last_err or "unknown"))

            pct = int(i * 100 / total)
            emit(JobEvent(type="progress", data={"pct": pct}))

        if failed:
            names = ", ".join(name for name, _ in failed[:5])
            if len(failed) > 5:
                names += " …"
            raise RuntimeError(
                f"Traducción incompleta: {total - len(failed)}/{total} OK, "
                f"{len(failed)} fallaron: {names}"
            )


def _hf_cache_root() -> Path:
    return Path(
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or "/models/huggingface"
    )


def _clear_hf_locks() -> dict:
    """Delete stale .lock files inside the HuggingFace hub cache.

    Returns a summary dict with counts. Safe: only touches files under
    `<cache>/hub/.locks/` (or legacy `<cache>/.locks/`).
    """
    root = _hf_cache_root()
    candidates = [root / "hub" / ".locks", root / ".locks"]
    removed, errors = 0, []
    for base in candidates:
        if not base.exists():
            continue
        for lock in base.rglob("*.lock"):
            try:
                lock.unlink()
                removed += 1
            except Exception as exc:
                errors.append(f"{lock}: {exc}")
    return {"removed": removed, "errors": errors, "roots": [str(c) for c in candidates]}


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
    """Deep diagnostic analysis of a video for subtitle generation debugging.

    Returns detailed info about audio energy, VAD decisions, transcription
    segments, and what filters would drop.
    """
    from pathlib import Path as _P
    import numpy as np

    from subtitle_generator.config import (
        DEFAULT_INITIAL_PROMPT,
        DEFAULT_HOTWORDS,
        SubtitleConfig,
        TranscriptionConfig,
    )
    from subtitle_generator.cuda_setup import setup_nvidia_dlls, setup_pytorch_safety
    from subtitle_generator.hallucination_filter import HallucinationFilter

    vp = _P(req.video_path)
    if not vp.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")

    setup_nvidia_dlls()
    setup_pytorch_safety()

    import whisperx

    t_config = TranscriptionConfig(
        model_name=req.model,
        language=req.language,
        initial_prompt=DEFAULT_INITIAL_PROMPT,
    )
    s_config = SubtitleConfig()

    # 1. Load audio
    audio = whisperx.load_audio(str(vp))
    sr = 16000
    duration = len(audio) / sr

    # 2. Energy map — RMS per second
    energy_map = []
    for sec in range(int(duration) + 1):
        start_s = sec * sr
        end_s = min((sec + 1) * sr, len(audio))
        chunk = audio[start_s:end_s]
        if len(chunk) == 0:
            energy_map.append({"sec": sec, "rms_db": -100.0})
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_db = float(20 * np.log10(rms + 1e-10))
        energy_map.append({"sec": sec, "rms_db": round(rms_db, 1)})

    # 3. Denoise and compare
    try:
        import noisereduce as nr
        denoised = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)
        denoised = denoised.astype(np.float32)
        has_denoise = True
    except ImportError:
        denoised = audio
        has_denoise = False

    energy_map_denoised = []
    if has_denoise:
        for sec in range(int(duration) + 1):
            start_s = sec * sr
            end_s = min((sec + 1) * sr, len(audio))
            chunk = denoised[start_s:end_s]
            if len(chunk) == 0:
                energy_map_denoised.append({"sec": sec, "rms_db": -100.0})
                continue
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_db = float(20 * np.log10(rms + 1e-10))
            energy_map_denoised.append({"sec": sec, "rms_db": round(rms_db, 1)})

    # 4. Transcribe (raw, no denoise) to see what Whisper produces
    hotwords = DEFAULT_HOTWORDS
    asr_options = {
        "initial_prompt": t_config.initial_prompt,
        "hotwords": hotwords,
        "beam_size": t_config.beam_size,
        "condition_on_previous_text": t_config.condition_on_previous_text,
    }

    model = whisperx.load_model(
        t_config.model_name,
        t_config.device,
        compute_type=t_config.compute_type,
        asr_options=asr_options,
        vad_options={
            "vad_onset": t_config.vad_onset,
            "vad_offset": t_config.vad_offset,
        },
    )

    # `s.get("start")` puede ser ``None`` en algunas ramas de WhisperX (segmentos
    # sin alineamiento); coerción a float previene ``TypeError: argument of type
    # 'NoneType' is not iterable`` en el round/sort posteriores.
    def _sec(v):
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    # 4a. Transcribe original audio
    result_raw = model.transcribe(audio, batch_size=t_config.batch_size, language=req.language)
    raw_segments = [
        {"start": round(_sec(s.get("start")), 3), "end": round(_sec(s.get("end")), 3), "text": s.get("text", "")}
        for s in result_raw.get("segments", [])
    ]

    # 4b. Transcribe denoised audio
    result_dn = model.transcribe(denoised, batch_size=t_config.batch_size, language=req.language)
    dn_segments = [
        {"start": round(_sec(s.get("start")), 3), "end": round(_sec(s.get("end")), 3), "text": s.get("text", "")}
        for s in result_dn.get("segments", [])
    ]

    # 5. Detect gaps in denoised transcription
    gaps = []
    sorted_segs = sorted(dn_segments, key=lambda s: s["start"])
    if sorted_segs and sorted_segs[0]["start"] > 2.0:
        gaps.append({"start": 0.0, "end": round(sorted_segs[0]["start"], 3), "duration": round(sorted_segs[0]["start"], 3)})
    for i in range(len(sorted_segs) - 1):
        gap_start = sorted_segs[i]["end"]
        gap_end = sorted_segs[i + 1]["start"]
        gap_dur = gap_end - gap_start
        if gap_dur > 2.0:
            # Check energy in gap
            g_start_s = int(gap_start * sr)
            g_end_s = int(gap_end * sr)
            g_chunk = denoised[g_start_s:g_end_s]
            g_rms = float(np.sqrt(np.mean(g_chunk ** 2))) if len(g_chunk) > 0 else 0
            g_rms_db = float(20 * np.log10(g_rms + 1e-10))
            gaps.append({
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "duration": round(gap_dur, 3),
                "energy_db": round(g_rms_db, 1),
                "likely_speech": bool(g_rms_db > -40.0),
            })
    if sorted_segs and (duration - sorted_segs[-1]["end"]) > 2.0:
        gaps.append({"start": round(sorted_segs[-1]["end"], 3), "end": round(duration, 3), "duration": round(duration - sorted_segs[-1]["end"], 3)})

    # 6. Run hallucination filter on denoised segments to see what gets dropped
    h_filter = HallucinationFilter(s_config, initial_prompt=t_config.initial_prompt)
    # Simulate — need word-level data for some filters, so do alignment first
    align_model, align_meta = whisperx.load_align_model(language_code=req.language, device=t_config.device)
    aligned_result = whisperx.align(
        result_dn.get("segments", []),
        align_model,
        align_meta,
        denoised,
        device=t_config.device,
        return_char_alignments=False,
    )
    aligned_segs = aligned_result.get("segments", [])

    # Extract aligned segments (pre-filter) for display
    aligned_display = [
        {"start": round(s.get("start", 0), 3), "end": round(s.get("end", 0), 3), "text": s.get("text", "")}
        for s in aligned_segs
    ]

    pre_filter_count = len(aligned_segs)
    filtered, dropped_segs = h_filter.filter_all(aligned_segs, audio_path=vp, return_dropped=True)
    post_filter_count = len(filtered)

    # Extract filtered segments for display
    filtered_display = [
        {"start": round(s.get("start", 0), 3), "end": round(s.get("end", 0), 3), "text": s.get("text", "")}
        for s in filtered
    ]

    # Detect gaps in FILTERED segments (what actually ends up in the SRT)
    gaps_filtered = []
    sorted_filtered = sorted(filtered_display, key=lambda s: s["start"])
    if sorted_filtered and sorted_filtered[0]["start"] > 2.0:
        gaps_filtered.append({
            "start": 0.0, "end": round(sorted_filtered[0]["start"], 3),
            "duration": round(sorted_filtered[0]["start"], 3),
        })
    for i in range(len(sorted_filtered) - 1):
        gf_start = sorted_filtered[i]["end"]
        gf_end = sorted_filtered[i + 1]["start"]
        gf_dur = gf_end - gf_start
        if gf_dur > 2.0:
            g_start_s = int(gf_start * sr)
            g_end_s = int(gf_end * sr)
            g_chunk = denoised[g_start_s:g_end_s]
            g_rms = float(np.sqrt(np.mean(g_chunk ** 2))) if len(g_chunk) > 0 else 0
            g_rms_db = float(20 * np.log10(g_rms + 1e-10))
            gaps_filtered.append({
                "start": round(gf_start, 3),
                "end": round(gf_end, 3),
                "duration": round(gf_dur, 3),
                "energy_db": round(g_rms_db, 1),
                "likely_speech": bool(g_rms_db > -40.0),
            })
    if sorted_filtered and (duration - sorted_filtered[-1]["end"]) > 2.0:
        gaps_filtered.append({
            "start": round(sorted_filtered[-1]["end"], 3),
            "end": round(duration, 3),
            "duration": round(duration - sorted_filtered[-1]["end"], 3),
        })

    # 7. Read existing SRT from disk (ground truth for gap detection)
    srt_path = vp.with_suffix(".srt")
    segments_srt: list[dict] = []
    gaps_srt: list[dict] = []
    if srt_path.exists():
        from subtitle_generator.srt_io import parse_srt
        srt_blocks = parse_srt(srt_path)
        segments_srt = [
            {"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"]}
            for s in srt_blocks
        ]
        # Detect gaps between consecutive SRT blocks
        sorted_srt = sorted(segments_srt, key=lambda s: s["start"])
        if sorted_srt and sorted_srt[0]["start"] > 2.0:
            gaps_srt.append({
                "start": 0.0, "end": round(sorted_srt[0]["start"], 3),
                "duration": round(sorted_srt[0]["start"], 3),
            })
        for i in range(len(sorted_srt) - 1):
            gs = sorted_srt[i]["end"]
            ge = sorted_srt[i + 1]["start"]
            gd = ge - gs
            if gd > 1.0:  # 1s threshold for SRT gaps (blocks are shorter than segments)
                g_start_s = int(gs * sr)
                g_end_s = int(ge * sr)
                g_chunk = denoised[g_start_s:g_end_s]
                g_rms = float(np.sqrt(np.mean(g_chunk ** 2))) if len(g_chunk) > 0 else 0
                g_rms_db = float(20 * np.log10(g_rms + 1e-10))
                gaps_srt.append({
                    "start": round(gs, 3),
                    "end": round(ge, 3),
                    "duration": round(gd, 3),
                    "energy_db": round(g_rms_db, 1),
                    "likely_speech": bool(g_rms_db > -40.0),
                })
        if sorted_srt and (duration - sorted_srt[-1]["end"]) > 2.0:
            gaps_srt.append({
                "start": round(sorted_srt[-1]["end"], 3),
                "end": round(duration, 3),
                "duration": round(duration - sorted_srt[-1]["end"], 3),
            })

    # Cleanup GPU
    import gc, torch
    del model, align_model, align_meta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _sanitize(obj):
        """Convert numpy types to Python natives for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return _sanitize({
        "video_path": str(vp),
        "duration_seconds": round(duration, 2),
        "has_denoise": has_denoise,
        "vad_params": {
            "onset": t_config.vad_onset,
            "offset": t_config.vad_offset,
        },
        "transcription": {
            "raw_segments": len(raw_segments),
            "denoised_segments": len(dn_segments),
            "improvement": len(dn_segments) - len(raw_segments),
        },
        "segments_raw": raw_segments,
        "segments_denoised": dn_segments,
        "segments_aligned": aligned_display,
        "segments_filtered": filtered_display,
        "segments_dropped": dropped_segs,
        "segments_srt": segments_srt,
        "gaps": gaps,
        "gaps_filtered": gaps_filtered,
        "gaps_srt": gaps_srt,
        "hallucination_filter": {
            "input_segments": pre_filter_count,
            "output_segments": post_filter_count,
            "dropped": pre_filter_count - post_filter_count,
            "stats": h_filter.stats,
        },
        "energy_map": energy_map,
        "energy_map_denoised": energy_map_denoised,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
