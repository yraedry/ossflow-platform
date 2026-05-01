"""Orquestador batch del flujo translate.

Migrado de ``app.py`` en T31.4.

Componentes:

* ``build_translator_with_fallback(opts)`` — factoría primary+fallback
  desde ``opts`` del job.
* ``translate_for_dubbing(primary, srt_path, out_path, cps, force_slot_mode)``
  — flujo dubbing nivel 2 (slot-anchored) o nivel 3 (speech-anchored)
  según presencia de ``<video>.words.json``.
* ``translate_for_dubbing_nivel3(primary, words_path, out_path, cps)``
  — speech-anchored: re-segmenta por pausas reales, adapta cada segmento
  a su budget de chars y escribe SRT con timestamps del habla real.
* ``run_translate_directory(req, emit)`` — recorre ``req.input_path``
  buscando SRTs, traduce cada uno, emite progreso SSE y acumula errores
  para fallar el job entero si hay traducciones incompletas.

Recibe ``RunRequest`` y ``emit`` (callable) — no toca state global y
no depende del FastAPI app.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ossflow_service_kit import JobEvent, RunRequest, emit_logs

from subtitle_generator.shared.paths import (
    literal_srt_path_for,
    words_json_for,
)


def build_translator_with_fallback(opts: dict) -> tuple[Any, Any]:
    """Factoría ``(primary, fallback)`` desde las options del job.

    Provider primary defaults a ``ollama``. Fallback solo se usa si el
    primary lanza excepción al traducir. Ollama no necesita api_key.
    """
    from subtitle_generator.translator import make_translator

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


def translate_for_dubbing(
    primary,
    srt_path: Path,
    out_path: Path,
    cps: float,
    force_slot_mode: bool = False,
) -> None:
    """Dubbing translation — nivel 3 (speech-anchored) con fallback nivel 2.

    Estrategia:
      1. Si ``<video>.words.json`` existe junto al SRT → re-segmentar
         speech por pausas reales (dub_segmenter) y construir el dub
         desde ahí. Timestamps del habla real, no del slot de lectura.
      2. Si no, fallback a slot-anchored (nivel 2): mantiene timestamps
         del SRT original, solo adapta texto al budget.

    ``force_slot_mode=True`` salta nivel 3 incluso con words.json.
    Útil cuando el output debe preservar timestamps originales (e.g.
    ``.es.srt`` literal compartido entre subtítulos y pipeline dubbing).

    Solo soportado con translator chat-based (Ollama o OpenAI).
    """
    from subtitle_generator.srt_io import parse_srt, serialize_srt
    from subtitle_generator.translator import _BaseChatTranslator

    if not isinstance(primary, _BaseChatTranslator):
        raise RuntimeError(
            "dubbing_mode requires a chat-based translator (Ollama or OpenAI)"
        )

    words_path = words_json_for(srt_path)
    if words_path.exists() and not force_slot_mode:
        translate_for_dubbing_nivel3(primary, words_path, out_path, cps)
        return

    # Fallback nivel 2 — SRT slots como source of truth.
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


def translate_for_dubbing_nivel3(
    primary, words_path: Path, out_path: Path, cps: float,
) -> None:
    """Speech-anchored dub translation.

    Re-segmenta speech por pausas reales desde word-timestamps, luego
    adapta cada segmento a su budget de chars. Escribe SRT con
    timestamps del habla real, eliminando los gaps heredados de
    los subtítulos de lectura.
    """
    from subtitle_generator.dub_segmenter import (
        SegmenterConfig,
        load_words,
        segment_speech,
    )
    from subtitle_generator.srt_io import serialize_srt

    words = load_words(words_path)
    segments = segment_speech(words, SegmenterConfig())
    if not segments:
        out_path.write_text("", encoding="utf-8")
        return

    items = [
        {"text": s["text"], "duration_ms": s["duration_ms"]}
        for s in segments
    ]
    # Speech-anchored: slots = real talk time, así que ES debe LLENAR
    # el budget (no solo respetarlo) o el TTS deja silencio dentro del habla.
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


def run_translate_directory(req: RunRequest, emit) -> None:
    """Traduce cada .srt bajo ``input_path`` a ``*.es.srt``.

    Acepta input file (un solo SRT o vídeo con sidecar EN) o directorio
    (recorrido recursivo). Excluye carpetas de output (``doblajes/``,
    ``elevenlabs/``) y SRTs ya traducidos (``.es.srt``, ``.dub.es.srt``,
    ``_ES``, ``_ESP_DUB``).

    Strict accounting: si un SRT no se traduce ni con primary ni con
    fallback, se acumula en ``failed`` y al final se hace ``raise``
    para que el step termine FAILED. Sin esto, el dubbing posterior
    doblaría solo los SRTs que sí se tradujeron y reportaría completed,
    confundiendo al usuario.
    """
    opts = req.options or {}
    level = logging.DEBUG if opts.get("verbose") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(req.input_path)
    if root.is_file():
        if root.suffix.lower() == ".srt":
            srts = [root]
        else:
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

    # Excluir subcarpetas de output (sus .srt vienen de vídeos ya procesados
    # y re-traducirlos crea .es.srt espurios junto a dubs).
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
        primary, fallback = build_translator_with_fallback(opts)
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

        failed: list[tuple[str, str]] = []

        for i, srt in enumerate(srts, 1):
            sub_out = literal_srt_path_for(srt)

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
                    translate_for_dubbing(primary, srt, sub_out, cps, force_slot_mode=False)
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
                        from subtitle_generator.translator import _BaseChatTranslator
                        try:
                            if isinstance(fallback, _BaseChatTranslator):
                                translate_for_dubbing(fallback, srt, sub_out, cps, force_slot_mode=False)
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
