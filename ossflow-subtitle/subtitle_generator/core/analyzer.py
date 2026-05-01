"""Análisis profundo de un vídeo para debug de generación de subtítulos.

Migrado de ``app.py:analyze_video`` en T31.3. Función pura — no usa
``app`` global y no toca el state container del servicio. Recibe el
``AnalyzeRequest`` (parametrizado para evitar acoplar a Pydantic v1/v2)
y devuelve el dict de diagnóstico (energy maps, transcripciones raw +
denoised, gaps, hallucination filter stats).

Es CPU/GPU-bound (carga WhisperX, hace transcribe + align). Se llama
desde ``api/router.py`` o el endpoint legacy ``/analyze`` en ``app.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fastapi import HTTPException


class _AnalyzeRequest(Protocol):
    video_path: str
    model: str
    language: str


def analyze_video(req: _AnalyzeRequest) -> dict[str, Any]:
    """Análisis diagnóstico profundo de un vídeo.

    Devuelve detalles de energy/RMS, decisiones VAD, segmentos
    transcritos (raw + denoised + aligned + filtered + SRT del disco),
    gaps temporales y stats del hallucination filter.
    """
    from subtitle_generator.config import (
        DEFAULT_HOTWORDS,
        DEFAULT_INITIAL_PROMPT,
        SubtitleConfig,
        TranscriptionConfig,
    )
    from subtitle_generator.cuda_setup import (
        setup_nvidia_dlls,
        setup_pytorch_safety,
    )
    from subtitle_generator.hallucination_filter import HallucinationFilter

    vp = Path(req.video_path)
    if not vp.exists():
        raise HTTPException(
            status_code=404, detail=f"Video not found: {req.video_path}",
        )

    setup_nvidia_dlls()
    setup_pytorch_safety()

    import whisperx

    t_config = TranscriptionConfig(
        model_name=req.model,
        language=req.language,
        initial_prompt=DEFAULT_INITIAL_PROMPT,
    )
    s_config = SubtitleConfig()

    # 1. Load audio.
    audio = whisperx.load_audio(str(vp))
    sr = 16000
    duration = len(audio) / sr

    # 2. Energy map — RMS por segundo.
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

    # 3. Denoise y comparación.
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

    # 4. Transcribe (raw, no denoise) para ver qué produce Whisper.
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

    # ``s.get("start")`` puede ser ``None`` en algunas ramas de WhisperX
    # (segmentos sin alineamiento); coerción a float previene
    # ``TypeError: argument of type 'NoneType' is not iterable``.
    def _sec(v):
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    # 4a. Transcribe original audio.
    result_raw = model.transcribe(
        audio, batch_size=t_config.batch_size, language=req.language,
    )
    raw_segments = [
        {
            "start": round(_sec(s.get("start")), 3),
            "end": round(_sec(s.get("end")), 3),
            "text": s.get("text", ""),
        }
        for s in result_raw.get("segments", [])
    ]

    # 4b. Transcribe denoised audio.
    result_dn = model.transcribe(
        denoised, batch_size=t_config.batch_size, language=req.language,
    )
    dn_segments = [
        {
            "start": round(_sec(s.get("start")), 3),
            "end": round(_sec(s.get("end")), 3),
            "text": s.get("text", ""),
        }
        for s in result_dn.get("segments", [])
    ]

    # 5. Detectar gaps en transcripción denoised.
    gaps = []
    sorted_segs = sorted(dn_segments, key=lambda s: s["start"])
    if sorted_segs and sorted_segs[0]["start"] > 2.0:
        gaps.append({
            "start": 0.0,
            "end": round(sorted_segs[0]["start"], 3),
            "duration": round(sorted_segs[0]["start"], 3),
        })
    for i in range(len(sorted_segs) - 1):
        gap_start = sorted_segs[i]["end"]
        gap_end = sorted_segs[i + 1]["start"]
        gap_dur = gap_end - gap_start
        if gap_dur > 2.0:
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
        gaps.append({
            "start": round(sorted_segs[-1]["end"], 3),
            "end": round(duration, 3),
            "duration": round(duration - sorted_segs[-1]["end"], 3),
        })

    # 6. Hallucination filter sobre los segmentos denoised.
    h_filter = HallucinationFilter(s_config, initial_prompt=t_config.initial_prompt)
    align_model, align_meta = whisperx.load_align_model(
        language_code=req.language, device=t_config.device,
    )
    aligned_result = whisperx.align(
        result_dn.get("segments", []),
        align_model,
        align_meta,
        denoised,
        device=t_config.device,
        return_char_alignments=False,
    )
    aligned_segs = aligned_result.get("segments", [])

    aligned_display = [
        {
            "start": round(s.get("start", 0), 3),
            "end": round(s.get("end", 0), 3),
            "text": s.get("text", ""),
        }
        for s in aligned_segs
    ]

    pre_filter_count = len(aligned_segs)
    filtered, dropped_segs = h_filter.filter_all(
        aligned_segs, audio_path=vp, return_dropped=True,
    )
    post_filter_count = len(filtered)

    filtered_display = [
        {
            "start": round(s.get("start", 0), 3),
            "end": round(s.get("end", 0), 3),
            "text": s.get("text", ""),
        }
        for s in filtered
    ]

    # Detectar gaps en segmentos FILTRADOS (lo que llega al SRT).
    gaps_filtered = []
    sorted_filtered = sorted(filtered_display, key=lambda s: s["start"])
    if sorted_filtered and sorted_filtered[0]["start"] > 2.0:
        gaps_filtered.append({
            "start": 0.0,
            "end": round(sorted_filtered[0]["start"], 3),
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

    # 7. Leer el SRT existente del disco (ground truth para gap detection).
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
        sorted_srt = sorted(segments_srt, key=lambda s: s["start"])
        if sorted_srt and sorted_srt[0]["start"] > 2.0:
            gaps_srt.append({
                "start": 0.0,
                "end": round(sorted_srt[0]["start"], 3),
                "duration": round(sorted_srt[0]["start"], 3),
            })
        for i in range(len(sorted_srt) - 1):
            gs = sorted_srt[i]["end"]
            ge = sorted_srt[i + 1]["start"]
            gd = ge - gs
            if gd > 1.0:
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

    # Cleanup GPU.
    import gc
    import torch
    del model, align_model, align_meta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _sanitize(obj):
        """Convierte tipos numpy a Python nativo para JSON."""
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
