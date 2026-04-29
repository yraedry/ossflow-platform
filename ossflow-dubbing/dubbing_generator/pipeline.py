"""Dubbing pipeline orchestrator."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from pydub import AudioSegment

from .config import DubbingConfig
from .audio.mixer import AudioMixer, TtsSegment
from .audio.separator import AudioSeparator
from .audio.stretcher import stretch_audio, trim_silence
from .sync.aligner import SrtBlock, SyncAligner
from .sync.drift_corrector import DriftCorrector
from .sync.words_index import WordsIndex
from .tts import build_synthesizer
from .tts.bjj_casting import castellanize as _castellanize
from .tts.voice_cloner import VoiceCloner

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


# ======================================================================
# SRT parsing helpers
# ======================================================================

def _parse_time(time_str: str) -> int:
    """Parse ``HH:MM:SS,mmm`` to milliseconds."""
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def parse_srt(srt_path: Path) -> list[SrtBlock]:
    """Parse an SRT file into a list of :class:`SrtBlock`."""
    content = srt_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d+)\n"
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"
        r"(.*?)(?=\n\n|\n$|\Z)",
        re.DOTALL,
    )
    blocks: list[SrtBlock] = []
    for m in pattern.finditer(content):
        text = m.group(4).replace("\n", " ").strip()
        text = re.sub(r"\((.*?)\)", r"\1", text)
        blocks.append(SrtBlock(
            index=int(m.group(1)),
            start_ms=_parse_time(m.group(2)),
            end_ms=_parse_time(m.group(3)),
            text=text,
        ))
    return blocks


_CONTINUATION_STARTS = (
    "y ", "o ", "u ", "e ", "pero ", "porque ", "pues ", "así que ",
    "aunque ", "sino ", "mientras ", "cuando ", "donde ", "como ",
    "que ", "para ", "al ", "del ", "de ", "en ", "con ", "sin ",
    "sobre ", "entre ", "hasta ", "desde ", "a ",
)


def _apply_prosodic_continuity(text: str, next_text: str | None) -> str:
    """Adjust trailing punctuation so the TTS doesn't close prosody.

    Si la siguiente frase continúa el discurso (empieza con minúscula o
    con conector), cambiamos el punto final `.` por coma para que el TTS
    no marque cierre entonativo. Signos fuertes (! ?) quedan intactos
    porque sí marcan intención. Sin siguiente frase, dejamos el punto
    (es el cierre real del bloque).
    """
    if not text or not next_text:
        return text
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return text
    if stripped.endswith("...") or stripped.endswith(".."):
        return text

    nxt_clean = next_text.lstrip()
    if not nxt_clean:
        return text

    first_char = nxt_clean[0]
    continues = first_char.islower()
    if not continues:
        lower_nxt = nxt_clean.lower()
        continues = any(lower_nxt.startswith(c) for c in _CONTINUATION_STARTS)

    if not continues:
        return text

    trailing_ws = text[len(stripped):]
    return stripped[:-1] + "," + trailing_ws


# ======================================================================
# Pipeline
# ======================================================================

class DubbingPipeline:
    """Orchestrate the full dubbing workflow for a single video."""

    def __init__(
        self,
        config: DubbingConfig,
        progress_cb: ProgressCallback = None,
        s2pro_manager=None,
    ) -> None:
        self.cfg = config
        self._progress_cb = progress_cb
        # Optional: lifecycle hook for the s2.cpp HTTP server. When
        # present, the pipeline boots it just before fase 3 (synthesis)
        # so it doesn't hold VRAM during Demucs (fase 0). The caller is
        # responsible for stopping it after the job ends.
        self._s2pro_manager = s2pro_manager

        self.separator = AudioSeparator(config)
        self.voice_cloner = VoiceCloner(config)
        # Defer synthesizer construction until fase 3 ONLY when we own
        # the s2.cpp lifecycle (manager passed in). If no manager is
        # given, assume the server is already running externally
        # (CLI / __main__ usage, tests) and build immediately so the
        # legacy code path keeps working.
        defer = (config.tts_engine == "s2pro" and s2pro_manager is not None)
        self.synthesizer = None if defer else build_synthesizer(config)
        self.aligner = SyncAligner(config)
        self.drift = DriftCorrector(config)
        self.mixer = AudioMixer(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(
        self,
        video_path: Path,
        srt_path: Path,
        voice_ref: Optional[Path] = None,
    ) -> Path:
        """Run the full dubbing pipeline on one video."""
        base_name = video_path.with_suffix("")
        # Final MKV lives under ``<Season>/doblajes/<name>.mkv`` (separate
        # from ElevenLabs Studio output at ``<Season>/elevenlabs/``). The
        # scratch WAV stays next to the source — it's cleaned up after
        # muxing so leaking it into the doblajes folder is pointless.
        doblajes_dir = base_name.parent / "doblajes"
        doblajes_dir.mkdir(parents=True, exist_ok=True)
        output_video = doblajes_dir / f"{base_name.name}.mkv"
        output_audio = base_name.parent / f"{base_name.name}_AUDIO_ESP.wav"

        if output_video.exists():
            logger.info("Output already exists, skipping: %s", output_video)
            return output_video

        # 1. Separate background audio
        self._report(0, 6, "Separating background audio...")
        background_path = self.separator.separate(video_path)
        # Demucs holds ~2 GB pyTorch tensors after the subprocess returns
        # because the parent process still imports torch (loaded by other
        # modules). Force a release before the s2.cpp server tries to mmap
        # the ~5 GB GGUF — on a 6 GB card the difference is OOM vs. fits.
        self._release_torch_vram()

        # 2. Voice reference.
        #
        # S2-Pro uses cfg.s2_ref_audio_path directly inside SynthesizerS2Pro
        # (the global voice WAV from settings, e.g. Martin Osborne). The
        # reference_wav arg passed to synth.generate() is ignored — see
        # SynthesizerS2Pro.generate(). Running VoiceCloner here would build
        # a 30-60 s VAD'd extract from the instructor's vocals and then
        # throw it away, plus emit a misleading "Using override voice
        # profile" log. Skip it entirely for s2pro.
        #
        # Other engines (ElevenLabs cloning, Piper, Kokoro) genuinely
        # consume the returned ref_wav, so keep the existing path.
        self._report(1, 6, "Extracting voice reference...")
        if self.cfg.tts_engine == "s2pro":
            ref_wav = Path(self.cfg.s2_ref_audio_path)
        else:
            vocals_stem = video_path.with_name(f"{video_path.stem}_VOCALS.wav")
            effective_voice_ref = voice_ref or (vocals_stem if vocals_stem.exists() else None)
            ref_wav = self.voice_cloner.get_reference(video_path, effective_voice_ref)

        # 3. Parse SRT, plan alignment (pass video duration so the last phrase
        #    can borrow the tail gap instead of spilling past end of video).
        self._report(2, 6, "Planning phrase alignment...")
        blocks = parse_srt(srt_path)
        if getattr(self.cfg, "merge_consecutive_blocks", False):
            blocks = self._merge_consecutive_blocks(
                blocks,
                self.cfg.merge_max_gap_ms,
                getattr(self.cfg, "merge_max_chars", 200),
            )
        video_duration_ms = self._probe_video_duration_ms(video_path)
        # Words index from WhisperX alignment — lets the aligner mark
        # artificial SRT gaps (speaker still talking) so the pipeline
        # can close them without invading real pauses.
        words_index = WordsIndex.load(video_path)
        if words_index is None:
            logger.info("No words.json sidecar — artificial-gap detection off")
        planned = self.aligner.plan(
            blocks,
            video_duration_ms=video_duration_ms,
            words_index=words_index,
        )

        # 4. Synthesize all phrases
        self._report(3, 6, "Synthesizing speech...")
        # Lazy-boot s2.cpp server now that Demucs has released its VRAM.
        # No-op for engines that don't use a manager (elevenlabs/piper/
        # kokoro). For s2pro the synthesizer was deferred in __init__ so
        # we build it AFTER the server is up.
        if self._s2pro_manager is not None and self.cfg.tts_engine == "s2pro":
            logger.info("Booting s2.cpp server (lazy-load before synthesis)...")
            self._s2pro_manager.start()
            ok = self._s2pro_manager.wait_until_ready(
                timeout=self.cfg.s2_health_timeout_s
            )
            if not ok:
                logger.error(
                    "s2.cpp server failed to become ready in %.0fs — "
                    "synthesis will yield silence and fail gracefully via the "
                    "circuit breaker.",
                    self.cfg.s2_health_timeout_s,
                )
            self.synthesizer = build_synthesizer(
                self.cfg, server_manager=self._s2pro_manager,
            )

        tts_segments = self._synthesize_all(
            planned, ref_wav,
            video_duration_ms=video_duration_ms,
            words_index=words_index,
        )

        # 5. Mix background + TTS with ducking
        self._report(4, 6, "Mixing audio with ducking...")
        background = AudioSegment.from_wav(str(background_path))
        # Asegura que el background cubre toda la duración del vídeo. Si
        # Demucs entrega un stem más corto, el mux con -shortest truncaría
        # imagen + audio al mínimo → output corrupto antes del final real.
        if video_duration_ms and len(background) < video_duration_ms:
            pad_ms = video_duration_ms - len(background)
            background = background + AudioSegment.silent(
                duration=pad_ms, frame_rate=background.frame_rate,
            )
        mixed = self.mixer.mix(background, tts_segments)
        # Red de seguridad floor: el audio final debe cubrir al menos
        # la duración del vídeo (si el mixer dio menos, algo raro pasó
        # con Demucs — pad con silencio para evitar MKV truncado).
        if video_duration_ms and len(mixed) < video_duration_ms:
            pad_ms = video_duration_ms - len(mixed)
            mixed = mixed + AudioSegment.silent(
                duration=pad_ms, frame_rate=mixed.frame_rate,
            )
        # Red de seguridad ceiling (R10): el mixer tiene cap vía
        # ``video_tail_extension_max_ms`` pero si algún segmento TTS
        # queda más largo de lo esperado tras overlap/compact passes
        # podría devolver un mixed más largo. Sin ``-shortest`` en
        # ``_mux_video``, ffmpeg alargaría el vídeo al audio → usuario
        # oye ES después de que el vídeo termine. Truncamos aquí.
        if video_duration_ms and getattr(
            self.cfg, "allow_video_tail_extension", False,
        ):
            max_audio_ms = video_duration_ms + getattr(
                self.cfg, "video_tail_extension_max_ms", 1500,
            )
            if len(mixed) > max_audio_ms:
                logger.info(
                    "Truncating mixed audio %d ms → %d ms (video %d ms + tail %d ms cap)",
                    len(mixed), max_audio_ms, video_duration_ms,
                    max_audio_ms - video_duration_ms,
                )
                fade_ms = min(200, (len(mixed) - max_audio_ms) // 2, 500)
                mixed = mixed[:max_audio_ms].fade_out(max(50, fade_ms))
        mixed.export(str(output_audio), format="wav")

        # 6. Mux into video
        self._report(5, 6, "Muxing final video...")
        self._mux_video(video_path, output_audio, output_video)

        # Stop s2.cpp server now — Demucs of the next episode (process_directory
        # batch mode) cannot share VRAM with a 4.4 GB GGUF residing on a 6 GB
        # card. The server is re-booted at fase 3 of the next file. Re-mmap is
        # cheap (~2-3s) since the GGUF stays in OS page cache.
        if self._s2pro_manager is not None and self.cfg.tts_engine == "s2pro":
            logger.info("Stopping s2.cpp server (release VRAM for next file)...")
            self._s2pro_manager.stop()

        # 7. QA — boundary metrics over the TTS-only timeline (no background
        # interference) plus UTMOS over the final mixed wav. Best-effort:
        # failures here never abort the dub.
        try:
            self._run_qa(video_path, tts_segments, mixed)
        except Exception:
            logger.exception("QA failed for %s (non-fatal)", output_video.name)

        # Cleanup. For s2pro the vocals stem was never bound (we skipped the
        # cloner branch); we still want to remove the artefact Demucs left
        # on disk so build artifacts don't accumulate per-episode.
        vocals_stem_path = video_path.with_name(f"{video_path.stem}_VOCALS.wav")
        vocals_stem_to_clean = vocals_stem_path if (not voice_ref and vocals_stem_path.exists()) else None
        self._cleanup(output_audio, background_path, *(p for p in [vocals_stem_to_clean] if p))

        self._report(6, 6, f"Done: {output_video.name}")
        logger.info("Dubbed video saved: %s", output_video)
        return output_video

    def process_directory(self, root_dir: Path) -> list[Path]:
        """Process all videos in *root_dir* that have matching SRT files.

        Strict mode (since 2026-04-29 fix): if any chapter is silently
        missing — either because no ES SRT exists for it OR because TTS
        raised — we raise ``RuntimeError`` after the loop with a summary.
        Previously a logged warning + continue meant a partial Season
        (e.g. 4/9 chapters dubbed) was reported as ``completed`` to the
        orchestrator, which is the worst-case failure mode: the user
        thinks the Season is done.
        """
        results: list[Path] = []
        chapters_missing_srt: list[str] = []
        chapters_failed: list[tuple[str, str]] = []  # (name, error)
        total_chapters = 0

        for dirpath, dirs, files in os.walk(root_dir):
            # Skip artefact folders so we never re-process our own outputs.
            # ``doblajes/`` holds the flujo-B pipeline outputs, ``elevenlabs/``
            # holds Studio E2E MP4s — feeding either back through the
            # pipeline would burn credits and overwrite good dubs with
            # worse ones. Mutating ``dirs`` in-place prunes the walk.
            dirs[:] = [d for d in dirs if d.lower() not in ("doblajes", "elevenlabs")]
            videos = sorted(
                f for f in files
                if f.lower().endswith(self.cfg.extensions)
                and "_DOBLADO" not in f
            )
            for video_name in videos:
                total_chapters += 1
                video_path = Path(dirpath) / video_name
                base = video_path.with_suffix("")

                srt_path = None
                # Híbrido: .es.srt literal (alineado con vídeo) preferido.
                # Merge consecutivo + stretch absorbe la compresión ES. El
                # .dub.es.srt (nivel 3) queda como fallback si no hay literal.
                for _sfx in (".es.srt", ".ES.srt", "_ES.srt", "_ESP.srt", ".dub.es.srt"):
                    _candidate = base.parent / f"{base.name}{_sfx}"
                    if _candidate.exists():
                        srt_path = _candidate
                        break
                if srt_path is None:
                    logger.warning("No SRT found for %s, skipping", video_name)
                    chapters_missing_srt.append(video_name)
                    continue

                logger.info("Using ES SRT: %s", srt_path.name)

                try:
                    out = self.process_file(video_path, srt_path)
                    results.append(out)
                except Exception as exc:
                    logger.exception("Error processing %s", video_name)
                    chapters_failed.append((video_name, str(exc)))

        # Strict accounting: every chapter must end up either in results or
        # in one of the failure lists. If anything failed, raise so the
        # orchestrator marks the step FAILED rather than COMPLETED.
        if chapters_missing_srt or chapters_failed:
            parts = []
            if chapters_missing_srt:
                parts.append(
                    f"{len(chapters_missing_srt)} sin SRT ES: "
                    + ", ".join(chapters_missing_srt[:5])
                    + (" …" if len(chapters_missing_srt) > 5 else "")
                )
            if chapters_failed:
                parts.append(
                    f"{len(chapters_failed)} fallaron TTS: "
                    + ", ".join(name for name, _ in chapters_failed[:5])
                    + (" …" if len(chapters_failed) > 5 else "")
                )
            summary = (
                f"Dubbing incompleto: {len(results)}/{total_chapters} chapters doblados. "
                + " | ".join(parts)
            )
            raise RuntimeError(summary)

        return results

    # ------------------------------------------------------------------
    # Synthesis loop — anchor-based sync (each phrase anchored to SRT start)
    # ------------------------------------------------------------------

    def _synthesize_all(
        self,
        planned: list,
        ref_wav: Path,
        video_duration_ms: int | None = None,
        words_index: Optional["WordsIndex"] = None,
    ) -> list[TtsSegment]:
        """Synthesize TTS anchored to SRT timestamps.

        Strategy:
          - Each phrase placed at its SRT start_ms (NO accumulated drift).
          - Fitted to allocated slot via pitch-preserving stretch.
          - Small overflow allowed (max_overflow_ms) to avoid over-compression.
          - Drift corrector adjusts TTS speed based on text-length pressure,
            not positional drift (anchor-based = no positional drift).
        """
        self.drift.reset()
        segments: list[TtsSegment] = []

        total = len(planned)
        max_overflow = self.cfg.max_overflow_ms
        lead_silence = self.cfg.tts_lead_silence_ms

        # Constant-speed synthesis: every phrase at the same natural pace
        # (config.speed_base). No density-driven acceleration. This trades
        # perfect sync for prosodic consistency — the user prefers ES to
        # sound uniform even if it drifts slightly past SRT cues.
        use_constant_speed = getattr(self.cfg, "constant_speed", False)
        allow_tail = getattr(self.cfg, "allow_video_tail_extension", False)

        # Tail-window anti-drift: cuando entramos en los últimos
        # ``tail_speed_nudge_window_ms`` del vídeo y llevamos deriva
        # acumulada, subimos el ratio de compresión hasta
        # ``tail_speed_nudge_max_ratio`` para reabsorber el retraso
        # antes del final — evita que el tail_extension tenga que
        # estirar el vídeo varios segundos al final del capítulo.
        tail_window_ms = getattr(self.cfg, "tail_speed_nudge_window_ms", 0)
        tail_trigger_ms = getattr(self.cfg, "tail_speed_nudge_trigger_ms", 0)
        tail_max_ratio = getattr(
            self.cfg, "tail_speed_nudge_max_ratio", self.cfg.max_compression_ratio,
        )
        tail_window_start = (
            (video_duration_ms - tail_window_ms)
            if (video_duration_ms and tail_window_ms > 0) else None
        )

        for i, block in enumerate(planned):
            if not block.text or len(block.text) < 2:
                continue

            if use_constant_speed:
                speed = self.cfg.speed_base
            else:
                density = len(block.text) / max(block.allocated_ms, 1)
                speed = self.drift.check_density(i, density)

            next_text = planned[i + 1].text if i + 1 < total else None
            text_for_tts = _apply_prosodic_continuity(block.text, next_text)
            # Castellanizar compuestos BJJ (≥2 palabras) para evitar que
            # XTTS alucine en la transición ES→EN dentro del mismo prompt.
            # Los términos de 1 palabra (guard, mount…) quedan en EN y se
            # pronuncian con acento inglés sin desviar.
            text_for_tts = _castellanize(text_for_tts)

            try:
                raw_audio = self.synthesizer.generate(
                    text_for_tts, ref_wav, speed=speed,
                )

                # Trim residual silence sólo bajo -55 dB. A -45 dB se
                # comía colas de consonantes sordas (s, f, p) al final
                # de frases largas — el usuario reportaba "frases
                # cortadas" justo en español largo donde XTTS cierra
                # con energía baja pero audible. Solo cortamos lo que
                # es inequívocamente silencio (respiración, room tone).
                raw_audio = trim_silence(
                    raw_audio, threshold_db=-55.0, chunk_ms=10,
                )

                # Gentle rescue stretch (max 1.10x) only if TTS overflows
                # the slot + generous overflow tolerance. We intentionally
                # avoid aggressive compression so delivery stays natural.
                target_ms = block.allocated_ms + max_overflow

                # When video tail extension is enabled we don't cap against
                # end-of-video here: the mixer/muxer will pad the video to
                # fit the final TTS instead of truncating a phrase.
                if not allow_tail and video_duration_ms is not None:
                    room_to_end = video_duration_ms - block.target_start_ms
                    if room_to_end > 0:
                        target_ms = min(target_ms, room_to_end)
                    else:
                        logger.warning(
                            "Phrase %d starts past video end; skipping", i,
                        )
                        continue

                # En la ventana final, si la frase previa excedió su slot
                # y nos estamos quedando sin margen hasta el final del
                # vídeo, subimos el max_ratio efectivo para comprimir
                # más esta frase. Así el tail_extension no tiene que
                # estirar 8s el vídeo al final.
                effective_max_ratio = self.cfg.max_compression_ratio
                if (
                    tail_window_start is not None
                    and block.target_start_ms >= tail_window_start
                    and len(segments) > 0
                ):
                    prev_seg = segments[-1]
                    # Validar planned_idx: si la síntesis previa falló y
                    # dejó planned_idx=-1, o si está fuera de rango,
                    # saltamos el nudge para esta frase (no tenemos
                    # referencia de slot anterior). Sin esta guarda,
                    # planned[-1] devuelve el último bloque — drift
                    # calculado sería basura y dispararía el nudge
                    # aleatoriamente.
                    prev_pidx = prev_seg.planned_idx
                    if 0 <= prev_pidx < len(planned):
                        prev_planned = planned[prev_pidx]
                        prev_slot_end = (
                            prev_planned.target_start_ms
                            + prev_planned.allocated_ms
                        )
                        drift_ms = prev_seg.end_ms - prev_slot_end
                        if drift_ms >= tail_trigger_ms:
                            effective_max_ratio = tail_max_ratio
                            logger.info(
                                "Tail nudge @phrase %d: drift=%d ms, "
                                "max_ratio %.2f→%.2f",
                                i, drift_ms,
                                self.cfg.max_compression_ratio,
                                tail_max_ratio,
                            )

                fitted = stretch_audio(
                    raw_audio,
                    target_duration_ms=target_ms,
                    max_ratio=effective_max_ratio,
                    min_ratio=self.cfg.min_compression_ratio,
                )

                # When tail extension is disabled, truncate only as last resort.
                if not allow_tail and video_duration_ms is not None:
                    room_to_end = video_duration_ms - block.target_start_ms
                    if room_to_end > 0 and len(fitted) > room_to_end:
                        fade_ms = min(60, room_to_end // 4)
                        fitted = fitted[:room_to_end].fade_out(fade_ms)

                # Prepend micro-silence for natural attack
                if lead_silence > 0:
                    pad = AudioSegment.silent(
                        duration=lead_silence,
                        frame_rate=fitted.frame_rate,
                    )
                    fitted = pad + fitted

                # Anchor at SRT start_ms (no accumulated drift).
                # ``original_start_ms`` captures the born-at anchor so
                # later compact passes can bound how far they drift a
                # segment from its EN lip-sync position.
                segments.append(TtsSegment(
                    audio=fitted,
                    start_ms=block.target_start_ms,
                    end_ms=block.target_start_ms + len(fitted),
                    planned_idx=i,
                    original_start_ms=block.target_start_ms,
                ))

            except Exception:
                logger.exception("Error synthesizing phrase %d", i)

            if i % 10 == 0:
                logger.info("Progress: %d / %d phrases", i, total)
                self._report(3, 6, f"Synthesizing: {i}/{total}")

        # Normalize loudness across all TTS renders so neighbouring phrases
        # don't jump 10-20 dB. Each XTTS call produces audio at whatever
        # level the sampler landed on; without this pass the listener
        # hears a loud phrase followed by a soft one. Done BEFORE overlap
        # resolution so the force-fade crossfades mix equally-loud
        # amplitudes and don't boost the quiet side into clipping.
        self._normalize_tts_levels(segments)

        # NOTE: edge_fades removidos deliberadamente. El pipeline
        # aplicaba fades de 20 ms a todos los bordes; el mixer aplica
        # también fade-in/out en boundaries (inter-phrase 150 ms, forced
        # 250 ms) — se acumulaban 2-3 fades en el mismo borde → la cola
        # casi muteada y ataque seco en la siguiente frase, dando
        # saltos RMS percibidos de 15-20 dB en la ventana de 400 ms.
        # Con RMS gated + mixer fades el borde queda bien suavizado sin
        # duplicar el tratamiento.

        # Resolve overlaps: if phrase N ends after N+1 starts, nudge N+1 right
        # within the following gap (capped so we never push past the next slot
        # or the end of the video).
        self._resolve_overlaps(segments, planned, video_duration_ms)

        # Close ONLY artificial SRT gaps (speaker kept talking but the VAD
        # cut between subtitle blocks). Real pauses (artificial==0) stay
        # intact so sync with the speaker's face doesn't drift.
        self._close_artificial_gaps(segments, planned)

        # Speech-anchored (nivel 3): close trailing silence by pulling the
        # next phrase earlier when the current TTS finishes well before the
        # planned slot ends. Keeps natural inter-phrase pad.
        if getattr(self.cfg, "compact_trailing_silence", False):
            self._compact_trailing_silence(segments)

        # Synthetic-silence pass (words.json guided): when XTTS renders
        # faster than the original EN pace, phrases finish before the
        # next anchor and leave gaps that didn't exist in the source.
        # We close a gap ONLY if words.json confirms the speaker kept
        # talking through it; otherwise we preserve the silence to keep
        # lip sync with the video.
        if (
            getattr(self.cfg, "compact_synthetic_silence", False)
            and words_index is not None
        ):
            self._compact_synthetic_silence(segments, words_index)

        return segments

    @staticmethod
    def _normalize_tts_levels(segments: list[TtsSegment]) -> None:
        """Igualar loudness entre frases TTS por RMS (no por peak).

        Cada inferencia XTTS aterriza en un nivel distinto; la ruta
        GPT-latent no es loudness-aware. Si solo igualamos ``dBFS``
        (peak) dos frases con mismo peak pueden tener RMS 10 dB
        distinto según haya o no plosivas/sibilantes al final — eso
        produce los "salto RMS 14-18 dB" que siguen apareciendo en la
        QA aun tras el pase anterior. Trabajamos en RMS: es lo que
        percibe el oyente entre frase y frase.

        Pipeline:

        1. Calcular RMS dB (sobre el array int16) de cada segmento.
        2. Meta absoluta: -23 dBFS RMS (estándar hablado, coherente
           con ``ducking_fg_volume=1.6`` sin saturar al mix).
        3. Aplicar ``apply_gain(delta)`` con cap ±18/12 dB.
        4. Limitador final con ``effects.normalize(headroom=1)`` para
           que el ajuste no lleve ningún pico por encima de -1 dBFS.

        El peak-clamp va al FINAL (no al principio como el pase anterior)
        porque si normalizas picos antes de medir RMS falseas la métrica:
        una frase con un transitorio sobresaliente baja su peak pero su
        RMS sube artificialmente.
        """
        import math as _m

        # Ronda 9 — ajuste fino RMS tras 11 hard_cuts en S01E02.
        # La distribución empírica muestra que algunas frases aterrizan
        # 13-15 dB por encima del resto del capítulo (sibilantes fuertes
        # o vocales tónicas con mucho cuerpo). Con MAX_CUT=12 el cap
        # dejaba la frase aún 1-3 dB por encima del target → boundary
        # vecino 10+ dB salto. Ampliar el cut a 15 dB convierge esas
        # frases al target sin necesitar un segundo pase.
        # TARGET -24 (antes -23): da 1 dB extra de headroom al duckling
        # boost (x1.6 = +4 dB) sin tocar límites de distorsión en el
        # mix final — el cuantil 90% del RMS percibido queda en -20 dBFS.
        # S2-Pro tuning (vs older XTTS values -24/+18/-15): the boundary
        # QA on Craig Jones S04E05 still showed 14 hard cuts with 7-11 dB
        # RMS jumps after the previous values. S2-Pro's gated RMS lands
        # in a tighter band per phrase but the pase capped outliers with
        # too much margin. Tighten target + caps so the distribution
        # converges within ~3 dB of TARGET, then let the pair-lift below
        # mop up the rest. Target -22 also matches the perceived loudness
        # of the BJJ coach reference better than -24 (less "narrative").
        TARGET_RMS_DBFS = -22.0
        MAX_BOOST = 14.0
        MAX_CUT = 12.0
        # Gating threshold (ITU-R BS.1770 spirit): ignoramos silencio
        # residual bajo -40 dBFS al medir RMS. Sin gate, la cola de
        # ~100 ms bajo -55 dB que trim_silence deja detrás falsea la
        # medida RMS hacia abajo → el gain sube demasiado → la
        # siguiente frase queda 10-20 dB por encima. El usuario QA
        # reporta exactamente estos saltos (22 dB en boundary 9).
        GATE_THRESHOLD_DBFS = -40.0
        # Peak guard: si tras el ajuste RMS el pico supera -1 dBFS,
        # REDUCIMOS (nunca amplificamos). `effects.normalize` sube hasta
        # que el pico llegue a headroom → deshace el ajuste RMS que
        # acabamos de aplicar. El peak-only-cut preserva RMS target.
        PEAK_CEIL_DBFS = -1.0

        def _gated_rms_dbfs(seg) -> float:
            """Gated RMS in dBFS — ignora samples bajo ``GATE_THRESHOLD_DBFS``.

            Sin gate, el silencio residual tras trim baja artificialmente
            la RMS global y el gain se sobreajusta. El gate restringe la
            medida a la parte vocalizada de la frase, que es lo que el
            oyente percibe como "loudness".
            """
            samples = seg.get_array_of_samples()
            if len(samples) == 0:
                return float("-inf")
            max_val = float(1 << (8 * seg.sample_width - 1))
            # Threshold en magnitud normalizada: 10^(dBFS/20)
            gate = 10.0 ** (GATE_THRESHOLD_DBFS / 20.0)
            sq_sum = 0.0
            count = 0
            for v in samples:
                mag = abs(v) / max_val
                if mag >= gate:
                    sq_sum += mag * mag
                    count += 1
            # Fallback: si todo está bajo el gate (caso extremo),
            # medimos sin gate para no devolver -inf.
            if count == 0:
                count = len(samples)
                sq_sum = sum((v / max_val) ** 2 for v in samples)
            if sq_sum <= 0:
                return float("-inf")
            rms = _m.sqrt(sq_sum / count)
            return 20.0 * _m.log10(rms) if rms > 0 else float("-inf")

        # Umbral de silencio: si el RMS gated está por debajo de -55 dBFS
        # la frase es efectivamente silencio (XTTS produjo ruido o fallo
        # silencioso). Amplificar +18 dB sobre eso produce ruido audible.
        # Saltamos la normalización y dejamos pasar tal cual.
        SILENCE_FLOOR_DBFS = -55.0

        for seg in segments:
            if seg.audio is None or len(seg.audio) == 0:
                continue
            cur_rms = _gated_rms_dbfs(seg.audio)
            if not _m.isfinite(cur_rms):
                continue
            if cur_rms < SILENCE_FLOOR_DBFS:
                logger.debug(
                    "Skipping level normalization on silent segment "
                    "(gated RMS %.1f dBFS)", cur_rms,
                )
                continue
            delta = TARGET_RMS_DBFS - cur_rms
            if delta > MAX_BOOST:
                delta = MAX_BOOST
            elif delta < -MAX_CUT:
                delta = -MAX_CUT
            if abs(delta) >= 0.3:
                seg.audio = seg.audio.apply_gain(delta)
            # Peak safety: sólo bajamos si el pico tras el boost se sale
            # del headroom. Nunca subimos — eso reintroduciría saltos RMS.
            cur_peak = seg.audio.max_dBFS
            if _m.isfinite(cur_peak) and cur_peak > PEAK_CEIL_DBFS:
                seg.audio = seg.audio.apply_gain(PEAK_CEIL_DBFS - cur_peak)

        # ─── Second pass: pairwise RMS leveling ────────────────────────────
        # After the per-segment pass converges everyone near
        # TARGET_RMS_DBFS there are still edge cases where a neighbour
        # landed several dB apart (common when one side hit the gain cap
        # at MAX_CUT/MAX_BOOST). We detect those adjacent jumps and
        # nudge the LOWER segment up — never push the louder one down,
        # because ducking downstream already leans on the voice track
        # having enough body.
        # Trigger lower (was 6.0) so we mop up the 4-6 dB residuals that
        # the per-segment pass leaves between phrases the listener still
        # hears. Lift cap raised (was 4.0) so an 11 dB jump can be closed
        # to ~4 dB instead of 7 dB; peak guard below still prevents
        # clipping. Together with the tighter MAX_BOOST/MAX_CUT this
        # collapses the dispersion the boundary QA was flagging.
        PAIR_JUMP_TRIGGER_DB = 4.0
        PAIR_LIFT_MAX_DB = 7.0
        sorted_segs = sorted(
            (s for s in segments if s.audio is not None and len(s.audio) > 0),
            key=lambda s: s.start_ms,
        )
        for i in range(len(sorted_segs) - 1):
            a, b = sorted_segs[i], sorted_segs[i + 1]
            rms_a = _gated_rms_dbfs(a.audio)
            rms_b = _gated_rms_dbfs(b.audio)
            if not (_m.isfinite(rms_a) and _m.isfinite(rms_b)):
                continue
            diff = rms_a - rms_b
            if abs(diff) < PAIR_JUMP_TRIGGER_DB:
                continue
            # Only lift the quieter side. Cap the lift so we don't ruin
            # the overall RMS distribution, and respect peak headroom.
            lift = min(PAIR_LIFT_MAX_DB, abs(diff) - PAIR_JUMP_TRIGGER_DB + 2.0)
            victim = b if diff > 0 else a
            new_audio = victim.audio.apply_gain(lift)
            cur_peak = new_audio.max_dBFS
            if _m.isfinite(cur_peak) and cur_peak > PEAK_CEIL_DBFS:
                new_audio = new_audio.apply_gain(PEAK_CEIL_DBFS - cur_peak)
            victim.audio = new_audio
            # Flag both neighbours so the mixer uses the *rms_jump* fade
            # tier (longer than force_fade) on this boundary — lift
            # alone won't hide a 10 dB jump, but lift + stronger fade
            # does. rms_jump_boundary triggers rms_xfade in the mixer.
            a.rms_jump_boundary = True
            b.rms_jump_boundary = True
            logger.debug(
                "pair-lift boundary %d: diff=%.1f dB → lift %.1f dB on %s side",
                i, diff, lift, "next" if diff > 0 else "prev",
            )

    def _resolve_overlaps(
        self,
        segments: list[TtsSegment],
        planned: list,
        video_duration_ms: int | None = None,
    ) -> None:
        """Push later segments forward + preserve a mandatory pad.

        Without the pad, two TTS renders end up butted up against each
        other (``end_prev == start_next``) and the listener hears a hard
        click between phrases — exactly the `overlap (0 ms)` seen in
        the QA sidecar. We leave ``inter_phrase_pad_ms`` of breathing
        room so the boundary has space for a fade.

        If the next phrase can't be shifted that much (no room left before
        end-of-video), we **trim the tail of the current phrase** with a
        short fade-out instead of accepting a zero gap. Losing ~100 ms
        of the prev's tail is preferable to a click — and it keeps the
        next phrase anchored to its video timestamp.

        Segments modified by this pass get ``force_fade=True`` so the
        mixer applies a stronger crossfade at those boundaries.
        """
        allow_tail = getattr(self.cfg, "allow_video_tail_extension", False)
        pad_ms = int(getattr(self.cfg, "inter_phrase_pad_ms", 0))

        for i in range(len(segments) - 1):
            cur, nxt = segments[i], segments[i + 1]
            gap = nxt.start_ms - cur.end_ms

            # We want `gap >= pad_ms`. If it's already there, leave it alone.
            if gap >= pad_ms:
                continue

            needed = pad_ms - gap  # positive: ms we have to claw back

            if allow_tail or video_duration_ms is None:
                shift = needed
            else:
                room = max(0, video_duration_ms - nxt.end_ms)
                shift = min(needed, room)

            if shift > 0:
                nxt.start_ms += shift
                nxt.end_ms += shift
                cur.force_fade = True
                nxt.force_fade = True

            # Si el shift no basta (tail-extension off y sin room),
            # recortamos SOLO silencio residual + un máximo del 15% de
            # cola. NO aplicamos fade_out aquí: el mixer aplicará el
            # force_crossfade (250 ms) al ver force_fade=True. Hacer
            # fade-out aquí + fade-out en el mixer = doble fade → la
            # cola queda muteada y el boundary salta 15-20 dB. Dejar
            # el corte seco permite al mixer controlar la totalidad
            # del fade en una única aplicación.
            still_needed = needed - shift
            if still_needed > 0:
                cur_len = len(cur.audio)
                max_trim = max(0, int(cur_len * 0.15))
                trim = min(still_needed, max_trim)
                if trim > 0:
                    cur.audio = cur.audio[: cur_len - trim]
                    cur.end_ms = cur.start_ms + len(cur.audio)
                    cur.force_fade = True
                    nxt.force_fade = True

    def _close_artificial_gaps(
        self,
        segments: list[TtsSegment],
        planned: list,
    ) -> None:
        """Pull the next phrase earlier over artificial SRT gaps only.

        When the aligner flagged a boundary as artificial (speaker kept
        talking across the SRT split), we shift the next segment left
        until either:

        * the gap closes to ``inter_phrase_pad_ms`` (target), or
        * we've reclaimed at most ``artificial_gap_to_next_ms`` (never
          eat into real silence).

        Real pauses — boundaries where the aligner recorded
        ``artificial_gap_to_next_ms == 0`` — are left alone so lip/face
        sync with the speaker doesn't drift.

        This pass runs AFTER ``_resolve_overlaps``, so its shift amount
        is computed against the current (possibly already-nudged)
        start/end times.
        """
        pad_ms = int(getattr(self.cfg, "inter_phrase_pad_ms", 0))
        min_gap = max(pad_ms, 1)

        for i in range(len(segments) - 1):
            cur, nxt = segments[i], segments[i + 1]
            # Prefer the segment's own planned index — some blocks can be
            # skipped during synthesis (empty text), so positional
            # mapping would misalign metadata vs audio.
            pidx = cur.planned_idx
            if pidx < 0 or pidx >= len(planned):
                continue
            artificial = getattr(
                planned[pidx], "artificial_gap_to_next_ms", 0,
            )
            if artificial < 300:  # not worth closing — avoids jitter
                continue

            gap = nxt.start_ms - cur.end_ms
            if gap <= min_gap:
                continue  # already tight

            shift_back = min(gap - min_gap, artificial)
            if shift_back <= 0:
                continue

            nxt.start_ms -= shift_back
            nxt.end_ms -= shift_back
            cur.force_fade = True
            nxt.force_fade = True

    def _compact_trailing_silence(self, segments: list) -> None:
        """Pull later phrases earlier to close silence left by short TTS.

        Context: with speech-anchored slots (nivel 3 dub track) the slot
        duration = real speaker talk time. The ES TTS often finishes before
        the slot is over — if we leave the next phrase anchored to its
        original start, the dub has audible silence inside the speaker's
        continuous speech. We shift later phrases backwards while keeping
        a minimum inter-phrase gap so the delivery still sounds natural.

        Only runs when ``compact_trailing_silence`` is on. Threshold and
        min gap come from config. Cascades — if phrase i+1 is pulled back,
        phrase i+2 recomputes relative to the new i+1 end.
        """
        threshold = self.cfg.compact_trailing_silence_threshold_ms
        min_gap = self.cfg.compact_min_gap_ms

        for i in range(len(segments) - 1):
            cur, nxt = segments[i], segments[i + 1]
            gap = nxt.start_ms - cur.end_ms
            if gap <= threshold:
                continue
            # Pull nxt back but preserve min_gap for breath.
            shift_back = gap - min_gap
            if shift_back <= 0:
                continue
            nxt.start_ms -= shift_back
            nxt.end_ms -= shift_back

    def _compact_synthetic_silence(
        self,
        segments: list,
        words_index,
    ) -> None:
        """Close synthetic gaps without drifting away from the EN anchor.

        Context:

        * XTTS clones the reference's rhythm. When the reference speaks
          faster than the EN coach, ES phrases finish before their SRT
          slot ends → trailing silence that doesn't exist in the
          original audio.
        * Naïvely pulling the next phrase left fixes the silence but
          cascades: every subsequent phrase also moves left, and by
          phrase 30 the dub can be 10+ seconds ahead of the speaker's
          lips.

        This pass closes gaps **only** when:

        1. The gap exceeds ``compact_synthetic_silence_threshold_ms``
           (avoid touching natural short pauses).
        2. ``words.json`` confirms the speaker was still talking — at
           least ``compact_synthetic_min_speech_ratio`` of the gap is
           covered by WhisperX word spans.
        3. The shift doesn't move ``nxt.start_ms`` more than
           ``compact_synthetic_max_drift_ms`` before its
           ``original_start_ms`` (lip-sync rail).

        When #1 or #2 fails → the gap is a real pause of the coach,
        leave it intact (lip sync preserved even if a silence remains).
        When #3 would trip → clamp the shift to what the rail allows,
        even if that means the gap only partially closes.

        Real pauses and drift-capped boundaries are logged so the QA
        pass can surface them. ``force_fade`` is only set when an
        actual shift happened, so the mixer's stronger crossfade
        doesn't fire on untouched boundaries.
        """
        threshold = self.cfg.compact_synthetic_silence_threshold_ms
        min_gap = self.cfg.compact_synthetic_min_gap_ms
        min_ratio = self.cfg.compact_synthetic_min_speech_ratio
        max_drift = self.cfg.compact_synthetic_max_drift_ms

        closed = 0
        preserved = 0
        capped = 0

        for i in range(len(segments) - 1):
            cur, nxt = segments[i], segments[i + 1]
            gap = nxt.start_ms - cur.end_ms
            if gap <= threshold:
                continue

            speech_ms = words_index.speech_coverage_ms(
                cur.end_ms, nxt.start_ms,
            )
            ratio = speech_ms / gap if gap > 0 else 0.0
            if ratio < min_ratio:
                preserved += 1
                logger.debug(
                    "[compact] preserve gap @%dms (dur=%dms, speech=%.0f%%)",
                    cur.end_ms, gap, ratio * 100,
                )
                continue

            wanted_shift = gap - min_gap
            if wanted_shift <= 0:
                continue

            # Lip-sync rail: nxt.start_ms was born at original_start_ms;
            # earlier compact/overlap passes may have nudged it already.
            # Compute the most we can pull it left without putting the
            # phrase further than ``max_drift`` before its anchor.
            if nxt.original_start_ms >= 0:
                floor_start = nxt.original_start_ms - max_drift
                max_allowed_shift = max(0, nxt.start_ms - floor_start)
                actual_shift = min(wanted_shift, max_allowed_shift)
            else:
                actual_shift = wanted_shift

            if actual_shift <= 0:
                # Lip-sync rail fully blocks this shift. Gap stays.
                capped += 1
                logger.debug(
                    "[compact] cap-blocked @%dms (gap=%dms, nxt at anchor)",
                    cur.end_ms, gap,
                )
                continue

            nxt.start_ms -= actual_shift
            nxt.end_ms -= actual_shift
            cur.force_fade = True
            nxt.force_fade = True
            if actual_shift < wanted_shift:
                capped += 1
                logger.debug(
                    "[compact] partial @%dms (wanted %dms, shifted %dms, "
                    "drift from anchor %dms)",
                    cur.end_ms, wanted_shift, actual_shift,
                    nxt.original_start_ms - nxt.start_ms,
                )
            else:
                closed += 1
                logger.debug(
                    "[compact] closed @%dms (gap %dms → %dms)",
                    cur.end_ms, gap, gap - actual_shift,
                )

        if closed or preserved or capped:
            logger.info(
                "Synthetic-silence pass: %d closed, %d capped by "
                "lip-sync rail, %d real pause(s) preserved",
                closed, capped, preserved,
            )

    # ------------------------------------------------------------------
    # QA (post-mux, non-blocking)
    # ------------------------------------------------------------------

    def _run_qa(
        self,
        video_path: Path,
        tts_segments: list[TtsSegment],
        final_mixed: AudioSegment,
    ) -> None:
        """Analyze the dub and write ``{base}.dub-qa.json``.

        Both boundary metrics AND MOS run on the **TTS-only** timeline
        (voice concatenated with silence at its timeline positions) — not
        on the final mix. Two reasons:

        * Background audio masks RMS / spectral-centroid jumps between
          phrases; the boundary report would under-report real issues.
        * UTMOS was trained on clean speech. Feeding it a ducked mix
          (voice + background + ambient) drags the predicted MOS into
          1.x territory regardless of actual voice quality — that's
          what dragged the last run to 1.28.

        The sidecar layout is deliberately flat so the frontend can map
        each boundary to its ``timestamp_ms`` directly, and the overall
        verdict can be rendered without a second round-trip.
        """
        from .qa import analyze_boundaries, score_mos

        base = video_path.with_suffix("")
        sidecar_path = base.parent / f"{base.name}.dub-qa.json"

        # ----- TTS-only buffer (numpy, for boundary metrics) -----
        tts_samples, tts_sr = self._build_tts_only_samples(
            tts_segments, len(final_mixed),
        )
        boundaries = self._compute_boundaries(tts_segments)
        boundary_report = analyze_boundaries(
            tts_samples, tts_sr, boundaries,
        ) if tts_samples is not None else None

        # ----- TTS-only wav (for UTMOS) -----
        mos = None
        mos_wav: Optional[Path] = None
        try:
            mos_wav = base.parent / f"{base.name}_QA_TMP.wav"
            self._export_tts_only_wav(
                tts_segments, len(final_mixed), mos_wav,
            )
            if mos_wav.exists():
                mos = score_mos(mos_wav)
        except Exception:
            logger.exception("MOS export/score failed")
        finally:
            if mos_wav is not None and mos_wav.exists():
                try:
                    mos_wav.unlink()
                except OSError:
                    pass

        payload = {
            "video": str(video_path),
            "total_tts_segments": len(tts_segments),
            "boundaries": boundary_report.to_dict() if boundary_report else None,
            "mos": mos.to_dict() if mos else None,
            "verdict": _compute_verdict(boundary_report, mos),
        }
        try:
            sidecar_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("QA sidecar written: %s", sidecar_path.name)
        except OSError:
            logger.exception("Could not write QA sidecar")

    @staticmethod
    def _build_tts_only_samples(
        tts_segments: list[TtsSegment], timeline_ms: int,
    ) -> tuple[Optional[object], int]:
        """Return a mono float32 array with TTS on silence at original ms.

        Built from pydub ``AudioSegment`` objects → convert to numpy at
        the shared frame rate. On any failure returns ``(None, 0)`` so
        the QA caller skips boundary analysis.
        """
        try:
            import numpy as np
        except ImportError:
            return None, 0
        if not tts_segments:
            return np.zeros(1, dtype=np.float32), 16000

        sr = tts_segments[0].audio.frame_rate
        total_samples = int(timeline_ms * sr / 1000) + 1
        buf = np.zeros(total_samples, dtype=np.float32)

        for seg in tts_segments:
            if seg.audio.frame_rate != sr:
                # Skip mismatched segments; boundary report is robust to
                # missing data (analyzer returns None for that boundary).
                continue
            arr = np.array(
                seg.audio.get_array_of_samples(), dtype=np.float32,
            )
            if seg.audio.channels > 1:
                arr = arr.reshape(-1, seg.audio.channels).mean(axis=1)
            # pydub samples are int16/int32; normalise to [-1, 1].
            max_val = float(1 << (8 * seg.audio.sample_width - 1))
            arr = arr / max_val
            start = int(seg.start_ms * sr / 1000)
            end = min(start + len(arr), total_samples)
            if end > start:
                buf[start:end] = arr[: end - start]
        return buf, sr

    @staticmethod
    def _export_tts_only_wav(
        tts_segments: list[TtsSegment],
        timeline_ms: int,  # kept for API compatibility; unused now
        output: Path,
    ) -> None:
        """Render a wav with just the concatenated TTS phrases.

        UTMOS22 was trained on ~5-10 s clips of continuous speech. Feeding
        it a 160-s track where 130 s are silence (the SRT-anchored timeline
        has long gaps between coach utterances) makes the model collapse
        onto its silence prior → predicted MOS around 1.3-1.4 regardless
        of voice quality. Concatenating the phrase audio gives UTMOS a
        continuous-speech clip that reflects the TTS it's supposed to
        score, not the silence between lines.
        """
        del timeline_ms  # retained for signature compatibility
        if not tts_segments:
            return
        sr = tts_segments[0].audio.frame_rate
        channels = tts_segments[0].audio.channels
        sample_width = tts_segments[0].audio.sample_width
        ordered = sorted(tts_segments, key=lambda s: s.start_ms)
        # 100 ms of silence between phrases mirrors the natural inter-
        # phrase pad the mixer applies; without it the concatenation
        # sounds glued and UTMOS marks it as unnatural. With a small
        # pad UTMOS sees "coach talking with short breaths", which is
        # what we want to be measured against.
        gap = AudioSegment.silent(duration=100, frame_rate=sr)
        if channels > 1 or sample_width != 2:
            gap = gap.set_channels(channels).set_sample_width(sample_width)
        pieces: list[AudioSegment] = []
        for seg in ordered:
            if len(seg.audio) == 0:
                continue
            if pieces:
                pieces.append(gap)
            pieces.append(seg.audio)
        if not pieces:
            return
        track = pieces[0]
        for p in pieces[1:]:
            track += p
        track.export(str(output), format="wav")

    @staticmethod
    def _compute_boundaries(
        tts_segments: list[TtsSegment],
    ) -> list[tuple[int, int]]:
        """Return list of (end_prev_ms, start_next_ms) in timeline order."""
        if len(tts_segments) < 2:
            return []
        ordered = sorted(tts_segments, key=lambda s: s.start_ms)
        return [
            (ordered[i].end_ms, ordered[i + 1].start_ms)
            for i in range(len(ordered) - 1)
        ]

    # ------------------------------------------------------------------
    # FFmpeg muxing
    # ------------------------------------------------------------------

    @staticmethod
    def _mux_video(
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> None:
        """Mux the original video + new ES audio + any SRTs.

        Result: one MKV with only the ES dub audio track (EN discarded — the
        original file keeps the EN track). Both subtitle tracks (EN + ES)
        are kept if present so the user can still switch subs in the player.
        """
        base = video_path.with_suffix("")
        # ES subtitle track for the MKV: prefer the literal ES srt (good for
        # reading), never the dubbing-adapted one (.dub.es.srt), which is
        # shortened for iso-synchrony and reads awkwardly as subtitles.
        srt_es = None
        for _sfx in (".es.srt", ".ES.srt", "_ES.srt", "_ESP.srt", "_ESP_DUB.srt"):
            candidate = base.parent / f"{base.name}{_sfx}"
            if candidate.exists():
                srt_es = candidate
                break
        srt_en = None
        for _sfx in (".en.srt", ".EN.srt", "_EN.srt", ".srt"):
            candidate = base.parent / f"{base.name}{_sfx}"
            if candidate.exists() and candidate != srt_es:
                srt_en = candidate
                break

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(audio_path),
        ]
        if srt_en:
            cmd += ["-i", str(srt_en)]
        if srt_es:
            cmd += ["-i", str(srt_es)]

        # Map: video + ONLY the new ES audio (EN original audio discarded by
        # design — the dubbed MKV is meant to be watched in Spanish; the
        # original EN track lives in the source file).
        cmd += [
            "-map", "0:v",
            "-map", "1:a",
        ]
        sub_inputs = []
        if srt_en:
            sub_inputs.append(("2", "eng", "English"))
        if srt_es:
            idx = "3" if srt_en else "2"
            sub_inputs.append((idx, "spa", "Español"))
        for idx, _lang, _title in sub_inputs:
            cmd += ["-map", f"{idx}:0"]

        # R10: `-fflags +shortest` NO. El pipeline ya garantiza que:
        #  (a) `mixed` >= video_duration_ms (padding con silencio si
        #      Demucs dio stem corto) — protege contra truncado abrupto,
        #  (b) `mixed` <= video_duration_ms + tail_extension_max_ms
        #      (truncate explícito con fade_out) — protege de que el
        #      vídeo se alargue con el último frame congelado durante
        #      segundos como reportó el usuario en E02.
        # Con (a)+(b) garantizados, el mux stream-copy da MKV con
        # `video_track_len == audio_track_len` dentro del margen
        # `tail_extension`. No necesitamos `-shortest` (que cortaría
        # en el MIN de ambos, arriesgando perder la cola ES) ni
        # `-fflags +shortest` (mismo problema).
        cmd += [
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-c:s", "srt",
            "-metadata:s:a:0", "language=spa",
            "-metadata:s:a:0", "title=Doblaje (ES)",
            "-disposition:a:0", "default",
        ]
        for sub_i, (_idx, lang, title) in enumerate(sub_inputs):
            cmd += [
                f"-metadata:s:s:{sub_i}", f"language={lang}",
                f"-metadata:s:s:{sub_i}", f"title={title}",
            ]

        cmd += [str(output_path)]
        # Timeout 30 min: stream-copy de vídeo + re-encode del audio ES
        # de ~40 min no debe tardar >5 min en hardware normal. 30 es un
        # tope defensivo para no colgar el worker si ffmpeg entra en
        # loop por un codec raro.
        subprocess.run(cmd, check=True, timeout=1800)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_consecutive_blocks(
        blocks: list[SrtBlock], max_gap_ms: int, max_chars: int = 200,
    ) -> list[SrtBlock]:
        """Group adjacent SRT blocks whose gap is below ``max_gap_ms``.

        Keeps start from the first block and end from the last so the merged
        block stays anchored to the original video timestamps. Produces a
        single TTS call per merged block → continuous prosody, no mid-speech
        resets. Blocks separated by a real pause (> max_gap_ms) stay split.
        Bounded by ``max_chars`` so the merged text never forces XTTS
        to split internally (which resets prosody and risks truncation).
        """
        if not blocks:
            return []
        merged: list[SrtBlock] = []
        cur = SrtBlock(
            index=blocks[0].index,
            start_ms=blocks[0].start_ms,
            end_ms=blocks[0].end_ms,
            text=blocks[0].text,
        )
        for nxt in blocks[1:]:
            gap = nxt.start_ms - cur.end_ms
            combined_len = len(cur.text) + 1 + len(nxt.text)
            if gap <= max_gap_ms and combined_len <= max_chars:
                cur = SrtBlock(
                    index=cur.index,
                    start_ms=cur.start_ms,
                    end_ms=nxt.end_ms,
                    text=f"{cur.text} {nxt.text}".strip(),
                )
            else:
                merged.append(cur)
                cur = SrtBlock(
                    index=nxt.index,
                    start_ms=nxt.start_ms,
                    end_ms=nxt.end_ms,
                    text=nxt.text,
                )
        merged.append(cur)
        return merged

    @staticmethod
    def _probe_video_duration_ms(video_path: Path) -> int | None:
        """Return video duration in ms via ffprobe, or None on failure."""
        try:
            out = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True, text=True, timeout=10, check=True,
            )
            return int(float(out.stdout.strip()) * 1000)
        except Exception as exc:
            logger.warning("ffprobe failed for %s: %s", video_path, exc)
            return None

    @staticmethod
    def _release_torch_vram() -> None:
        """Free CUDA tensors / cached blocks held by torch in this process.

        Demucs is invoked as a subprocess so the heavy allocations live
        and die there, but the parent inherits torch state from earlier
        imports (silero VAD, MOS scorer, etc.). Calling
        ``empty_cache()`` after fase 0 is the cheapest way to make sure
        s2.cpp's mmap of the 5 GB GGUF doesn't lose to fragmentation on
        a 6 GB card.
        """
        try:
            import torch  # type: ignore
        except ImportError:
            return
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as exc:  # noqa: BLE001
            logger.debug("torch VRAM release skipped: %s", exc)

    def _report(self, step: int, total: int, message: str) -> None:
        logger.info("[%d/%d] %s", step, total, message)
        if self._progress_cb:
            self._progress_cb(step, total, message)

    @staticmethod
    def _cleanup(*paths: Path) -> None:
        for p in paths:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


def _compute_verdict(boundary_report, mos) -> dict:
    """Combine boundary + MOS signals into a single green/amber/red verdict.

    Boundary-first logic — boundaries directly reflect listener-perceived
    cuts, while UTMOS scores dubbed-ducked audio unfairly low (trained on
    clean studio speech). MOS is a tiebreaker, not a gate.

    * **red** — at least one hard boundary (real listener "this is broken")
    * **amber** — only warnings (potentially audible, never jarring)
    * **green** — zero boundaries flagged

    MOS only nudges the severity up when it's *very* poor (<2.0) and we
    were already amber — otherwise a capítulo with zero boundaries but
    MOS 2.1 (normal for TTS+ducking) stays green where it belongs.
    """
    hard = boundary_report.hard_cuts if boundary_report else 0
    warn = boundary_report.warnings if boundary_report else 0
    mos_score = mos.score if mos else None

    if hard > 0:
        level = "red"
    elif warn > 0:
        level = "amber"
    else:
        level = "green"

    # Only let MOS lift an amber to red in genuinely broken territory.
    if mos_score is not None and mos_score < 2.0 and level == "amber":
        level = "red"

    return {
        "level": level,
        "mos": mos_score,
        "hard_cuts": hard,
        "warnings": warn,
    }
