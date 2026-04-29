"""Main orchestration pipeline: transcribe, align, filter, fix, write, validate."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Optional

from .config import SubtitleConfig, TranscriptionConfig, EXTENSIONES, DEFAULT_HOTWORDS
from .hallucination_filter import HallucinationFilter
from .timestamp_fixer import TimestampFixer
from .writer import SubtitleWriter
from .validator import SubtitleValidator

log = logging.getLogger("subtitler")


class SubtitlePipeline:
    """Orchestrator: transcribe -> align -> filter -> fix -> write -> validate."""

    def __init__(
        self,
        transcription_config: TranscriptionConfig,
        subtitle_config: SubtitleConfig,
    ) -> None:
        self.t_config = transcription_config
        self.s_config = subtitle_config

        self.hallucination_filter = HallucinationFilter(
            subtitle_config, initial_prompt=transcription_config.initial_prompt
        )
        self.timestamp_fixer = TimestampFixer(subtitle_config)
        self.writer = SubtitleWriter(subtitle_config, self.timestamp_fixer)
        self.validator = SubtitleValidator(subtitle_config)

        self._model = None
        self._align_model = None
        self._align_metadata = None

    def load_models(self) -> None:
        """Load WhisperX ASR and alignment models."""
        import torch
        import whisperx

        device = self.t_config.device
        log.info("Loading WhisperX model '%s' on %s...", self.t_config.model_name, device)

        hotwords = self.t_config.hotwords if self.t_config.hotwords is not None else DEFAULT_HOTWORDS
        asr_options = {
            "initial_prompt": self.t_config.initial_prompt,
            "hotwords": hotwords,
            "beam_size": self.t_config.beam_size,
            "condition_on_previous_text": self.t_config.condition_on_previous_text,
        }

        self._model = whisperx.load_model(
            self.t_config.model_name,
            device,
            compute_type=self.t_config.compute_type,
            asr_options=asr_options,
            vad_options={
                "vad_onset": self.t_config.vad_onset,
                "vad_offset": self.t_config.vad_offset,
            },
        )

        log.info("Loading alignment model for '%s'...", self.t_config.language)
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=self.t_config.language, device=device
        )
        log.info("Models loaded successfully.")

    def process_file(self, video_path: Path, force: bool = False) -> Optional[Path]:
        """Process a single video file. Returns the output SRT path, or None if skipped."""
        import torch
        import whisperx

        lang_code = self.t_config.language.lower()
        output_srt = video_path.with_name(video_path.stem + f".{lang_code}.srt")

        if output_srt.exists() and not force:
            log.debug("Skipping (SRT exists): %s", video_path.name)
            return None
        if output_srt.exists() and force:
            log.info("Force mode: overwriting existing SRT for %s", video_path.name)

        if self._model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        log.info("Processing: %s", video_path.name)

        # 1. Load and denoise audio
        log.info("  Loading audio...")
        audio = whisperx.load_audio(str(video_path))
        audio_duration = len(audio) / 16000.0  # WhisperX loads at 16kHz

        audio = self._denoise_audio(audio)

        # 2. Transcribe (auto-retry with halved batch_size on CUDA OOM)
        log.info("  Transcribing...")

        result = self._transcribe_with_oom_retry(audio)
        segments = result.get("segments", [])
        log.info("  Transcription produced %d raw segments.", len(segments))

        if not segments:
            log.warning("  No segments produced. Skipping.")
            self._cleanup(audio, result)
            return None

        # 3. Gap-fill pass — re-transcribe large gaps with boosted audio
        gaps = self._find_gaps(segments, audio_duration)
        if gaps:
            log.info("  Found %d gaps (>%.0fs), running gap-fill pass...",
                     len(gaps), self.t_config.gap_fill_min_gap)
            gap_segments = self._transcribe_gaps(gaps, audio)
            if gap_segments:
                segments = sorted(segments + gap_segments, key=lambda s: s.get("start", 0))
                log.info("  Recovered %d segments from gaps.", len(gap_segments))

        # 4. Align (per-segment with fallback)
        log.info("  Aligning timestamps...")
        segments = self._align_segments(segments, audio)

        # 5. Hallucination filtering
        log.info("  Filtering hallucinations...")
        segments = self.hallucination_filter.filter_all(segments, audio_path=video_path)
        log.info("  %d segments after filtering.", len(segments))

        if not segments:
            log.warning("  All segments filtered out. Skipping.")
            self._cleanup(audio, result)
            return None

        # 6. Flatten to words
        all_words = self._flatten_words(segments)
        if not all_words:
            log.warning("  No words extracted. Skipping.")
            self._cleanup(audio, result)
            return None

        # 7. Write SRT (includes timestamp fixing)
        log.info("  Writing SRT: %s", output_srt.name)
        subtitles = self.writer.write_srt(all_words, output_srt)

        # 7b. Persist word-timestamps alongside the SRT. Required by the
        # dubbing nivel-3 pipeline (dub_segmenter) to re-segment speech by
        # real pauses instead of inheriting reading-oriented SRT slots.
        self._write_words_json(all_words, video_path)

        # 7c. Optional OpenAI post-process — cleans WhisperX artifacts
        # (syllable duplication, broken boundaries) without touching timing.
        if self.t_config.postprocess_openai:
            subtitles = self._postprocess_srt(subtitles, output_srt)

        # 8. Validate
        self.validator.validate(subtitles, audio_duration)

        # 9. Cleanup GPU memory
        self._cleanup(audio, result)

        return output_srt

    def _align_segments(self, segments: list[dict], audio) -> list[dict]:
        """Align segments individually with per-segment fallback."""
        import whisperx

        device = self.t_config.device
        aligned_segments: list[dict] = []

        for seg in segments:
            try:
                result_aligned = whisperx.align(
                    [seg],
                    self._align_model,
                    self._align_metadata,
                    audio,
                    device=device,
                    return_char_alignments=False,
                )
                aligned = result_aligned.get("segments", [])
                if aligned and aligned[0].get("words"):
                    aligned_segments.extend(aligned)
                else:
                    # Alignment returned no words - use synthetic fallback
                    aligned_segments.append(self._synthetic_word_timing(seg))
            except Exception as e:
                log.debug("  Alignment failed for segment at %.1fs: %s", seg.get("start", 0), e)
                aligned_segments.append(self._synthetic_word_timing(seg))

        return aligned_segments

    def _synthetic_word_timing(self, segment: dict) -> dict:
        """Create synthetic word-level timing by distributing duration proportionally by character count."""
        text = segment.get("text", "").strip()
        start = segment.get("start", 0.0)
        end = segment.get("end", start + 1.0)
        duration = end - start

        words_text = text.split()
        if not words_text:
            return segment

        total_chars = sum(len(w) for w in words_text)
        if total_chars == 0:
            total_chars = len(words_text)

        synth_words: list[dict] = []
        cursor = start
        for wt in words_text:
            word_duration = duration * (len(wt) / total_chars)
            synth_words.append({
                "word": wt,
                "start": cursor,
                "end": cursor + word_duration,
                "score": 0.45,  # Below-neutral confidence for synthetic timing
            })
            cursor += word_duration

        seg_copy = dict(segment)
        seg_copy["words"] = synth_words
        return seg_copy

    def _find_gaps(self, segments: list[dict], audio_duration: float) -> list[tuple[float, float]]:
        """Return list of (start, end) gaps larger than gap_fill_min_gap."""
        min_gap = self.t_config.gap_fill_min_gap
        gaps: list[tuple[float, float]] = []

        sorted_segs = sorted(segments, key=lambda s: s.get("start", 0))

        # Gap before first segment
        if sorted_segs and sorted_segs[0].get("start", 0) > min_gap:
            gaps.append((0.0, sorted_segs[0]["start"]))

        # Gaps between segments
        for i in range(len(sorted_segs) - 1):
            end_cur = sorted_segs[i].get("end", 0)
            start_next = sorted_segs[i + 1].get("start", 0)
            if start_next - end_cur > min_gap:
                gaps.append((end_cur, start_next))

        # Gap after last segment
        if sorted_segs:
            last_end = sorted_segs[-1].get("end", 0)
            if audio_duration - last_end > min_gap:
                gaps.append((last_end, audio_duration))

        return gaps

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """Quick check for obvious hallucinated content from gap-fill."""
        import re
        t = text.strip().lower()
        if not t:
            return True
        # URLs, repeated .com, email-like garbage
        if re.search(r'\.com|\.net|\.org|www\.|http|@', t):
            return True
        # Repeated short tokens (e.g. "com com com", "na na na")
        words = t.split()
        if len(words) >= 3:
            unique = set(words)
            if len(unique) <= 2:
                return True
        # Music/sound markers
        if re.search(r'♪|♫|\[music\]|\[applause\]|\[laughter\]', t):
            return True
        return False

    def _transcribe_gaps(self, gaps: list[tuple[float, float]], audio) -> list[dict]:
        """Re-transcribe gap regions using faster-whisper directly (no VAD).

        Only processes gaps where the audio has enough energy to be speech.
        Filters out obvious hallucinations before returning.
        """
        import numpy as np
        import io
        import scipy.io.wavfile as wav_io

        recovered: list[dict] = []
        boost_db = self.t_config.gap_fill_audio_boost_db
        boost_factor = 10 ** (boost_db / 20.0)
        sample_rate = 16000

        # Reuse the WhisperModel already loaded inside WhisperX pipeline
        fw_model = self._model.model  # faster_whisper.WhisperModel

        hotwords = self.t_config.hotwords if self.t_config.hotwords is not None else DEFAULT_HOTWORDS

        for gap_start, gap_end in gaps:
            start_sample = int(gap_start * sample_rate)
            end_sample = int(gap_end * sample_rate)
            chunk = audio[start_sample:end_sample].copy()

            # Check if this gap has enough audio energy to be speech
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_db = 20 * np.log10(rms + 1e-10)
            if rms_db < -45.0:
                log.debug("  Gap %.1f-%.1f too quiet (%.1f dB), skipping",
                          gap_start, gap_end, rms_db)
                continue

            # Boost audio volume
            chunk = np.clip(chunk * boost_factor, -1.0, 1.0).astype(np.float32)

            # faster-whisper needs a file path or BinaryIO — write to memory buffer
            chunk_int16 = (chunk * 32767).astype(np.int16)
            buf = io.BytesIO()
            wav_io.write(buf, sample_rate, chunk_int16)
            buf.seek(0)

            transcribe_kwargs = dict(
                language=self.t_config.language,
                beam_size=1,  # conservative — model already occupies VRAM during gap-fill
                initial_prompt=self.t_config.initial_prompt,
                hotwords=hotwords,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-0.5,
                word_timestamps=True,
            )
            try:
                segments_iter, info = fw_model.transcribe(buf, **transcribe_kwargs)
                segments_iter = list(segments_iter)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                buf.seek(0)
                try:
                    transcribe_kwargs["beam_size"] = 1
                    segments_iter, info = fw_model.transcribe(buf, **transcribe_kwargs)
                    segments_iter = list(segments_iter)
                    log.warning("  Gap-fill OOM — retried with beam_size=1 for gap %.1f-%.1f",
                                gap_start, gap_end)
                except RuntimeError:
                    log.warning("  Gap-fill OOM on retry — skipping gap %.1f-%.1f",
                                gap_start, gap_end)
                    continue

            for seg in segments_iter:
                text = seg.text.strip()

                # Skip low-confidence or hallucinated segments
                if seg.avg_logprob < -0.7:
                    log.debug("  Gap-fill dropped low-conf (%.2f): %r",
                              seg.avg_logprob, text[:60])
                    continue
                if seg.no_speech_prob > 0.5:
                    log.debug("  Gap-fill dropped high no-speech (%.2f): %r",
                              seg.no_speech_prob, text[:60])
                    continue
                if self._is_hallucination(text):
                    log.debug("  Gap-fill dropped hallucination: %r", text[:60])
                    continue

                words = []
                if seg.words:
                    for w in seg.words:
                        words.append({
                            "word": w.word.strip(),
                            "start": w.start + gap_start,
                            "end": w.end + gap_start,
                            "score": w.probability,
                        })
                recovered.append({
                    "text": text,
                    "start": seg.start + gap_start,
                    "end": seg.end + gap_start,
                    "words": words,
                })

        return recovered

    def _postprocess_srt(self, subtitles: list[dict], srt_path: Path) -> list[dict]:
        """Run OpenAI cleanup and rewrite the SRT file in place.

        Preserves timestamps and block count. On any failure returns the
        original subtitles so the SRT on disk is always valid.
        """
        from .srt_postprocess import SrtPostprocessor
        from .utils import format_timestamp

        try:
            pp = SrtPostprocessor(
                api_key=self.t_config.postprocess_api_key,
                model=self.t_config.postprocess_model,
            )
        except ValueError as exc:
            log.warning("  Postprocess disabled: %s", exc)
            return subtitles

        log.info("  Postprocessing SRT with OpenAI (%s)...", self.t_config.postprocess_model)
        try:
            cleaned = pp.clean_subtitles(subtitles)
        except Exception as exc:  # noqa: BLE001
            log.warning("  Postprocess failed: %s — keeping original SRT", exc)
            return subtitles

        if len(cleaned) != len(subtitles):
            log.warning(
                "  Postprocess returned %d blocks vs %d — keeping original",
                len(cleaned), len(subtitles),
            )
            return subtitles

        changed = sum(
            1 for a, b in zip(subtitles, cleaned)
            if a.get("text") != b.get("text")
        )
        log.info("  Postprocess cleaned %d/%d blocks", changed, len(cleaned))

        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, sub in enumerate(cleaned, 1):
                f.write(
                    f"{idx}\n"
                    f"{format_timestamp(sub['start'])} --> {format_timestamp(sub['end'])}\n"
                    f"{sub['text']}\n\n"
                )
        return cleaned

    def _write_words_json(self, words: list[dict], video_path: Path) -> None:
        """Dump word-level timestamps as <video>.words.json.

        Enables the dubbing nivel-3 pipeline to re-segment speech from the
        original audio rhythm rather than the reading-oriented SRT blocks.
        """
        import json

        out = video_path.with_name(video_path.stem + ".words.json")
        payload = [
            {
                "word": str(w.get("word", "")).strip(),
                "start": round(float(w.get("start", 0.0)), 3),
                "end": round(float(w.get("end", 0.0)), 3),
                "score": round(float(w.get("score", 0.0)), 3),
            }
            for w in words
            if str(w.get("word", "")).strip()
        ]
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        log.info("  Wrote %d word-timestamps to %s", len(payload), out.name)

    def _flatten_words(self, segments: list[dict]) -> list[dict]:
        """Extract all words from segments, with fallback for segments without word-level data."""
        all_words: list[dict] = []
        for seg in segments:
            if "words" in seg and seg["words"]:
                all_words.extend(seg["words"])
            else:
                # Segment without words - create a single word entry
                all_words.append({
                    "word": seg.get("text", "").strip(),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "score": 0.45,
                })
        return all_words

    def _transcribe_with_oom_retry(self, audio) -> dict:
        """Transcribe with automatic batch_size halving on CUDA OOM."""
        import torch

        batch_size = self.t_config.batch_size
        while batch_size >= 1:
            try:
                return self._model.transcribe(
                    audio,
                    batch_size=batch_size,
                    language=self.t_config.language,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size <= 1:
                    raise
                log.warning(
                    "  CUDA OOM with batch_size=%d — retrying with batch_size=%d",
                    batch_size,
                    batch_size // 2,
                )
                batch_size //= 2
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise RuntimeError("CUDA OOM even with batch_size=1")

    def _denoise_audio(self, audio):
        """Apply spectral noise reduction to improve VAD and transcription.

        Uses noisereduce (stationary noise reduction) which is effective
        against constant background noise like room echo, camera hiss,
        and white noise from microphones.
        """
        import numpy as np

        try:
            import noisereduce as nr
        except ImportError:
            log.warning("noisereduce not installed -- skipping audio denoising")
            return audio

        log.info("  Denoising audio (spectral noise reduction)...")
        # WhisperX audio is float32 at 16kHz
        denoised = nr.reduce_noise(
            y=audio,
            sr=16000,
            stationary=True,      # Optimized for constant background noise
            prop_decrease=0.75,   # Reduce noise by 75% (not 100% to avoid artifacts)
            n_fft=2048,
            n_std_thresh_stationary=1.5,
        )

        # Log SNR improvement
        orig_rms = float(np.sqrt(np.mean(audio ** 2)))
        dn_rms = float(np.sqrt(np.mean(denoised ** 2)))
        if orig_rms > 0:
            log.info("  Denoise: original RMS=%.4f, denoised RMS=%.4f (ratio=%.2f)",
                     orig_rms, dn_rms, dn_rms / orig_rms if orig_rms else 0)

        return denoised.astype(np.float32)

    def _cleanup(self, *objects) -> None:
        """Free GPU memory."""
        import torch
        for obj in objects:
            del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_directory(self, root_dir: Path, force: bool = False) -> None:
        """Walk directory tree and process all matching video files.

        Excluye subcarpetas de outputs (``doblajes``, ``elevenlabs``) — si no,
        WhisperX encuentra los .mkv doblados y les genera .en.srt espurios.

        Strict mode (since 2026-04-29): si CUALQUIER chapter falla, se
        lanza ``RuntimeError`` al terminar el bucle con la lista de
        chapters fallidos. Antes los errores se loggeaban y el job
        terminaba "completed" con sólo M/N SRTs reales en disco — el
        siguiente paso (translate / dubbing) procesaba parcialmente y
        el usuario no se enteraba hasta inspeccionar la Season.
        """
        _EXCLUDED_DIR_NAMES = {"doblajes", "elevenlabs"}
        video_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d.lower() not in _EXCLUDED_DIR_NAMES]
            for fname in sorted(filenames):
                if fname.lower().endswith(EXTENSIONES):
                    video_files.append(Path(dirpath) / fname)

        if not video_files:
            log.warning("No video files found in %s", root_dir)
            return

        log.info("Found %d video files in %s", len(video_files), root_dir)
        processed = 0
        skipped = 0
        failed: list[tuple[str, str]] = []  # (name, error)

        import torch

        for vpath in video_files:
            try:
                result = self.process_file(vpath, force=force)
                if result is None:
                    skipped += 1
                else:
                    processed += 1
            except Exception as e:
                failed.append((vpath.name, str(e)))
                log.error("Error processing %s: %s", vpath.name, e, exc_info=True)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        log.info(
            "Done. Processed: %d, Skipped (existing): %d, Errors: %d",
            processed, skipped, len(failed),
        )

        if failed:
            names = ", ".join(name for name, _ in failed[:5])
            if len(failed) > 5:
                names += " …"
            raise RuntimeError(
                f"Subtítulos incompletos: {processed}/{len(video_files)} OK, "
                f"{len(failed)} fallaron: {names}"
            )
