"""Mix background audio with TTS segments using professional ducking."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from pydub import AudioSegment

from ..config import DubbingConfig

logger = logging.getLogger(__name__)


@dataclass
class TtsSegment:
    """A single TTS audio chunk placed on the timeline.

    ``force_fade`` signals that the overlap resolver had to claw ms back
    at the *start* of this segment (or the previous one was trimmed for
    its sake). The mixer should apply a stronger crossfade on that
    boundary regardless of the natural gap — without it the listener
    hears a click where two independently-generated TTS renders meet.
    """

    audio: AudioSegment
    start_ms: int
    end_ms: int
    force_fade: bool = False
    planned_idx: int = -1  # index into the PlannedBlock list, -1 if unknown
    # Immutable anchor: the SRT-derived start the segment was born with,
    # before any overlap/compact pass shifted it. Post-hoc silence closers
    # use this to bound how far they can drag a segment away from the
    # original EN speech timestamp (lip-sync safety rail).
    original_start_ms: int = -1
    # Set by the pipeline's pairwise RMS leveling pass when this segment
    # (or the one before it) had an audible level jump even after
    # per-segment normalization. The mixer uses a longer crossfade here
    # than the regular ``force_crossfade_ms`` to mask the residual jump.
    rms_jump_boundary: bool = False


class AudioMixer:
    """Mix background + TTS voice with ducking.

    During TTS voice playback the background is reduced to
    ``ducking_bg_volume`` with fade transitions of ``ducking_fade_ms``.
    The TTS is boosted by ``ducking_fg_volume`` so it dominates clearly.
    """

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mix(
        self,
        background: AudioSegment,
        tts_segments: list[TtsSegment],
    ) -> AudioSegment:
        """Return the final mixed AudioSegment (background + ducked TTS)."""

        if not tts_segments:
            return background

        fg_db = _vol_to_db(self.cfg.ducking_fg_volume)
        ducking_db = _vol_to_db(self.cfg.ducking_bg_volume)
        fade_ms = self.cfg.ducking_fade_ms

        allow_tail = getattr(self.cfg, "allow_video_tail_extension", False)
        tail_max = getattr(self.cfg, "video_tail_extension_max_ms", 8000)

        bg_len = len(background)
        last_tts_end = max(
            (seg.start_ms + len(seg.audio) for seg in tts_segments), default=0,
        )

        if allow_tail and last_tts_end > bg_len:
            # Extend background with silence so trailing ES phrases remain
            # fully audible. Cap extension to avoid runaway (bad SRT → huge pad).
            pad_ms = min(last_tts_end - bg_len, tail_max)
            if last_tts_end - bg_len > tail_max:
                logger.warning(
                    "TTS exceeds video by %d ms but tail cap is %d ms; "
                    "trailing speech will be clipped",
                    last_tts_end - bg_len, tail_max,
                )
            background = background + AudioSegment.silent(
                duration=pad_ms, frame_rate=background.frame_rate,
            )
            bg_len = len(background)
            logger.info("Extended audio track by %d ms to fit ES tail", pad_ms)

        # Clip only phrases that still fall outside the (possibly extended) bg.
        clipped_segments: list[TtsSegment] = []
        for seg in tts_segments:
            if seg.start_ms >= bg_len:
                logger.warning(
                    "Dropping TTS starting at %d ms (past audio end %d ms)",
                    seg.start_ms, bg_len,
                )
                continue
            audio = seg.audio
            end_ms = seg.start_ms + len(audio)
            if end_ms > bg_len:
                trim_to = max(0, bg_len - seg.start_ms)
                if trim_to <= 0:
                    continue
                fade = min(60, trim_to // 4)
                audio = audio[:trim_to].fade_out(fade) if fade > 0 else audio[:trim_to]
            clipped_segments.append(TtsSegment(
                audio=audio, start_ms=seg.start_ms, end_ms=seg.start_ms + len(audio),
                force_fade=seg.force_fade, planned_idx=seg.planned_idx,
                original_start_ms=seg.original_start_ms,
            ))

        if not clipped_segments:
            return background

        # Build ducked background (concatenation approach — preserves length)
        ducked_bg = self._apply_ducking(background, clipped_segments, ducking_db, fade_ms)

        # Smooth inter-phrase transitions: when two consecutive TTS segments
        # are close together, apply a short fade-out on the tail of the first
        # and fade-in on the head of the second so the boundary stops sounding
        # like a hard cut between two independent renders.
        smoothed = self._apply_inter_phrase_fades(clipped_segments)

        # Overlay TTS on top of ducked background
        result = ducked_bg
        for seg in smoothed:
            if len(seg.audio) == 0:
                continue
            tts_boosted = seg.audio.apply_gain(fg_db)
            result = result.overlay(tts_boosted, position=seg.start_ms)

        return result

    def _apply_inter_phrase_fades(
        self, segments: list[TtsSegment],
    ) -> list[TtsSegment]:
        """Apply fade-in / fade-out at boundaries of close neighbours.

        Two triggers:

        * Natural-close: pair whose real gap is within
          ``inter_phrase_crossfade_max_gap_ms``. Uses the standard xfade
          length.
        * Force-fade: pair where ``_resolve_overlaps`` modified the
          timing to keep a pad. These get a **stronger** fade
          (``force_crossfade_ms``) because two independently-rendered
          TTS phrases butted up against each other need more smoothing
          than the natural overlap case.

        Segments shorter than ``2 * xfade`` fall back to a half-length
        fade so we don't skip smoothing on rapid-fire phrases (the old
        behaviour skipped entirely, which is exactly where the 0-ms
        click lived).
        """
        xfade = int(getattr(self.cfg, "inter_phrase_crossfade_ms", 0))
        force_xfade = int(getattr(
            self.cfg, "force_crossfade_ms", max(xfade, 180),
        ))
        # Third tier: boundaries flagged by the pipeline's pairwise RMS
        # leveling pass get a longer fade than ``force_xfade``. With
        # ElevenLabs a 10 dB jump survives the gain lift partially; a
        # 550 ms fade hides what's left. Falls back to force_xfade + 120
        # ms when the config entry is absent so nothing regresses for
        # XTTS users who never set it.
        rms_xfade = int(getattr(
            self.cfg, "rms_jump_crossfade_ms", force_xfade + 120,
        ))
        max_gap = int(getattr(self.cfg, "inter_phrase_crossfade_max_gap_ms", 0))
        if xfade <= 0 or len(segments) < 2:
            return segments

        sorted_segs = sorted(segments, key=lambda s: s.start_ms)
        n = len(sorted_segs)
        # For each boundary pick the fade length (0 = no fade).
        fade_in = [0] * n
        fade_out = [0] * n
        for i in range(n - 1):
            cur = sorted_segs[i]
            nxt = sorted_segs[i + 1]
            gap = nxt.start_ms - cur.end_ms
            rms_jump = bool(cur.rms_jump_boundary or nxt.rms_jump_boundary)
            forced = bool(cur.force_fade or nxt.force_fade)
            if rms_jump:
                length = rms_xfade
            elif forced:
                length = force_xfade
            elif gap <= max_gap:
                length = xfade
            else:
                continue
            fade_out[i] = max(fade_out[i], length)
            fade_in[i + 1] = max(fade_in[i + 1], length)

        out: list[TtsSegment] = []
        for i, seg in enumerate(sorted_segs):
            audio = seg.audio
            alen = len(audio)
            fi = fade_in[i]
            fo = fade_out[i]
            # Cap every fade so both fit the segment: the union shouldn't
            # exceed the audio length. Short phrases get shorter fades
            # instead of being skipped outright.
            if alen <= 0 or (fi == 0 and fo == 0):
                out.append(seg)
                continue
            # Cap al 20% de la frase (antes 50%): en frases cortas
            # (<1 s) un fade de 250 ms sobre 500 ms de audio muta la
            # mitad del contenido → la QA lo interpreta como "salto
            # RMS 15-20 dB" y auditivamente suena a palabra cortada.
            # 20% es suficiente para suavizar el borde sin deformar
            # la palabra.
            cap = max(1, alen // 5)
            fi = min(fi, cap)
            fo = min(fo, cap)
            if fi > 0:
                audio = audio.fade_in(fi)
            if fo > 0:
                audio = audio.fade_out(fo)
            out.append(TtsSegment(
                audio=audio,
                start_ms=seg.start_ms,
                end_ms=seg.start_ms + len(audio),
                force_fade=seg.force_fade,
                planned_idx=seg.planned_idx,
                original_start_ms=seg.original_start_ms,
            ))
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_ducking(
        self,
        bg: AudioSegment,
        segments: list[TtsSegment],
        ducking_db: float,
        fade_ms: int,
    ) -> AudioSegment:
        """Reduce background during TTS regions.

        Uses concatenation to preserve exact timeline length.
        The duck regions use apply_gain so the total length is unchanged.
        """
        sorted_segs = sorted(segments, key=lambda s: s.start_ms)
        sustain_gap = getattr(self.cfg, "ducking_sustain_gap_ms", 0)

        # Merge overlapping/adjacent TTS regions (with fade padding). Si
        # dos regiones están separadas por menos de sustain_gap, también
        # se mergean para mantener el background ducked durante huecos
        # breves (evita subida/bajada abrupta de volumen = "frenazo").
        regions: list[tuple[int, int]] = []
        for seg in sorted_segs:
            r_start = max(0, seg.start_ms - fade_ms)
            r_end = min(len(bg), seg.start_ms + len(seg.audio) + fade_ms)
            if r_end <= r_start:
                continue
            if regions and r_start - regions[-1][1] <= sustain_gap:
                regions[-1] = (regions[-1][0], max(regions[-1][1], r_end))
            else:
                regions.append((r_start, r_end))

        if not regions:
            return bg

        # Assemble ducked background by splicing
        result = AudioSegment.empty()
        prev_end = 0

        for r_start, r_end in regions:
            # Unchanged gap before this region
            if r_start > prev_end:
                result += bg[prev_end:r_start]

            # Duck this region
            chunk = bg[r_start:r_end]
            chunk_len = len(chunk)

            # Fade from full → ducked at entry
            if fade_ms > 0 and chunk_len > fade_ms * 2:
                chunk = chunk.fade(
                    from_gain=0, to_gain=ducking_db,
                    start=0, duration=fade_ms,
                )
                # Fade from ducked → full at exit
                chunk = chunk.fade(
                    from_gain=ducking_db, to_gain=0,
                    start=chunk_len - fade_ms, duration=fade_ms,
                )
            else:
                chunk = chunk.apply_gain(ducking_db)

            result += chunk
            prev_end = r_end

        # Remainder after last ducked region
        if prev_end < len(bg):
            result += bg[prev_end:]

        return result


def _vol_to_db(ratio: float) -> float:
    if ratio <= 0:
        return -120.0
    return 20.0 * math.log10(ratio)
