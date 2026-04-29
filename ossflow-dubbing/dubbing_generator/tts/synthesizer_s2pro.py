"""Fish Audio S2-Pro local TTS via in-container HTTP server.

Connection model: one ``httpx.Client`` per synthesizer instance. We POST
one multipart form per phrase; the reference WAV + transcript stay
constant across the episode.

Resilience:
- On HTTP error: log + return 200 ms silence so the pipeline survives.
- After ``_BREAKER_THRESHOLD`` consecutive errors: short-circuit for
  ``_BREAKER_COOLDOWN_S`` to avoid 150× hammering a dead server.
- Resets on first success.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx
from pydub import AudioSegment

from ..config import DubbingConfig

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_S = 60.0


class SynthesizerS2Pro:
    """Generate speech via the local s2.cpp HTTP server."""

    def __init__(self, config: DubbingConfig, server_manager=None) -> None:
        self.cfg = config
        self._client = httpx.Client(
            base_url=f"http://{config.s2_server_host}:{config.s2_server_port}",
            timeout=config.s2_request_timeout,
        )
        self._consecutive_failures = 0
        self._breaker_open_until: float = 0.0
        # Optional: lifecycle manager. When present we can resurrect the
        # server after a crash mid-job (s2.cpp occasionally dies after
        # generating long outputs — observed on phrases ≥400 frames).
        self._manager = server_manager
        # Set to True after a connection failure so the next request that
        # bypasses the breaker forces a server restart before retrying.
        self._needs_restart = False
        # Loud warning when ref_text is the dataclass default but the WAV
        # is NOT the matching default WAV. The clone collapses if the two
        # disagree (the model conditions on aligned phoneme→codec pairs)
        # so this is the most common foot-gun: user swaps voice in
        # settings UI but forgets to update the transcript. Detected here
        # rather than at request time so it shows up exactly once per job.
        default_ref_text = DubbingConfig.__dataclass_fields__["s2_ref_text"].default
        default_ref_wav = DubbingConfig.__dataclass_fields__["s2_ref_audio_path"].default
        if (
            config.s2_ref_text == default_ref_text
            and config.s2_ref_audio_path != default_ref_wav
        ):
            logger.warning(
                "S2-Pro: ref WAV is %s but ref_text is the default transcript "
                "for %s. Voice cloning collapses when audio and transcript "
                "drift — update s2_ref_text in Settings (or save a sidecar "
                "transcript next to the WAV).",
                Path(config.s2_ref_audio_path).name,
                Path(default_ref_wav).name,
            )

    @property
    def sample_rate(self) -> int:
        return 44100  # s2.cpp emits 44.1 kHz

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        # Best-effort cleanup; pipeline should call close() explicitly.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def generate(
        self,
        text: str,
        reference_wav: Path,
        speed: Optional[float] = None,
    ) -> AudioSegment:
        """Synthesize *text*. ``reference_wav`` and ``speed`` are accepted
        for protocol parity with the other synthesizers but are unused —
        s2.cpp does not expose a speed parameter (the pipeline absorbs
        cadence post-stretch instead)."""
        if speed is not None and speed != 1.0:
            logger.debug(
                "S2-Pro ignores speed=%.3f (use post-stretch instead)", speed,
            )
        _ = reference_wav
        if not text.strip():
            return AudioSegment.silent(duration=100)

        ref_path = Path(self.cfg.s2_ref_audio_path)
        if not ref_path.exists():
            logger.error("S2-Pro ref audio missing: %s", ref_path)
            return AudioSegment.silent(duration=200)

        # Server-down recovery: if we've previously seen a connection-level
        # failure, the s2.cpp subprocess almost certainly died. Try to bring
        # it back BEFORE issuing the POST — otherwise we'd waste another
        # failure increment hitting a dead socket.
        if self._needs_restart and self._manager is not None:
            if self._try_restart():
                self._needs_restart = False
                self._consecutive_failures = 0
                self._breaker_open_until = 0.0
            # If restart failed we fall through; the breaker logic below
            # handles the inevitable connection error.

        # Circuit breaker check.
        now = time.monotonic()
        if now < self._breaker_open_until:
            logger.warning(
                "S2-Pro circuit breaker open (cooldown %.0f s remaining)",
                self._breaker_open_until - now,
            )
            return AudioSegment.silent(duration=200)

        params = {
            "max_new_tokens": self.cfg.s2_max_tokens,
            "temperature": self.cfg.s2_temperature,
            "top_p": self.cfg.s2_top_p,
            "top_k": self.cfg.s2_top_k,
        }

        try:
            with ref_path.open("rb") as fh:
                resp = self._client.post(
                    "/generate",
                    files={"reference": (ref_path.name, fh, "audio/wav")},
                    data={
                        "text": text,
                        "reference_text": self.cfg.s2_ref_text,
                        "params": json.dumps(params),
                    },
                )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # Connection-level errors (refused, disconnected) almost always
            # mean the server died — flag for resurrection on next call.
            msg = str(exc)
            if "Connection refused" in msg or "disconnected" in msg.lower():
                self._needs_restart = True
            self._consecutive_failures += 1
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._breaker_open_until = now + _BREAKER_COOLDOWN_S
                logger.error(
                    "S2-Pro circuit breaker tripped (%d failures); "
                    "cooling down %.0f s", self._consecutive_failures,
                    _BREAKER_COOLDOWN_S,
                )
                # Reset counter on trip so post-cooldown probe gets a clean
                # baseline; a single failure after cooldown re-trips immediately.
                self._consecutive_failures = 0
            logger.error("S2-Pro synthesize failed (text=%r): %s",
                         text[:80], exc)
            return AudioSegment.silent(duration=200)

        self._consecutive_failures = 0
        self._needs_restart = False
        try:
            return AudioSegment.from_file(io.BytesIO(resp.content), format="wav")
        except Exception as exc:  # noqa: BLE001
            logger.warning("S2-Pro returned undecodable audio (%r): %s",
                           text[:80], exc)
            return AudioSegment.silent(duration=200)

    def _try_restart(self) -> bool:
        """Restart the s2.cpp server after a crash. Returns True on success.

        Called from inside ``generate`` when a connection error suggests the
        subprocess died. Best-effort: if the manager fails to bring it back
        we keep the breaker closed so the rest of the job degrades to
        silence rather than spending its time re-crashing the GPU.
        """
        if self._manager is None:
            return False
        try:
            logger.error("S2-Pro server appears dead — attempting restart")
            self._manager.stop()
            self._manager.start()
            ok = self._manager.wait_until_ready(
                timeout=self.cfg.s2_health_timeout_s,
            )
            if ok:
                logger.info("S2-Pro server restarted successfully")
                return True
            logger.error(
                "S2-Pro server failed to come back up within %.0f s",
                self.cfg.s2_health_timeout_s,
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("S2-Pro restart raised: %s", exc)
            return False
