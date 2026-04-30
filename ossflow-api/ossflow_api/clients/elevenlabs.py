"""ElevenLabs Dubbing Studio client wrapper.

Thin adapter around the ElevenLabs SDK's ``dubbing`` API so the rest of
the app sees a tiny, well-typed surface: ``start`` / ``poll`` / ``download``.
The SDK call shapes change between versions and also between the sync and
streaming endpoints — keeping that messiness isolated here means routes
never import ``elevenlabs`` directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

log = logging.getLogger(__name__)


class ElevenLabsDubbingError(RuntimeError):
    """Raised when the ElevenLabs Dubbing API rejects or fails a job."""


@dataclass(frozen=True)
class DubbingJob:
    dubbing_id: str
    status: str


class ElevenLabsDubbingClient:
    """Thin wrapper around ``client.dubbing``.

    The SDK surface is intentionally narrow here: ``start``, ``poll``,
    ``download``. Anything more elaborate (Studio edits, resource render,
    speaker segmentation tweaks) stays out of this class so a future
    provider swap only has to reimplement those three calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        key = (api_key or os.environ.get("ELEVENLABS_API_KEY", "")).strip()
        if not key:
            raise ElevenLabsDubbingError(
                "ELEVENLABS_API_KEY is empty. Set it in the processor-api "
                "container environment to use the Dubbing Studio flow."
            )
        from elevenlabs.client import ElevenLabs
        self._client = ElevenLabs(api_key=key, timeout=timeout)

    def start(
        self,
        *,
        file: BinaryIO,
        filename: str,
        target_lang: str,
        source_lang: Optional[str] = "en",
        num_speakers: int = 1,
        watermark: bool = True,
        name: Optional[str] = None,
    ) -> DubbingJob:
        """Kick off a dubbing job for the given video file.

        ``watermark=True`` is the 33%-cheaper tier the user picked for
        their 120k-char Creator plan — we default to it. Pass explicitly
        ``watermark=False`` to burn extra credits for the watermark-free
        output.
        """
        # SDK v2 renamed ``dub`` → ``create``. Use whichever the installed
        # SDK exposes so the code works with both. The SDK >= 2.x only has
        # ``create``; earlier 1.x only had ``dub``. Trying ``create`` first.
        call = getattr(self._client.dubbing, "create", None) or getattr(
            self._client.dubbing, "dub", None
        )
        if call is None:
            raise ElevenLabsDubbingError(
                "ElevenLabs SDK has neither dubbing.create nor dubbing.dub; "
                "check the installed elevenlabs package version."
            )
        try:
            resp = call(
                file=(filename, file, "video/mp4"),
                source_lang=source_lang,
                target_lang=target_lang,
                num_speakers=num_speakers,
                watermark=watermark,
                name=name or filename,
            )
        except Exception as exc:
            raise ElevenLabsDubbingError(f"dubbing.create failed: {exc}") from exc

        dubbing_id = getattr(resp, "dubbing_id", None) or getattr(resp, "id", None)
        if not dubbing_id:
            raise ElevenLabsDubbingError(
                f"dubbing.dub returned no dubbing_id (resp={resp!r})"
            )
        status = getattr(resp, "status", "created")
        log.info("ElevenLabs dubbing started: id=%s status=%s", dubbing_id, status)
        return DubbingJob(dubbing_id=dubbing_id, status=status)

    def poll(self, dubbing_id: str) -> DubbingJob:
        """Fetch current status of a dubbing job.

        ElevenLabs exposes these states: ``dubbing`` (in progress),
        ``dubbed`` (ready to download), ``failed``. The SDK may also
        return ``created``/``processing`` transiently — we pass them
        through as-is so callers can decide what to treat as terminal.
        """
        try:
            resp = self._client.dubbing.get(dubbing_id=dubbing_id)
        except Exception as exc:
            raise ElevenLabsDubbingError(f"dubbing.get failed: {exc}") from exc
        status = getattr(resp, "status", "unknown")
        return DubbingJob(dubbing_id=dubbing_id, status=status)

    def download(self, dubbing_id: str, target_lang: str) -> bytes:
        """Download the rendered dubbed file as raw bytes (MP4 stream)."""
        try:
            stream = self._client.dubbing.audio.get(
                dubbing_id=dubbing_id,
                language_code=target_lang,
            )
        except Exception as exc:
            raise ElevenLabsDubbingError(f"dubbing.audio.get failed: {exc}") from exc

        if isinstance(stream, (bytes, bytearray)):
            return bytes(stream)
        buf = bytearray()
        for chunk in stream:
            if chunk:
                buf.extend(chunk)
        if not buf:
            raise ElevenLabsDubbingError(
                f"dubbing.audio.get returned no bytes for {dubbing_id}"
            )
        return bytes(buf)


def resolve_output_path(source_video: Path) -> Path:
    """Compute the destination inside ``<Season>/elevenlabs/<filename>``.

    The source layout we need to support looks like::

        /media/.../Instructional/Season 01/S01E02 - Foo.mp4
        /media/.../Instructional/S01E02 - Foo.mp4   # flat (no Season)

    In both cases the dubbed output lands in a sibling ``elevenlabs/``
    folder next to the source, keeping ElevenLabs outputs physically
    separated from the XTTS pipeline outputs (which write next to the
    source file as ``*_DOBLADO.mkv``).
    """
    season_dir = source_video.parent
    out_dir = season_dir / "elevenlabs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / source_video.name
