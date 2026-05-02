"""Boot the s2.cpp HTTP server alongside the FastAPI process.

Lifetime model:
- ``start()`` is called from FastAPI's startup hook. It must NOT block:
  it spawns the subprocess and a daemon thread that polls /health.
  FastAPI must reach the ready state immediately so /health on port 8003
  responds even while the GGUF is still mmap-ing.
- A second daemon thread drains the subprocess's merged stdout/stderr
  into ``logger.info`` — without this the OS pipe buffer (~64 KB on
  Linux) fills and s2 deadlocks on its next print, which would manifest
  as every synthesis request hanging forever.
- ``stop()`` sends SIGTERM, then SIGKILL on a 5 s timeout.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from ..config import DubbingConfig

logger = logging.getLogger(__name__)

# Path inside the runtime image (set by Dockerfile COPY --from=s2cpp-builder).
_S2_BINARY = Path("/usr/local/bin/s2")


def _wait_for_health(host: str, port: int, timeout_s: float) -> bool:
    """Poll until the cpp-httplib server answers any GET, or timeout."""
    deadline = time.monotonic() + timeout_s
    url = f"http://{host}:{port}/"
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=1.0)
            return True  # cpp-httplib answers 404; presence is enough
        except httpx.HTTPError:
            time.sleep(0.5)
    return False


class S2ProServerManager:
    """Lifecycle wrapper for the s2.cpp HTTP server subprocess."""

    def __init__(self, config: DubbingConfig) -> None:
        self.cfg = config
        self._process: Optional[subprocess.Popen] = None
        self._ready = threading.Event()
        self._readiness_thread: Optional[threading.Thread] = None
        self._drain_thread: Optional[threading.Thread] = None

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def wait_until_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout=timeout)

    def start(self) -> None:
        # Idempotent: safe to call when the process is already alive (returns
        # immediately). The pipeline calls start() at fase 3 of each file and
        # stop() right after fase 6, so on a 6 GB card Demucs of file N+1 has
        # the GPU to itself. The OS keeps the GGUF in page cache so re-mmap
        # between files is ~2-3 s, not the cold-start ~10 s.
        if self._process is not None and self._process.poll() is None:
            logger.debug("S2-Pro server already running (pid=%s)",
                         self._process.pid)
            return
        if self.cfg.tts_engine != "s2pro":
            logger.debug("S2-Pro server skipped (engine=%s)",
                         self.cfg.tts_engine)
            return
        if not _S2_BINARY.exists():
            logger.warning("S2-Pro binary not found at %s — server NOT started",
                           _S2_BINARY)
            return
        gguf = Path(self.cfg.s2_gguf_path)
        if not gguf.exists():
            logger.warning("S2-Pro GGUF model missing at %s — server NOT started",
                           gguf)
            return

        cmd = [
            str(_S2_BINARY),
            "--server",
            "-H", self.cfg.s2_server_host,
            "-P", str(self.cfg.s2_server_port),
            "-c", str(self.cfg.s2_cuda_device),
            "-m", str(gguf),
            "-t", str(self.cfg.s2_tokenizer_path),
        ]
        logger.info("Starting s2.cpp server: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ},
        )
        # Drain stdout to logger so the pipe buffer never fills.
        self._drain_thread = threading.Thread(
            target=self._drain_stdout, daemon=True, name="s2pro-drain",
        )
        self._drain_thread.start()
        # Probe health asynchronously so start() returns immediately.
        self._readiness_thread = threading.Thread(
            target=self._probe_until_ready, daemon=True, name="s2pro-ready",
        )
        self._readiness_thread.start()

    def _drain_stdout(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        try:
            for line in iter(self._process.stdout.readline, b""):
                if not line:
                    break
                logger.info("s2: %s", line.decode(errors="replace").rstrip())
        except Exception as exc:  # noqa: BLE001
            logger.debug("s2pro stdout drain ended: %s", exc)

    def _probe_until_ready(self) -> None:
        ok = _wait_for_health(
            self.cfg.s2_server_host,
            self.cfg.s2_server_port,
            timeout_s=self.cfg.s2_health_timeout_s,
        )
        if ok:
            self._ready.set()
            logger.info("s2.cpp server is up on %s:%d",
                        self.cfg.s2_server_host, self.cfg.s2_server_port)
        else:
            logger.error(
                "s2.cpp server failed health probe in %.0fs — "
                "synthesis requests will fail-fast until it recovers",
                self.cfg.s2_health_timeout_s,
            )

    def stop(self) -> None:
        proc = self._process
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._process = None
        self._ready.clear()
