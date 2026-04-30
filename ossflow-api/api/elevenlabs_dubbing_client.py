"""Compat shim. Lógica movida a ``ossflow_api.clients.elevenlabs``."""

from ossflow_api.clients.elevenlabs import (  # noqa: F401
    DubbingJob,
    ElevenLabsDubbingClient,
    ElevenLabsDubbingError,
    resolve_output_path,
)
