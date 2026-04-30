"""Módulo elevenlabs: doblaje vía ElevenLabs Dubbing Studio (cloud, paid).

Camino completamente separado del doblaje local (XTTS/Coqui). Sube el vídeo
a ElevenLabs, hace polling, descarga el MP4 doblado y lo deja en
``<Season>/elevenlabs/<filename>``. No usa SRT ni traductor local.
"""

from .router import router as elevenlabs_router
from .service import resume_orphan_jobs

__all__ = ["elevenlabs_router", "resume_orphan_jobs"]
