"""Módulo promote: remux ffmpeg para promover un vídeo doblado a forma multi-track.

Tras el pipeline de doblaje, ``<Season>/doblajes/<name>.mkv`` contiene el
vídeo doblado. Cuando el usuario lo aprueba, "promueve" el capítulo: se
construye un único ``<Season>/<name>.mkv`` con audio español doblado
(track default) + audio inglés original + subtítulos ES/EN, y se borran
los artefactos intermedios.

La lógica vive en ``PromoteService``; el router sólo valida payloads y
delega.
"""

from .router import router as promote_router

__all__ = ["promote_router"]
