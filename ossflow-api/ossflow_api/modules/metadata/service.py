"""Servicio de metadata: lee/escribe sidecar ``.bjj-meta.json``."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ossflow_api.shared.exceptions import ApiError, NotFoundError, ValidationError

from .schemas import DEFAULT_METADATA

SIDECAR_NAME = ".bjj-meta.json"


class _Forbidden(ApiError):
    status_code = 403


class MetadataService:
    """Encapsula la resolución del path y la persistencia del sidecar."""

    def __init__(self, library_path: str | None) -> None:
        self._library_path = library_path

    def _resolve_target(self, name: str) -> Path:
        if not self._library_path:
            raise NotFoundError("library_path not configured")
        base = Path(self._library_path).resolve()
        try:
            target = (base / name).resolve()
        except OSError as exc:
            raise _Forbidden("invalid path") from exc
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise _Forbidden("path traversal denied") from exc
        if target == base:
            raise _Forbidden("invalid target")
        if not target.exists() or not target.is_dir():
            raise NotFoundError("instructional not found")
        return target

    def get(self, name: str) -> dict[str, Any]:
        target = self._resolve_target(name)
        sidecar = target / SIDECAR_NAME
        if not sidecar.exists():
            return copy.deepcopy(DEFAULT_METADATA)
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return copy.deepcopy(DEFAULT_METADATA)

        result = copy.deepcopy(DEFAULT_METADATA)
        if isinstance(raw, dict):
            for k in DEFAULT_METADATA:
                if k in raw:
                    result[k] = raw[k]
        return result

    def put(self, name: str, body: Any) -> dict[str, Any]:
        target = self._resolve_target(name)
        payload = self._validate(body)
        sidecar = target / SIDECAR_NAME
        sidecar.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload

    @staticmethod
    def _validate(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValidationError("body must be a JSON object", status_code=422)

        instructor = data.get("instructor", "")
        topic = data.get("topic", "")
        tags = data.get("tags", [])
        synopsis = data.get("synopsis", "")
        year = data.get("year", None)
        voice_profile = data.get("voice_profile", "")

        if not isinstance(instructor, str):
            raise ValidationError("instructor must be string", status_code=422)
        if not isinstance(topic, str):
            raise ValidationError("topic must be string", status_code=422)
        if not isinstance(synopsis, str):
            raise ValidationError("synopsis must be string", status_code=422)
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValidationError("tags must be list[str]", status_code=422)
        if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
            raise ValidationError("year must be integer or null", status_code=422)
        if not isinstance(voice_profile, str):
            raise ValidationError("voice_profile must be string", status_code=422)

        return {
            "instructor": instructor,
            "topic": topic,
            "tags": tags,
            "synopsis": synopsis,
            "year": year,
            "voice_profile": voice_profile,
        }
