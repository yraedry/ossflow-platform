"""Servicio de NAS mount (CIFS/SMB) para el library picker.

Encapsula la lógica de los endpoints ``/api/mount`` POST y GET que
montan/inspeccionan ``MEDIA_ROOT`` con un share SMB.

El comando ``mount`` se ejecuta vía ``subprocess`` con un timeout
defensivo: ``mountpoint -q`` puede colgarse si el mount está zombie
(CIFS perdió conexión), y eso bloqueaba al worker entero antes.

El config se persiste en ``CONFIG_DIR/mount.json`` para auto-mount
en restart (lo dispara ``api.app:_auto_mount_on_startup``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = "/data/config"
_MOUNT_TIMEOUT = 15
_MOUNTPOINT_TIMEOUT = 5


class _BadRequest(Exception):
    """Body de mount inválido (e.g. falta el campo ``share``)."""


class _MountFailed(Exception):
    """``mount`` retornó un código distinto de 0 o timeout."""


def _media_root() -> Path:
    return Path(os.environ.get("MEDIA_ROOT", "/media"))


def _config_dir() -> Path:
    return Path(os.environ.get("CONFIG_DIR", _DEFAULT_CONFIG_DIR))


def mount_share(body: dict[str, Any]) -> dict[str, Any]:
    """Monta un share CIFS en ``MEDIA_ROOT``. Persiste config en mount.json.

    Lanza ``_BadRequest`` si falta el campo share, ``_MountFailed`` si
    el comando ``mount`` falla.
    """
    share = (body.get("share", "") or "").strip()
    if not share:
        raise _BadRequest(
            "Campo 'share' requerido (ej: //10.10.100.6/multimedia)",
        )
    share = share.replace("\\", "/")
    if not share.startswith("//"):
        share = "//" + share.lstrip("/")

    media = _media_root()
    media.mkdir(parents=True, exist_ok=True)

    username = body.get("username", "guest")
    password = body.get("password", "")

    opts = f"username={username},password={password},vers=3.0,iocharset=utf8,noperm"
    cmd = ["mount", "-t", "cifs", share, str(media), "-o", opts]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_MOUNT_TIMEOUT)
        if result.returncode != 0:
            opts_fallback = f"username={username},password={password},iocharset=utf8,noperm"
            cmd_fb = ["mount", "-t", "cifs", share, str(media), "-o", opts_fallback]
            result = subprocess.run(cmd_fb, capture_output=True, text=True, timeout=_MOUNT_TIMEOUT)
            if result.returncode != 0:
                raise _MountFailed(
                    result.stderr.strip() or "mount returned non-zero status",
                )
    except subprocess.TimeoutExpired as exc:
        raise _MountFailed("Timeout al montar. Verifica que el NAS es accesible.") from exc

    cfg_dir = _config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "mount.json").write_text(json.dumps({
        "share": share,
        "username": username,
        "password": password,
    }))

    dirs = [d for d in media.iterdir() if d.is_dir()]
    return {"mounted": True, "share": share, "directories": len(dirs)}


def mount_status() -> dict[str, Any]:
    """Devuelve si ``MEDIA_ROOT`` está montado y un sample de directorios."""
    media = _media_root()
    if not media.exists():
        return {"mounted": False, "share": None}

    try:
        result = subprocess.run(
            ["mountpoint", "-q", str(media)],
            capture_output=True,
            timeout=_MOUNTPOINT_TIMEOUT,
        )
        is_mount = result.returncode == 0
    except subprocess.TimeoutExpired:
        is_mount = False

    dirs = [d.name for d in media.iterdir() if d.is_dir()] if media.exists() else []
    return {
        "mounted": is_mount,
        "path": str(media),
        "directories": len(dirs),
        "items": dirs[:20],
    }
