"""Path translation between host (Windows/UNC) and container (POSIX).

Single responsibility: translate paths so microservices (which mount
``library_path`` as ``/library``) receive POSIX container paths, regardless
of how the host stored them (UNC ``\\\\server\\share\\...`` or drive-mapped
``Z:\\...``).

Idempotent: a path already rooted at ``container_root`` is returned unchanged.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath


def _normalize_sep(p: str) -> str:
    return p.replace("\\", "/")


def _strip_trailing_slash(p: str) -> str:
    return p.rstrip("/")


def to_container_path(
    host_path: str,
    library_path: str,
    container_root: str = "/media",
) -> str:
    """Translate a host path into the container-visible path.

    - If ``host_path`` already lives under ``container_root`` → returned as-is.
    - If ``host_path`` starts with ``/media`` or ``/library`` → returned as-is
      (pass-through for the current NAS mount and legacy compat).
    - Else it must live under ``library_path``; the common prefix is replaced.
    - Case-insensitive prefix compare (Windows paths are case-insensitive).

    Raises ``ValueError`` if ``host_path`` is outside both roots.
    """
    if not host_path:
        raise ValueError("host_path is empty")
    if not library_path:
        raise ValueError("library_path is not configured")

    host_norm = _strip_trailing_slash(_normalize_sep(host_path))
    lib_norm = _strip_trailing_slash(_normalize_sep(library_path))
    root_norm = _strip_trailing_slash(_normalize_sep(container_root)) or "/"

    # Idempotent pass-through for configured root + known container roots.
    passthrough_roots = {root_norm.lower(), "/media", "/library"}
    for pr in passthrough_roots:
        if host_norm.lower() == pr or host_norm.lower().startswith(pr + "/"):
            return host_norm

    # Within library_path?
    if host_norm.lower() == lib_norm.lower():
        return root_norm
    if host_norm.lower().startswith(lib_norm.lower() + "/"):
        rest = host_norm[len(lib_norm) + 1 :]
        return f"{root_norm}/{rest}"

    raise ValueError(
        f"Path {host_path!r} is outside library_path {library_path!r} "
        f"and container root {container_root!r}"
    )


def from_container_path(
    container_path: str,
    library_path: str,
    container_root: str = "/media",
) -> str:
    """Inverse of :func:`to_container_path`.

    Useful when the API needs to hand a host path back (e.g. to ffmpeg on
    the host, or `/api/browse`).
    """
    if not container_path:
        raise ValueError("container_path is empty")

    cp_norm = _strip_trailing_slash(_normalize_sep(container_path))
    root_norm = _strip_trailing_slash(_normalize_sep(container_root)) or "/"
    lib_norm = _strip_trailing_slash(_normalize_sep(library_path))

    if cp_norm.lower() == root_norm.lower():
        return lib_norm
    if cp_norm.lower().startswith(root_norm.lower() + "/"):
        rest = cp_norm[len(root_norm) + 1 :]
        # Preserve UNC backslashes if library_path is UNC/Windows-like
        if library_path.startswith("\\\\") or (
            len(library_path) >= 2 and library_path[1] == ":"
        ):
            return str(PureWindowsPath(library_path) / rest.replace("/", "\\"))
        return str(PurePosixPath(lib_norm) / rest)

    # Not a container path → return as-is
    return container_path
