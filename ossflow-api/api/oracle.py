"""Oracle proxy/orchestrator router.

Bridges the processor-api with the chapter-splitter ``oracle`` subsystem:

- ``GET    /api/oracle/providers``                    list registered providers
- ``GET    /api/oracle/{instructional_path:path}``    read cached oracle
- ``POST   /api/oracle/{path}/resolve``               search candidates
- ``POST   /api/oracle/{path}/scrape``                scrape and persist
- ``PUT    /api/oracle/{path}``                       manual edit
- ``DELETE /api/oracle/{path}``                       invalidate cache

This module never imports ``chapter_splitter``; communication is HTTP only.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from api.scan_cache import ScanCache, patch_poster_in_cache
from api.settings import CONFIG_DIR, get_library_path

log = logging.getLogger(__name__)

_scan_cache = ScanCache(CONFIG_DIR / "library.json")

router = APIRouter(prefix="/api/oracle", tags=["oracle"])

SIDECAR_NAME = ".bjj-meta.json"
DEFAULT_TIMEOUT = 30.0


def _splitter_base() -> str:
    return os.environ.get("SPLITTER_URL", "http://chapter-splitter:8001").rstrip("/")


# ---------------------------------------------------------------------------
# Path resolution + atomic sidecar IO
# ---------------------------------------------------------------------------

def _resolve_instructional(raw_path: str) -> Path:
    """Decode + validate ``raw_path`` (must be inside library_path)."""
    decoded = unquote(raw_path).strip()
    if not decoded:
        raise HTTPException(status_code=422, detail="empty instructional path")

    lib = get_library_path()
    if not lib:
        raise HTTPException(status_code=400, detail="library_path no configurado")

    # Try as absolute first (common: frontend sends the full host path).
    candidate = Path(decoded)
    if not candidate.is_absolute():
        candidate = Path(lib) / decoded

    try:
        resolved = candidate.resolve()
        lib_resolved = Path(lib).resolve()
        resolved.relative_to(lib_resolved)
    except (OSError, ValueError):
        raise HTTPException(status_code=403, detail="path outside library")

    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail="instructional not found")
    return resolved


def _read_meta(folder: Path) -> dict[str, Any]:
    sidecar = folder / SIDECAR_NAME
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        log.warning("failed to read meta %s: %s", sidecar, exc)
        return {}


def _write_meta_atomic(folder: Path, meta: dict[str, Any]) -> None:
    sidecar = folder / SIDECAR_NAME
    tmp = folder / (SIDECAR_NAME + ".tmp")
    payload = json.dumps(meta, indent=2, ensure_ascii=False)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, sidecar)


# ---------------------------------------------------------------------------
# OracleResult validation (minimal, schema-compatible with chapter-splitter)
# ---------------------------------------------------------------------------

def _validate_oracle_result(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="OracleResult must be an object")

    product_url = data.get("product_url", "")
    scraped_at = data.get("scraped_at", "")
    volumes = data.get("volumes", [])

    if not isinstance(product_url, str):
        raise HTTPException(status_code=422, detail="product_url must be string")
    if not isinstance(scraped_at, str):
        raise HTTPException(status_code=422, detail="scraped_at must be string")
    if not isinstance(volumes, list):
        raise HTTPException(status_code=422, detail="volumes must be list")

    clean_volumes: list[dict[str, Any]] = []
    for vi, vol in enumerate(volumes):
        if not isinstance(vol, dict):
            raise HTTPException(status_code=422, detail=f"volumes[{vi}] must be object")
        number = vol.get("number")
        chapters = vol.get("chapters", [])
        total_duration_s = vol.get("total_duration_s", 0)
        if not isinstance(number, int):
            raise HTTPException(status_code=422, detail=f"volumes[{vi}].number must be int")
        if not isinstance(chapters, list):
            raise HTTPException(status_code=422, detail=f"volumes[{vi}].chapters must be list")
        if not isinstance(total_duration_s, (int, float)):
            raise HTTPException(status_code=422, detail=f"volumes[{vi}].total_duration_s must be number")

        clean_chapters: list[dict[str, Any]] = []
        for ci, ch in enumerate(chapters):
            if not isinstance(ch, dict):
                raise HTTPException(status_code=422, detail=f"volumes[{vi}].chapters[{ci}] must be object")
            title = ch.get("title", "")
            start_s = ch.get("start_s")
            end_s = ch.get("end_s")
            if not isinstance(title, str):
                raise HTTPException(status_code=422, detail=f"chapters[{ci}].title must be string")
            if not isinstance(start_s, (int, float)) or not isinstance(end_s, (int, float)):
                raise HTTPException(status_code=422, detail=f"chapters[{ci}] start_s/end_s must be number")
            clean_chapters.append({"title": title, "start_s": float(start_s), "end_s": float(end_s)})

        clean_volumes.append({
            "number": number,
            "chapters": clean_chapters,
            "total_duration_s": float(total_duration_s),
        })

    out: dict[str, Any] = {
        "product_url": product_url,
        "scraped_at": scraped_at,
        "volumes": clean_volumes,
    }
    if "provider_id" in data and isinstance(data["provider_id"], str):
        out["provider_id"] = data["provider_id"]
    if "poster_url" in data and isinstance(data["poster_url"], str) and data["poster_url"]:
        out["poster_url"] = data["poster_url"]
    return out


# ---------------------------------------------------------------------------
# Poster auto-download
# ---------------------------------------------------------------------------

_POSTER_STEMS = ("poster", "cover", "folder")
_POSTER_EXTS = ("jpg", "jpeg", "png", "webp")


def _has_local_poster(folder: Path) -> bool:
    for stem in _POSTER_STEMS:
        for ext in _POSTER_EXTS:
            if (folder / f"{stem}.{ext}").exists():
                return True
            if (folder / f"{stem}.{ext.upper()}").exists():
                return True
    return False


def _trim_black_borders(path: Path, *, threshold: int = 18, min_keep_ratio: float = 0.4) -> bool:
    """Crop solid black (or near-black) borders from a poster file in-place.

    BJJFanatics serves many posters as portrait-canvas JPEGs with the real
    cover art centred between black bands — e.g. "Engaging without regrets"
    and "Tripod Passing" arrive as 1631×2194 images whose top ~250 px and
    bottom ~350 px are plain black. Older, square-capable products like
    Half Guard Anthology ship flush art and need no trim. We detect this
    per-file by thresholding the image to a binary mask and picking up the
    tight bounding box of non-black pixels.

    Guards:
      * ``threshold`` — pixels darker than this in all channels are treated
        as border. 18 catches the JPEG compression noise around true black
        without eating dark clothing in the art.
      * ``min_keep_ratio`` — never trim away more than 60% in either
        dimension. Stops a false positive on genuinely dark art (e.g. a
        black-gi instructional) from destroying the poster.

    Returns True if the file was modified.
    """
    try:
        from PIL import Image, ImageChops  # noqa: WPS433 — optional dep path
    except ImportError:
        log.debug("Pillow not available, skipping poster trim")
        return False
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            # ImageChops.difference with a solid black reference gives us
            # per-pixel "distance from black"; getbbox() on a thresholded
            # version is the portable way to find the tight crop.
            bg = Image.new("RGB", rgb.size, (0, 0, 0))
            diff = ImageChops.difference(rgb, bg)
            mask = diff.convert("L").point(lambda v: 255 if v > threshold else 0)
            bbox = mask.getbbox()
            if not bbox:
                return False  # image is entirely black — leave alone
            left, top, right, bottom = bbox
            width, height = rgb.size
            new_w = right - left
            new_h = bottom - top
            if (
                new_w >= width
                and new_h >= height
            ):
                return False  # nothing to trim
            if new_w < width * min_keep_ratio or new_h < height * min_keep_ratio:
                # Suspicious — we'd be cropping >60% away. Likely a
                # genuinely dark poster; leave the original alone.
                log.info(
                    "skipping poster trim for %s (would keep %dx%d of %dx%d)",
                    path.name, new_w, new_h, width, height,
                )
                return False
            cropped = rgb.crop(bbox)
            cropped.save(path, quality=92, optimize=True)
            log.info(
                "trimmed poster %s: %dx%d → %dx%d",
                path.name, width, height, new_w, new_h,
            )
            return True
    except Exception as exc:
        log.warning("poster trim failed for %s: %s", path, exc)
        return False


def _strip_shopify_thumb(url: str) -> str:
    """Remove Shopify CDN size mutations (query + in-path) on a poster URL.

    Older cached sidecars stored URLs with ``?crop=center&height=300&width=300``
    or an ``_NNNxMMM`` suffix in the path. Those serve a 300×300 centre-cropped
    thumbnail that loses the top of the cover art (e.g. the word "ENGAGING"
    on Jozef Chen's product). Strip both forms so the redownload grabs the
    full-res original. Mirrors BjjFanaticsProvider._strip_shopify_size.
    """
    import re as _re
    if "cdn.shop" not in url and "cdn.shopify" not in url and "bjjfanatics.com/cdn" not in url:
        return url
    try:
        base, _sep, query = url.partition("?")
        base = _re.sub(r"_\d+x\d+(?=\.[A-Za-z0-9]+$)", "", base)
        if not query:
            return base
        keep = []
        for part in query.split("&"):
            key = part.split("=", 1)[0].lower()
            if key in {"width", "height", "crop", "pad_color"}:
                continue
            keep.append(part)
        return base + ("?" + "&".join(keep) if keep else "")
    except Exception:
        return url


async def _download_poster_if_missing(
    folder: Path, poster_url: str | None, *, force: bool = False
) -> str | None:
    """Download poster_url to folder/poster.<ext>.

    If ``force`` is True, removes any existing local poster first.
    Returns the saved filename, or None if skipped/failed.
    """
    if not poster_url:
        return None
    # Defensive: old sidecars may carry crop-center size mutations. Strip
    # them so we always fetch the highest-res original available.
    poster_url = _strip_shopify_thumb(poster_url)
    if force:
        for stem in _POSTER_STEMS:
            for ext in _POSTER_EXTS:
                for candidate in (folder / f"{stem}.{ext}", folder / f"{stem}.{ext.upper()}"):
                    if candidate.exists():
                        try:
                            candidate.unlink()
                        except OSError as exc:
                            log.warning("could not remove existing poster %s: %s", candidate, exc)
    elif _has_local_poster(folder):
        return None
    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT, follow_redirects=True
        ) as client:
            r = await client.get(poster_url)
            if r.status_code >= 400:
                log.warning("poster download HTTP %d for %s", r.status_code, poster_url)
                return None
            content_type = (r.headers.get("content-type") or "").lower()
            ext = "jpg"
            for known_ext, ct in (
                ("png", "image/png"),
                ("webp", "image/webp"),
                ("jpg", "image/jpeg"),
            ):
                if ct in content_type:
                    ext = known_ext
                    break
            else:
                # Fallback to URL extension if content-type is unhelpful.
                lower = poster_url.lower().split("?", 1)[0]
                for known in _POSTER_EXTS:
                    if lower.endswith("." + known):
                        ext = known if known != "jpeg" else "jpg"
                        break
            dest = folder / f"poster.{ext}"
            tmp = folder / f"poster.{ext}.tmp"
            raw = r.content
            tmp.write_bytes(raw)
            os.replace(tmp, dest)
            # Remove the black canvas padding BJJFanatics bakes into many
            # portrait posters. Safe to no-op if the file is already tight.
            _trim_black_borders(dest)
            log.info("downloaded poster to %s", dest)
            return dest.name
    except (httpx.HTTPError, OSError) as exc:
        log.warning("poster download failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Heuristics: derive title/author from folder name when meta missing
# ---------------------------------------------------------------------------

def _derive_title_author(folder: Path, meta: dict[str, Any]) -> tuple[str, str]:
    title = meta.get("topic") or ""
    author = meta.get("instructor") or ""
    if title and author:
        return title, author

    name = folder.name
    # Patterns seen in the wild:
    #   "Title - Author"  (library convention — right side is usually a person)
    #   "Author - Title"  (legacy)
    #   "Title by Author"
    # Heuristic: on " - " split, the side with ≤3 words is the author.
    # If both sides have >3 words, prefer the RIGHT as author (library convention).
    if " - " in name and not author:
        left, right = [s.strip() for s in name.split(" - ", 1)]
        left_words = len(left.split())
        right_words = len(right.split())
        if right_words <= 3 and left_words > right_words:
            title = title or left
            author = author or right
        elif left_words <= 3 and right_words > left_words:
            author = author or left
            title = title or right
        else:
            # Ambiguous — default to "Title - Author" (library convention)
            title = title or left
            author = author or right
    elif " by " in name.lower() and not author:
        idx = name.lower().rindex(" by ")
        title = title or name[:idx].strip()
        author = author or name[idx + 4:].strip()
    else:
        title = title or name
    return title, author


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/providers")
async def list_providers() -> Any:
    """Proxy GET to chapter-splitter /oracle/providers."""
    url = f"{_splitter_base()}/oracle/providers"
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.get(url)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"backend unreachable: {exc}")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"backend error {r.status_code}: {r.text}")
    try:
        return r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="backend returned invalid JSON")


@router.get("/{instructional_path:path}")
async def get_oracle(instructional_path: str):
    folder = _resolve_instructional(instructional_path)
    meta = _read_meta(folder)
    oracle = meta.get("oracle")
    if not oracle:
        raise HTTPException(status_code=404, detail="no oracle cached")
    return JSONResponse(oracle)


@router.post("/{instructional_path:path}/resolve")
async def resolve_oracle(instructional_path: str, request: Request):
    folder = _resolve_instructional(instructional_path)
    try:
        body = await request.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    provider_id = body.get("provider_id")  # may be None for autodetect
    override_title = body.get("title")
    override_author = body.get("author")

    meta = _read_meta(folder)
    title, author = _derive_title_author(folder, meta)
    if isinstance(override_title, str) and override_title.strip():
        title = override_title.strip()
    if isinstance(override_author, str) and override_author.strip():
        author = override_author.strip()

    payload = {"title": title, "author": author}
    if provider_id is not None:
        payload["provider_id"] = provider_id

    url = f"{_splitter_base()}/oracle/search"
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"backend unreachable: {exc}")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"backend error {r.status_code}: {r.text}")
    try:
        return r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="backend returned invalid JSON")


@router.post("/{instructional_path:path}/scrape")
async def scrape_oracle(instructional_path: str, request: Request):
    folder = _resolve_instructional(instructional_path)
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid JSON")
    if not isinstance(body, dict) or not isinstance(body.get("url"), str) or not body["url"]:
        raise HTTPException(status_code=422, detail="body must include 'url' string")

    target_url = body["url"]
    backend_url = f"{_splitter_base()}/oracle/scrape"
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.post(backend_url, json={"url": target_url})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"backend unreachable: {exc}")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"backend error {r.status_code}: {r.text}")
    try:
        oracle_result = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="backend returned invalid JSON")

    validated = _validate_oracle_result(oracle_result)

    meta = _read_meta(folder)
    meta["oracle"] = validated
    meta["url_bjjfanatics"] = target_url
    _write_meta_atomic(folder, meta)

    saved = await _download_poster_if_missing(folder, validated.get("poster_url"))
    response = dict(validated)
    if saved:
        response["poster_downloaded"] = saved
        patch_poster_in_cache(_scan_cache, folder.name, saved)
    return JSONResponse(response)


@router.put("/{instructional_path:path}")
async def put_oracle(instructional_path: str, request: Request):
    folder = _resolve_instructional(instructional_path)
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid JSON")
    validated = _validate_oracle_result(body)

    meta = _read_meta(folder)
    meta["oracle"] = validated
    if validated.get("product_url"):
        meta["url_bjjfanatics"] = validated["product_url"]
    _write_meta_atomic(folder, meta)

    saved = await _download_poster_if_missing(folder, validated.get("poster_url"))
    response = dict(validated)
    if saved:
        response["poster_downloaded"] = saved
        patch_poster_in_cache(_scan_cache, folder.name, saved)
    return JSONResponse(response)


@router.delete("/{instructional_path:path}")
async def delete_oracle(instructional_path: str):
    folder = _resolve_instructional(instructional_path)
    meta = _read_meta(folder)
    changed = False
    if "oracle" in meta:
        meta.pop("oracle", None)
        changed = True
    if changed:
        _write_meta_atomic(folder, meta)
    return {"ok": True}
