"""Servicio de promote: remux ffmpeg + cleanup de un capítulo doblado.

Responsabilidades:

* Resolver paths de entrada/salida a partir del original ``.mp4``.
* Construir la línea de comandos de ``ffmpeg`` para el mux multi-track.
* Ejecutar ``ffmpeg`` y hacer un swap atómico (``.mkv.tmp`` → ``.mkv``).
* Borrar los artefactos intermedios (originals, dubbed, sidecars) sólo
  tras un mux exitoso.
* Refrescar la cache de escaneo para que la UI vea el nuevo estado sin
  esperar a un rescan manual.
* Mantener un lock por season para serializar promociones sobre la misma
  carpeta sin bloquear seasons distintas. **El estado vive en la instancia**
  del servicio (``self._season_locks``); en producción el factory
  ``get_promote_service`` mantiene un singleton scope-app.

Mantiene el comportamiento exacto de ``api/promote.py``; los cambios son
sólo de empaquetado para encajar en el patrón vertical slice.

``shutil``/``subprocess`` quedan expuestos a nivel de módulo para que los
tests puedan ``monkeypatch.setattr(service, "subprocess", ...)`` igual que
hacían contra el módulo legacy.
"""

from __future__ import annotations

import logging
import shutil  # noqa: F401  (re-exportado para tests legacy que monkeypatcheen)
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from fastapi import HTTPException

from .schemas import Inputs

log = logging.getLogger(__name__)


_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


# ---------------------------------------------------------------------------
# Resolución de inputs (puro respecto a sus argumentos)
# ---------------------------------------------------------------------------


def resolve_inputs(original_path: str) -> Inputs:
    """Resuelve los paths necesarios para promocionar un único capítulo.

    Lanza ``HTTPException(409)`` con códigos estables (``original_missing``,
    ``dubbed_missing``, ``already_promoted``) cuando los pre-requisitos no
    se cumplen.
    """
    original = Path(original_path)
    if not original.is_file():
        raise HTTPException(status_code=409, detail={
            "code": "original_missing",
            "message": f"Original video not found: {original}",
        })
    season = original.parent
    base = original.stem
    dubbed = season / "doblajes" / f"{base}.mkv"
    if not dubbed.is_file():
        raise HTTPException(status_code=409, detail={
            "code": "dubbed_missing",
            "message": f"Dubbed file not found: {dubbed}",
        })
    output = season / f"{base}.mkv"
    if output.exists() and output.resolve() != original.resolve():
        # Ya existe un ``<name>.mkv`` separado del original ``.mp4`` —
        # probablemente un resultado parcial de una corrida previa.
        # Rechazamos en lugar de sobrescribir; requiere limpieza manual.
        raise HTTPException(status_code=409, detail={
            "code": "already_promoted",
            "message": (
                f"Output collision: {output} already exists. Move or delete it "
                "before promoting."
            ),
        })
    output_tmp = season / f"{base}.mkv.tmp"

    def _existing(*candidates: Path) -> Optional[Path]:
        for c in candidates:
            if c.exists():
                return c
        return None

    es_srt = _existing(
        season / f"{base}.es.srt",
        season / f"{base}.ES.srt",
        season / f"{base}_ES.srt",
    )
    en_srt = _existing(
        season / f"{base}.en.srt",
        season / f"{base}.srt",
    )

    # Ficheros que se borran tras un mux exitoso. ``.bjj-meta.json`` vive
    # en la raíz del instructional (no por capítulo) — nunca se toca.
    sidecars = [
        season / f"{base}.dub.es.srt",
        season / f"{base}_VOCALS.wav",
        season / f"{base}_BACKGROUND.wav",
        season / f"{base}_ref.wav",
        season / f"{base}_AUDIO_ESP.wav",
        season / f"{base}_QA_TMP.wav",
        season / f"{base}.words.json",
        season / f"{base}.dub-qa.json",
    ]
    if es_srt is not None:
        sidecars.append(es_srt)
    if en_srt is not None:
        sidecars.append(en_srt)

    return Inputs(
        original=original, dubbed=dubbed,
        output=output, output_tmp=output_tmp,
        es_srt=es_srt, en_srt=en_srt,
        sidecars_to_delete=sidecars,
    )


# ---------------------------------------------------------------------------
# Construcción del comando ffmpeg (puro)
# ---------------------------------------------------------------------------


def build_ffmpeg_argv(inp: Inputs) -> list[str]:
    """Construye la línea de comandos de ffmpeg para el mux multi-track.

    Layout de streams en el output:
      v:0  ← vídeo doblado     (input 0, stream copy)
      a:0  ← Español doblado   (input 0, disposition default)
      a:1  ← Inglés original   (input 1)
      s:0  ← .srt español      (input 2 si es_srt) — default
      s:1  ← .srt inglés       (input 3 si en_srt)
    """
    argv: list[str] = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(inp.dubbed),
        "-i", str(inp.original),
    ]
    sub_inputs: list[Path] = []
    if inp.es_srt is not None:
        argv += ["-i", str(inp.es_srt)]
        sub_inputs.append(inp.es_srt)
    if inp.en_srt is not None:
        argv += ["-i", str(inp.en_srt)]
        sub_inputs.append(inp.en_srt)

    # Maps. ``?`` hace el map opcional (sin error si el stream de audio
    # falta en input 1 — improbable pero barato como seguro).
    argv += ["-map", "0:v:0", "-map", "0:a:0?", "-map", "1:a:0?"]
    for idx, _ in enumerate(sub_inputs, start=2):
        argv += ["-map", f"{idx}:0?"]

    # Stream copy — sin re-encode. Los dos audios son AAC del pipeline /
    # source; los subs son SRT y MKV los soporta nativos.
    argv += ["-c:v", "copy", "-c:a", "copy", "-c:s", "copy"]

    # Metadatos de idioma + título por audio.
    argv += [
        "-metadata:s:a:0", "language=spa",
        "-metadata:s:a:0", "title=Español (doblaje IA)",
        "-metadata:s:a:1", "language=eng",
        "-metadata:s:a:1", "title=English (original)",
    ]
    argv += ["-disposition:a:0", "default", "-disposition:a:1", "0"]

    # Metadatos de subtítulos siguiendo el orden de ``sub_inputs``.
    for s_idx, src in enumerate(sub_inputs):
        if src is inp.es_srt:
            argv += [
                f"-metadata:s:s:{s_idx}", "language=spa",
                f"-metadata:s:s:{s_idx}", "title=Español",
                f"-disposition:s:{s_idx}", "default",
            ]
        else:  # en_srt
            argv += [
                f"-metadata:s:s:{s_idx}", "language=eng",
                f"-metadata:s:s:{s_idx}", "title=English",
                f"-disposition:s:{s_idx}", "0",
            ]

    # Forzamos formato matroska — ffmpeg no lo infiere desde la
    # extensión ``.tmp``.
    argv += ["-f", "matroska", str(inp.output_tmp)]
    return argv


# ---------------------------------------------------------------------------
# Pipeline guard (lazy import contra ciclos)
# ---------------------------------------------------------------------------


def pipeline_active_for(paths: list[Path]) -> Optional[str]:
    """Devuelve el ``pipeline_id`` de un pipeline RUNNING cuyo target intersecte
    con *paths*, o ``None``. Import diferido para evitar ciclo en import time.
    """
    try:
        from api.pipeline import _pipelines, StepStatus
    except ImportError:
        return None
    targets = {str(p.resolve()) for p in paths}
    for pid, pipe in _pipelines.items():
        if pipe.status != StepStatus.RUNNING:
            continue
        try:
            ppath = str(Path(pipe.path).resolve())
        except OSError:
            continue
        if ppath in targets:
            return pid
        # Un pipeline apuntando a la carpeta de season también bloquea
        # operaciones por capítulo dentro de ella.
        for t in targets:
            if t.startswith(ppath + "/") or t.startswith(ppath + "\\"):
                return pid
    return None


# ---------------------------------------------------------------------------
# Servicio
# ---------------------------------------------------------------------------


class PromoteService:
    """Orquesta el remux + cleanup. Mantiene los locks por season en
    instancia (no globales).

    ``library_path_loader`` y ``cache_factory`` se inyectan por constructor
    para no acoplar el módulo a ``api.settings`` ni a ``api.scan_cache`` en
    tiempo de import. ``get_promote_service`` mantiene un singleton
    scope-app para compartir los locks entre requests del mismo proceso.
    """

    def __init__(
        self,
        *,
        library_path_loader: Optional[Callable[[], Optional[str]]] = None,
        cache_factory: Optional[Callable[[], object]] = None,
        refresh_flags: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._season_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._library_path_loader = library_path_loader
        self._cache_factory = cache_factory
        self._refresh_flags = refresh_flags

    # ------------------------------------------------------------------
    # Locks por season (atributos de instancia, no globales)
    # ------------------------------------------------------------------

    def _season_lock(self, season_dir: Path) -> threading.Lock:
        key = str(season_dir)
        with self._locks_guard:
            lock = self._season_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._season_locks[key] = lock
            return lock

    # ------------------------------------------------------------------
    # ffmpeg
    # ------------------------------------------------------------------

    def _run_ffmpeg(self, argv: list[str]) -> None:
        """Invoca ffmpeg; lanza ``HTTPException(500)`` con tail del stderr en fallo."""
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail={
                "code": "ffmpeg_missing",
                "message": f"ffmpeg not in PATH: {exc}",
            })
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail={
                "code": "ffmpeg_timeout",
                "message": "ffmpeg exceeded 600 s — file may be huge or stuck",
            })
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()[-15:]
            log.error("ffmpeg failed (rc=%d):\n%s", r.returncode, "\n".join(tail))
            raise HTTPException(status_code=500, detail={
                "code": "ffmpeg_failed",
                "message": "ffmpeg returned non-zero",
                "stderr_tail": tail,
            })

    # ------------------------------------------------------------------
    # Refresh cache (lazy: depende de api.settings / api.scan_cache)
    # ------------------------------------------------------------------

    def _refresh_cache_for(self, video_path: Path) -> None:
        """Re-stat del instructional que contiene *video_path* para que el
        nuevo .mkv (con audio ES embebido) reemplace la entrada en cache.

        Best-effort: cualquier excepción se loguea como warning y no
        rompe la promoción.
        """
        try:
            cache_factory = self._cache_factory
            refresh_flags = self._refresh_flags
            lib_loader = self._library_path_loader
            if cache_factory is None or refresh_flags is None:
                # Sin inyección (p.ej. en tests aislados) saltamos el
                # refresh — el endpoint sigue funcionando.
                return

            cache = cache_factory()
            data = cache.load()
            if not data:
                return
            items = data.get("instructionals", []) if isinstance(data, dict) else []
            # Subimos desde el video hasta encontrar la raíz del
            # instructional (hijo directo de ``library_path``).
            lib = lib_loader() if lib_loader else None
            folder = video_path.parent
            if lib:
                lib_p = Path(lib)
                while folder.parent != lib_p and folder.parent != folder:
                    folder = folder.parent
            folder_str = str(folder)
            match = next(
                (it for it in items if it.get("path") == folder_str),
                None,
            )
            if match is None:
                return
            refresh_flags(match)
            cache.save(items)
        except Exception:  # noqa: BLE001
            log.warning("Could not refresh cache after promote", exc_info=True)

    # ------------------------------------------------------------------
    # Operaciones públicas
    # ------------------------------------------------------------------

    def promote_one(self, original_path: str) -> dict:
        """Mux + cleanup de un único capítulo. Lanza ``HTTPException`` en error."""
        inp = resolve_inputs(original_path)

        active = pipeline_active_for([inp.original, inp.dubbed])
        if active:
            raise HTTPException(status_code=409, detail={
                "code": "pipeline_active",
                "message": f"Pipeline {active} is running on this video",
            })

        season = inp.original.parent
        with self._season_lock(season):
            argv = build_ffmpeg_argv(inp)
            log.info("Promoting %s → %s  cmd: %s", inp.dubbed.name, inp.output.name, " ".join(argv))
            try:
                self._run_ffmpeg(argv)
                # Swap atómico: reemplazamos cualquier ``.tmp`` previo (ya
                # validamos que ``inp.output`` no existe o es el propio
                # original).
                inp.output_tmp.replace(inp.output)
            except Exception:
                # ffmpeg falló o ``replace`` explotó — barremos el ``.tmp``
                # para que la siguiente llamada no tropiece con basura.
                try:
                    if inp.output_tmp.exists():
                        inp.output_tmp.unlink()
                except OSError:
                    pass
                raise

            # Mux exitoso. Ahora borramos los inputs. El orden importa:
            #   1. Vídeo original (el ``.mp4`` que comparte stem con el
            #      output ``.mkv``). Si tienen el mismo path exacto
            #      (.mkv original), el ``.replace`` ya lo sustituyó.
            deleted: list[str] = []
            if inp.original.exists() and inp.original.resolve() != inp.output.resolve():
                try:
                    inp.original.unlink()
                    deleted.append(str(inp.original))
                except OSError as exc:
                    log.warning("Could not delete original %s: %s", inp.original, exc)

            # 2. Doblado en ``doblajes/``, y la propia carpeta ``doblajes/``
            #    si queda vacía.
            try:
                inp.dubbed.unlink()
                deleted.append(str(inp.dubbed))
            except OSError as exc:
                log.warning("Could not delete dubbed %s: %s", inp.dubbed, exc)
            try:
                doblajes_dir = inp.dubbed.parent
                if doblajes_dir.is_dir() and not any(doblajes_dir.iterdir()):
                    doblajes_dir.rmdir()
                    deleted.append(str(doblajes_dir))
            except OSError:
                pass

            # 3. Sidecars (best effort; cualquier fallo se loguea pero no
            #    se propaga).
            for sc in inp.sidecars_to_delete:
                try:
                    if sc.exists():
                        sc.unlink()
                        deleted.append(str(sc))
                except OSError as exc:
                    log.warning("Could not delete sidecar %s: %s", sc, exc)

        # Refrescamos la cache para que la UI vea el nuevo estado sin
        # esperar a un rescan manual.
        self._refresh_cache_for(inp.output)

        return {
            "ok": True,
            "output_path": str(inp.output),
            "deleted": deleted,
            "muxed_streams": {
                "audio": ["spa", "eng"],
                "subs": [
                    tag for tag, src in (("spa", inp.es_srt), ("eng", inp.en_srt))
                    if src is not None
                ],
            },
        }

    def promote_season(self, season_path: str) -> dict:
        """Promueve cada capítulo doblado bajo ``season_path``. Secuencial."""
        season = Path(season_path)
        if not season.is_dir():
            raise HTTPException(status_code=409, detail={
                "code": "season_missing",
                "message": f"Season folder not found: {season}",
            })
        doblajes = season / "doblajes"
        if not doblajes.is_dir():
            return {"promoted": [], "skipped": [], "failed": [],
                    "message": "Nothing to promote — no doblajes/ folder"}

        candidates: list[Path] = []
        for f in sorted(season.iterdir()):
            if not f.is_file() or f.suffix.lower() not in _VIDEO_EXTS:
                continue
            if (doblajes / f"{f.stem}.mkv").exists():
                candidates.append(f)

        promoted: list[dict] = []
        skipped: list[dict] = []
        failed: list[dict] = []
        for orig in candidates:
            try:
                result = self.promote_one(str(orig))
                promoted.append({"path": str(orig), "output": result["output_path"]})
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
                bucket = skipped if exc.status_code == 409 else failed
                bucket.append({"path": str(orig), **detail})
            except Exception as exc:  # noqa: BLE001
                log.exception("Promote failed for %s", orig)
                failed.append({"path": str(orig), "code": "unexpected", "message": str(exc)})

        return {
            "promoted": [p["path"] for p in promoted],
            "skipped": skipped,
            "failed": failed,
            "promoted_count": len(promoted),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
        }
