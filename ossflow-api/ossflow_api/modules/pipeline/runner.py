"""Runner del pipeline: ejecuta steps y pipelines.

Migrado de ``api/pipeline.py`` en T_LATE_2.5c.

El state container (``_pipelines``, ``_pipeline_subscribers``,
``_pipeline_tasks``, ``_pipeline_cancel``, ``_batches``,
``_batch_tasks``, ``_batch_cancel``) sigue en el shim ``api/pipeline.py``
porque varios tests parchean esos dicts directamente. Este runner
los lee/escribe via ``import api.pipeline as _pmod`` lazy.

El acceso lazy también rompe el ciclo
``api.pipeline → modules.pipeline.runner → api.pipeline``: el módulo
shim importa runner al cargar, y runner solo accede al shim cuando
ejecuta una función (ya completamente importado).

Funciones públicas:

* ``run_step(pipeline, step_index, queue)`` — ejecuta un step,
  incluyendo skip-detection, snapshot/diff y emisión de eventos.
* ``run_pipeline(pipeline, queue)`` — secuencia los steps con
  flush GPU entre subtitles/dubbing y manejo de cancelación.
* ``flush_gpu_after_step(pipeline, queue)`` — restart del
  subtitle-generator + health-poll.
* ``refresh_scan_cache_for(pipeline_path)`` — refresh del library
  cache tras completar un pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from api.backend_client import BackendError
from api.event_normalizer import normalize

from . import store as _store_mod
from .schemas import PipelineInfo, StepStatus

log = logging.getLogger(__name__)


def _shim():
    """Resuelve el shim ``api.pipeline`` perezosamente.

    El runner usa ``_shim()._pipelines`` etc. en lugar de capturar
    los dicts en variables locales para que los monkeypatches de
    los tests (que reasignan los atributos del shim) afecten al
    runner sin necesidad de re-importar.
    """
    import api.pipeline as _pmod  # noqa: PLC0415
    return _pmod


async def _emit(pipeline: PipelineInfo, queue: asyncio.Queue, event: dict) -> None:
    """Wrapper interno: usa el _emit del shim para que tests que
    parchean ``api.pipeline._emit`` afecten también al runner."""
    await _shim()._emit(pipeline, queue, event)


DUBBING_MAX_RETRIES = 2
DUBBING_RETRY_BACKOFF_S = 30.0


async def run_step(
    pipeline: PipelineInfo,
    step_index: int,
    queue: asyncio.Queue,
) -> bool:
    """Ejecuta un step via su microservicio backend.

    Resilience plan (2026-05-02): el step ``dubbing`` tiene retry
    automático cuando el filesystem revela output parcial (capítulos
    pendientes) tras un terminal event ``done`` o un cierre de stream
    sin terminal. Hasta ``DUBBING_MAX_RETRIES`` reintentos, separados
    por ``DUBBING_RETRY_BACKOFF_S`` y un health-poll del backend. El
    payload de retry tiene ``force=False`` hardcoded para no destruir
    capítulos buenos.

    TODO: chapters/subtitles/translate tienen el mismo bug latente
    (si el container muere a mitad de un job, podemos asumir éxito
    sin que esté completo). El retry pattern de aquí es scoped a
    dubbing porque es donde el síntoma es más caro (~70 min) y porque
    el filesystem-check es trivial vía ``season_already_dubbed``. Para
    los otros steps habría que escribir verificadores equivalentes.
    """
    pmod = _shim()
    step = pipeline.steps[step_index]
    step.status = StepStatus.RUNNING
    step.started_at = datetime.now(timezone.utc).isoformat()
    pipeline.current_step = step_index

    target = pmod._target_dir(pipeline)
    before_snap = pmod._snapshot_dir(target)
    diff_emitted = {"done": False}

    # Skip si la Season ya tiene el artefacto que iba a producir el step
    # (subs EN/ES, audio dubbed) y no se pidió ``force``.
    if not pipeline.options.get("force") and step.name in ("subtitles", "translate", "dubbing"):
        season_dir = Path(pipeline.chained_path) if pipeline.chained_path else target
        if season_dir is not None:
            skip_info: Optional[tuple[str, int, int]] = None
            if step.name == "subtitles":
                ok, n, total = pmod._season_already_subbed_en(season_dir)
                if ok and total > 0:
                    skip_info = ("subtítulos EN", n, total)
            elif step.name == "translate":
                from api.settings import get_setting
                if "dubbing_mode" in pipeline.options:
                    dub_on = bool(pipeline.options["dubbing_mode"])
                else:
                    dub_on = bool(get_setting("translation_dubbing_mode"))
                ok, n, total = pmod._season_already_subbed_es(season_dir, dub_on)
                if ok and total > 0:
                    label = "subtítulos ES (dub)" if dub_on else "subtítulos ES"
                    skip_info = (label, n, total)
            elif step.name == "dubbing":
                ok, n, total = pmod._season_already_dubbed(season_dir)
                if ok and total > 0:
                    skip_info = ("audio ES", n, total)

            if skip_info is not None:
                label, n, total = skip_info
                step.status = StepStatus.SKIPPED
                step.progress = 100.0
                step.completed_at = datetime.now(timezone.utc).isoformat()
                step.message = f"Skipped — {n}/{total} chapters ya tienen {label}"
                await _emit(pipeline, queue, {
                    "type": "step_skipped",
                    "step": step.name,
                    "step_index": step_index,
                    "message": step.message,
                    "progress": 100,
                })
                step.diff = {"added": [], "modified": [], "removed": [], "truncated": False}
                diff_emitted["done"] = True
                await _emit(pipeline, queue, {
                    "type": "step_diff",
                    "step": step.name,
                    "step_index": step_index,
                    **step.diff,
                })
                return True

    async def _finalize_diff() -> None:
        if diff_emitted["done"]:
            return
        diff_emitted["done"] = True
        after_snap = pmod._snapshot_dir(target)
        diff = pmod._compute_diff(before_snap, after_snap)
        step.diff = diff
        # Tras chapters: si detectamos Season_NN/ creada, redirigimos los
        # pasos siguientes a esa carpeta (donde viven los capítulos reales).
        if step.name == "chapters" and step.status == StepStatus.COMPLETED:
            season_path = pmod._detect_season_folder(target, diff.get("added", []))
            if season_path:
                pipeline.chained_path = season_path
                await _emit(pipeline, queue, {
                    "type": "log",
                    "data": {"message": f"Pipeline chained to: {season_path}"},
                })
        await _emit(pipeline, queue, {
            "type": "step_diff",
            "step": step.name,
            "step_index": step_index,
            **diff,
        })

    await _emit(pipeline, queue, {
        "type": "step_started",
        "step": step.name,
        "step_index": step_index,
        "total_steps": len(pipeline.steps),
        "progress": 0,
    })

    try:
        client, payload, use_oracle = pmod._client_and_payload(
            step.name, pipeline.path, pipeline.options, pipeline.chained_path,
        )
        log.info(
            "[pipeline:%s] Delegating %s to %s%s",
            pipeline.pipeline_id, step.name, client.base_url,
            " (oracle)" if use_oracle else "",
        )

        remote_id = await (
            client.run_oracle(payload) if use_oracle else client.run(payload)
        )

        async for evt in client.stream(remote_id):
            if isinstance(evt, dict):
                evt = normalize(evt)
            msg = evt.message or ""
            if msg:
                step.message = msg
            if evt.progress is not None:
                step.progress = evt.progress

            if evt.kind == "error":
                step.status = StepStatus.FAILED
                step.completed_at = datetime.now(timezone.utc).isoformat()
                step.message = evt.message or "backend error"
                await _emit(pipeline, queue, {
                    "type": "step_failed",
                    "step": step.name,
                    "step_index": step_index,
                    "message": step.message,
                })
                return False

            if evt.kind == "done":
                # Dubbing-only: antes de marcar COMPLETED, verificar
                # filesystem; si hay capítulos pendientes, reintentar
                # hasta DUBBING_MAX_RETRIES. Otros steps caen directo.
                if step.name == "dubbing":
                    handled = await _dubbing_retry_if_partial(
                        pipeline, step, step_index, queue,
                    )
                    if handled is not None:
                        return handled
                step.status = StepStatus.COMPLETED
                step.progress = 100.0
                step.completed_at = datetime.now(timezone.utc).isoformat()
                await _emit(pipeline, queue, {
                    "type": "step_completed",
                    "step": step.name,
                    "step_index": step_index,
                    "progress": 100,
                })
                return True

            if msg or evt.progress is not None:
                await _emit(pipeline, queue, {
                    "type": "step_progress",
                    "step": step.name,
                    "step_index": step_index,
                    "message": msg,
                    "progress": step.progress,
                })

        # Stream cerrado sin terminal event. Antes asumíamos éxito;
        # ahora, para dubbing, verificamos filesystem y reintentamos
        # si hay capítulos pendientes (resilience plan 2026-05-02).
        # Para chapters/subtitles/translate sigue siendo "asumir éxito"
        # — ver TODO al inicio de run_step.
        if step.name == "dubbing":
            handled = await _dubbing_retry_if_partial(
                pipeline, step, step_index, queue,
            )
            if handled is not None:
                return handled
        step.status = StepStatus.COMPLETED
        step.progress = 100.0
        step.completed_at = datetime.now(timezone.utc).isoformat()
        await _emit(pipeline, queue, {
            "type": "step_completed",
            "step": step.name,
            "step_index": step_index,
            "progress": 100,
        })
        return True

    except asyncio.CancelledError:
        step.status = StepStatus.CANCELLED
        step.completed_at = datetime.now(timezone.utc).isoformat()
        step.message = "cancelled by user"
        await _emit(pipeline, queue, {
            "type": "step_failed",
            "step": step.name,
            "step_index": step_index,
            "message": "cancelled by user",
        })
        raise
    except BackendError as exc:
        # Resilience: para dubbing, un BackendError (incluído el 404
        # cuando el container murió y se reapeó) puede ser "tenemos
        # 8/12 capítulos en disco, intenta los 4 restantes". Verificar
        # filesystem y reintentar antes de marcar FAILED.
        if step.name == "dubbing":
            log.warning(
                "[pipeline:%s] dubbing BackendError: %s — verificando "
                "filesystem para retry",
                pipeline.pipeline_id, exc,
            )
            step.message = f"backend error: {exc}"
            handled = await _dubbing_retry_if_partial(
                pipeline, step, step_index, queue,
            )
            if handled is not None:
                return handled
        step.status = StepStatus.FAILED
        step.completed_at = datetime.now(timezone.utc).isoformat()
        step.message = f"backend error: {exc}"
        await _emit(pipeline, queue, {
            "type": "step_failed",
            "step": step.name,
            "step_index": step_index,
            "message": step.message,
        })
        return False
    except Exception as exc:
        step.status = StepStatus.FAILED
        step.completed_at = datetime.now(timezone.utc).isoformat()
        step.message = str(exc)
        await _emit(pipeline, queue, {
            "type": "step_failed",
            "step": step.name,
            "step_index": step_index,
            "message": str(exc),
        })
        return False
    finally:
        try:
            await _finalize_diff()
        except Exception as exc:
            log.warning("Failed to compute step diff: %s", exc)


async def flush_gpu_after_step(
    pipeline: PipelineInfo, queue: asyncio.Queue,
) -> None:
    """Restart subtitle-generator (libera VRAM) + health-poll hasta 60s."""
    import httpx
    pmod = _shim()
    subs_url = pmod.subs_client().base_url
    log.info(
        "[pipeline:%s] Flushing GPU (restarting subtitle-generator)…",
        pipeline.pipeline_id,
    )
    await _emit(pipeline, queue, {"type": "log", "message": "Liberando VRAM…"})
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{subs_url}/maintenance/restart")
    except Exception:
        pass  # service kills itself before responding
    for _ in range(30):
        await asyncio.sleep(2.0)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{subs_url}/health")
                if r.status_code == 200:
                    log.info(
                        "[pipeline:%s] subtitle-generator healthy again.",
                        pipeline.pipeline_id,
                    )
                    return
        except Exception:
            pass
    log.warning(
        "[pipeline:%s] subtitle-generator did not recover in 60s after GPU flush.",
        pipeline.pipeline_id,
    )


async def run_pipeline(pipeline: PipelineInfo, queue: asyncio.Queue) -> None:
    """Ejecuta los steps secuencialmente con flush GPU entre subtitles/dubbing."""
    pmod = _shim()
    pipeline.status = StepStatus.RUNNING
    await _emit(pipeline, queue, {
        "type": "pipeline_started",
        "pipeline_id": pipeline.pipeline_id,
    })

    try:
        for i, step in enumerate(pipeline.steps):
            if pmod._pipeline_cancel.get(pipeline.pipeline_id):
                for j in range(i, len(pipeline.steps)):
                    pipeline.steps[j].status = StepStatus.CANCELLED
                pipeline.status = StepStatus.CANCELLED
                pipeline.completed_at = datetime.now(timezone.utc).isoformat()
                await _emit(pipeline, queue, {
                    "type": "pipeline_failed",
                    "pipeline_id": pipeline.pipeline_id,
                    "message": "cancelled by user",
                })
                return
            success = await pmod._run_step(pipeline, i, queue)
            if success and step.name in ("subtitles", "dubbing"):
                await flush_gpu_after_step(pipeline, queue)
            if not success:
                for j in range(i + 1, len(pipeline.steps)):
                    pipeline.steps[j].status = StepStatus.SKIPPED
                pipeline.status = StepStatus.FAILED
                pipeline.completed_at = datetime.now(timezone.utc).isoformat()
                await _emit(pipeline, queue, {
                    "type": "pipeline_failed",
                    "pipeline_id": pipeline.pipeline_id,
                    "failed_step": step.name,
                    "message": step.message,
                })
                return

        pipeline.status = StepStatus.COMPLETED
        pipeline.completed_at = datetime.now(timezone.utc).isoformat()
        await _emit(pipeline, queue, {
            "type": "pipeline_completed",
            "pipeline_id": pipeline.pipeline_id,
        })
    except asyncio.CancelledError:
        pipeline.status = StepStatus.CANCELLED
        pipeline.completed_at = datetime.now(timezone.utc).isoformat()
        for s in pipeline.steps:
            if s.status in (StepStatus.PENDING, StepStatus.RUNNING):
                s.status = StepStatus.CANCELLED
        await _emit(pipeline, queue, {
            "type": "pipeline_failed",
            "pipeline_id": pipeline.pipeline_id,
            "message": "cancelled by user",
        })
    finally:
        pmod._pipeline_cancel.pop(pipeline.pipeline_id, None)
        pmod._pipeline_tasks.pop(pipeline.pipeline_id, None)
        pmod._save_history()
        pmod._refresh_scan_cache_for(pipeline.path)


async def run_batch(batch) -> None:
    """Lanza secuencialmente un pipeline por cada season del batch."""
    pmod = _shim()
    batch.status = StepStatus.RUNNING
    log.info(
        "[batch %s] starting %d seasons", batch.batch_id, len(batch.paths),
    )

    for idx, path in enumerate(batch.paths):
        if pmod._batch_cancel.get(batch.batch_id):
            log.info(
                "[batch %s] cancelled before season %d",
                batch.batch_id, idx + 1,
            )
            batch.status = StepStatus.CANCELLED
            batch.completed_at = datetime.now(timezone.utc).isoformat()
            return

        batch.current_index = idx
        log.info(
            "[batch %s] launching season %d/%d: %s",
            batch.batch_id, idx + 1, len(batch.paths), path,
        )

        pipeline, err, _code = pmod._launch_pipeline_internal(
            path, batch.steps, dict(batch.options),
        )
        if err is not None:
            msg = f"season {idx + 1} launch failed: {err.get('error', err)}"
            log.warning("[batch %s] %s", batch.batch_id, msg)
            batch.last_error = msg
            if not batch.continue_on_fail:
                batch.status = StepStatus.FAILED
                batch.completed_at = datetime.now(timezone.utc).isoformat()
                return
            continue

        batch.pipeline_ids.append(pipeline.pipeline_id)
        task = pmod._pipeline_tasks.get(pipeline.pipeline_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                log.info(
                    "[batch %s] pipeline %s cancelled",
                    batch.batch_id, pipeline.pipeline_id,
                )
            except Exception as exc:
                log.exception(
                    "[batch %s] pipeline %s crashed: %s",
                    batch.batch_id, pipeline.pipeline_id, exc,
                )

        final = pmod._pipelines.get(pipeline.pipeline_id)
        final_status = final.status if final else StepStatus.FAILED
        log.info(
            "[batch %s] season %d/%d → %s",
            batch.batch_id, idx + 1, len(batch.paths), final_status.value,
        )

        if final_status == StepStatus.FAILED:
            batch.last_error = f"season {idx + 1} pipeline failed"
            if not batch.continue_on_fail:
                batch.status = StepStatus.FAILED
                batch.completed_at = datetime.now(timezone.utc).isoformat()
                return
        elif final is not None and final_status == StepStatus.COMPLETED:
            # Resilience plan (2026-05-02): un pipeline puede acabar
            # COMPLETED con dubbing parcial (tras retries irreparables).
            # Surfacearlo en batch.last_error para que el batch nocturno
            # lo destaque al usuario aunque el pipeline en sí no esté
            # marcado FAILED.
            for s in final.steps:
                if (
                    s.name == "dubbing"
                    and s.status == StepStatus.COMPLETED
                    and s.message
                    and s.message.startswith("Doblaje parcial")
                ):
                    label = Path(path).name
                    batch.last_error = f"{label}: {s.message}"
                    log.warning(
                        "[batch %s] partial dubbing surfaced: %s",
                        batch.batch_id, batch.last_error,
                    )
                    break

        if pmod._batch_cancel.get(batch.batch_id):
            batch.status = StepStatus.CANCELLED
            batch.completed_at = datetime.now(timezone.utc).isoformat()
            return

    batch.current_index = len(batch.paths)
    batch.status = StepStatus.COMPLETED
    batch.completed_at = datetime.now(timezone.utc).isoformat()
    log.info(
        "[batch %s] completed all %d seasons",
        batch.batch_id, len(batch.paths),
    )


async def _dubbing_retry_if_partial(
    pipeline: PipelineInfo,
    step,
    step_index: int,
    queue: asyncio.Queue,
) -> Optional[bool]:
    """Verifica capítulos doblados en disco; reintenta hasta MAX_RETRIES.

    Llamado desde ``run_step`` cuando dubbing entra en un terminal
    legítimo (``done`` o stream cerrado limpio) o un ``BackendError``
    (e.g. 404 porque el container del backend murió).

    Returns:
        * ``None`` si no procede retry (todos completos, total==0, o
          step no es dubbing). El caller continúa con la lógica normal
          de COMPLETED.
        * ``True`` si tras retries (o sin retries necesarios pero con
          partial irreparable) ha emitido ``step_completed`` con
          warning. El caller debe ``return True`` directamente.
        * ``False`` si el retry fue cancelado por el usuario. El caller
          debe ``return False``.

    El payload de retry usa ``force=False`` hardcoded — nunca borrar
    capítulos buenos durante retries.
    """
    pmod = _shim()
    from .skip_detector import season_already_dubbed

    target = pmod._target_dir(pipeline)
    season_dir = Path(pipeline.chained_path) if pipeline.chained_path else target
    if season_dir is None or not season_dir.is_dir():
        return None

    all_done, dubbed, total = season_already_dubbed(season_dir)
    if total == 0 or all_done:
        return None  # nada que reintentar

    attempts_done = pipeline.options.get("_dubbing_attempts", 0)

    for retry_idx in range(attempts_done, DUBBING_MAX_RETRIES):
        # Cancelación entre intentos
        if pmod._pipeline_cancel.get(pipeline.pipeline_id):
            step.status = StepStatus.CANCELLED
            step.completed_at = datetime.now(timezone.utc).isoformat()
            step.message = "cancelled by user"
            await _emit(pipeline, queue, {
                "type": "step_failed",
                "step": step.name,
                "step_index": step_index,
                "message": "cancelled by user",
            })
            return False

        await asyncio.sleep(DUBBING_RETRY_BACKOFF_S)

        # Health-poll del backend (mismo patrón que flush_gpu_after_step)
        client_for_retry, payload_for_retry, use_oracle_retry = pmod._client_and_payload(
            step.name, pipeline.path,
            {**pipeline.options, "force": False},  # NUNCA borrar capítulos buenos en retry
            pipeline.chained_path,
        )
        healthy = False
        for _ in range(20):  # 20 * 5s = 100s window
            try:
                await client_for_retry.health()
                healthy = True
                break
            except Exception:
                await asyncio.sleep(5)
        if not healthy:
            step.message = (
                f"Doblaje parcial: {dubbed}/{total} capítulos. "
                f"Backend no responde tras {retry_idx} reintento(s)."
            )
            break

        # Re-emit log para que UI vea el retry
        await _emit(pipeline, queue, {
            "type": "log",
            "data": {
                "message": (
                    f"Reintentando dubbing ({retry_idx + 1}/{DUBBING_MAX_RETRIES}). "
                    f"Pendientes: {total - dubbed}/{total}"
                ),
            },
        })

        # Mark this attempt so a subsequent invocation (e.g. another
        # run_step on the same pipeline) doesn't re-burn retries.
        pipeline.options["_dubbing_attempts"] = retry_idx + 1

        try:
            remote_id2 = await (
                client_for_retry.run_oracle(payload_for_retry)
                if use_oracle_retry
                else client_for_retry.run(payload_for_retry)
            )
            async for evt2 in client_for_retry.stream(remote_id2):
                if isinstance(evt2, dict):
                    evt2 = normalize(evt2)
                if evt2.message:
                    step.message = evt2.message
                if evt2.progress is not None:
                    step.progress = evt2.progress
                if evt2.kind in ("done", "error"):
                    break
                if evt2.message or evt2.progress is not None:
                    await _emit(pipeline, queue, {
                        "type": "step_progress",
                        "step": step.name,
                        "step_index": step_index,
                        "message": evt2.message or "",
                        "progress": step.progress,
                    })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(
                "[pipeline:%s] dubbing retry %d failed: %s",
                pipeline.pipeline_id, retry_idx + 1, exc,
            )

        # Re-verificar capítulos en disco
        all_done, dubbed, total = season_already_dubbed(season_dir)
        if all_done:
            break

    if all_done:
        return None  # caller toma la rama COMPLETED normal

    # Tras todos los retries posibles: marcar COMPLETED con warning
    pendientes = total - dubbed
    step.message = (
        f"Doblaje parcial: {dubbed}/{total} capítulos. "
        f"Pendientes: {pendientes}. (tras {DUBBING_MAX_RETRIES} reintentos)"
    )
    step.status = StepStatus.COMPLETED
    step.progress = 100.0
    step.completed_at = datetime.now(timezone.utc).isoformat()
    await _emit(pipeline, queue, {
        "type": "step_completed",
        "step": step.name,
        "step_index": step_index,
        "progress": 100,
        "message": step.message,
        "warning": True,  # señal extra para UI si la quiere usar
    })
    return True


def refresh_scan_cache_for(pipeline_path: str) -> None:
    """Re-discover de vídeos en la carpeta del instructional + actualizar cache."""
    try:
        from api.settings import get_library_path
        from ossflow_api.modules.library.dependencies import get_library_cache
        from ossflow_api.modules.library.refresh import rediscover_instructional

        cache = get_library_cache()
        data = cache.load()
        if not data:
            return
        items = data.get("instructionals", []) if isinstance(data, dict) else []

        p = Path(pipeline_path)
        folder = p if p.is_dir() else p.parent
        lib = get_library_path()
        if lib:
            lib_p = Path(lib)
            while folder.parent != lib_p and folder.parent != folder:
                folder = folder.parent

        folder_str = str(folder)
        match = next(
            (
                it for it in items
                if it.get("path")
                and (
                    it["path"] == folder_str
                    or Path(it["path"]).resolve() == Path(folder_str).resolve()
                )
            ),
            None,
        )
        if not match:
            folder_name = folder.name
            match = next((it for it in items if it.get("name") == folder_name), None)

        if match:
            rediscover_instructional(match)
            cache.save(items)
            log.info(
                "Scan cache refreshed for %s after pipeline",
                match.get("name"),
            )
    except Exception:
        log.warning("Failed to refresh scan cache after pipeline", exc_info=True)
