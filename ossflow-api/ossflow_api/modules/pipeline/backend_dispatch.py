"""Mapeo step → microservicio HTTP.

Migrado de ``api/pipeline.py`` en T_LATE_2.4.

Aísla el acoplamiento al `backend_client` y a `api.settings`. Cada
step se traduce a un ``(client, payload, use_oracle)`` que el
runner consume sin saber qué microservicio hay detrás.

* ``client_and_payload(step_name, path, options, chained_path)``
  — dispatcher principal con un if/elif por step.
* ``load_oracle_for_path(host_path)`` — lee oracle del sidecar.
* ``load_voice_profile_for_path`` — alias del shared.

No se convierte a clases polimórficas (un ``Step`` por archivo)
porque sería un refactor independiente — el if/elif es claro y los
tests no parchean steps individualmente.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from api.backend_client import BackendClient
from api.paths import to_container_path
from ossflow_api.shared.voice_profiles import (
    load_voice_profile_for_path,  # noqa: F401  (re-exportada)
)

log = logging.getLogger(__name__)

SIDECAR_NAME = ".bjj-meta.json"


def load_oracle_for_path(host_path: str) -> Optional[dict]:
    """Lee el bloque ``oracle`` del ``.bjj-meta.json`` del instructional."""
    p = Path(host_path)
    folder = p if p.is_dir() else p.parent
    sidecar = folder / SIDECAR_NAME
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data.get("oracle") if isinstance(data, dict) else None
    except (OSError, ValueError) as exc:
        log.warning("Failed to read oracle from %s: %s", sidecar, exc)
        return None


def client_and_payload(
    step_name: str,
    path: str,
    options: dict,
    chained_path: Optional[str] = None,
) -> tuple[BackendClient, dict, bool]:
    """Devuelve ``(client, payload, use_oracle)`` para el step pedido.

    ``chained_path`` redirige los pasos posteriores a la Season_NN/
    creada por chapters; chapters siempre usa el ``path`` original.

    Si ``options["mode"] == "oracle"`` y step es ``chapters``, lee el
    oracle del sidecar y devuelve un payload para ``/run-oracle``.
    """
    # Late-bind clients + get_library_path so tests that monkeypatch
    # ``api.pipeline.splitter_client`` / ``api.pipeline.get_library_path``
    # afecten al dispatcher. Imports diferidos rompen también el ciclo
    # backend_dispatch → api.pipeline si lo hubiera.
    import api.pipeline as _pmod  # noqa: PLC0415

    effective = path if step_name == "chapters" else (chained_path or path)
    lib = _pmod.get_library_path() or ""
    container_path = to_container_path(effective, lib) if lib else effective
    video_dir = (
        container_path.rsplit("/", 1)[0]
        if "." in container_path.rsplit("/", 1)[-1]
        else container_path
    )
    if not video_dir:
        video_dir = "/library"
    user_out = options.get("output_dir")
    if user_out:
        out_dir = to_container_path(user_out, lib) if lib else user_out
    else:
        out_dir = video_dir
    base = {"input_path": container_path, "output_dir": out_dir}

    if step_name == "chapters":
        if options.get("mode") == "oracle":
            oracle_data = load_oracle_for_path(path)
            if oracle_data:
                return _pmod.splitter_client(), {
                    "path": container_path,
                    "oracle": oracle_data,
                    "output_dir": out_dir,
                }, True
            raise ValueError(
                f"Oracle mode requested but no oracle data found for '{path}'. "
                "Run Oracle first from the instructional detail page."
            )
        return _pmod.splitter_client(), {
            **base,
            "options": {
                "dry_run": bool(options.get("dry_run", False)),
                "verbose": True,
            },
        }, False

    if step_name == "subtitles":
        from api.settings import get_setting

        sub_opts: dict[str, Any] = {"verbose": True}
        if options.get("force"):
            sub_opts["force"] = True
        if "postprocess_openai" in options:
            pp_on = bool(options["postprocess_openai"])
        else:
            pp_on = bool(get_setting("subtitle_postprocess_openai"))
        if pp_on:
            sub_opts["postprocess_openai"] = True
            pp_model = options.get("postprocess_model") or get_setting("subtitle_postprocess_model")
            if pp_model:
                sub_opts["postprocess_model"] = pp_model
            pp_key = options.get("postprocess_api_key") or get_setting("openai_api_key")
            if pp_key:
                sub_opts["postprocess_api_key"] = pp_key
        return _pmod.subs_client(), {**base, "options": sub_opts}, False

    if step_name == "translate":
        from api.settings import get_setting

        provider = (options.get("provider") or get_setting("translation_provider") or "ollama").lower()
        fallback = (
            options.get("fallback_provider")
            or get_setting("translation_fallback_provider")
            or ""
        ).lower() or None
        model = options.get("model") or get_setting("translation_model")

        topts: dict[str, Any] = {
            "translate_only": True,
            "verbose": True,
            "target_lang": options.get("target_lang", "ES"),
            "source_lang": options.get("source_lang", "EN"),
            "provider": provider,
        }
        if options.get("force"):
            topts["force"] = True
        if model:
            topts["model"] = model
        if options.get("formality"):
            topts["formality"] = options["formality"]

        if "dubbing_mode" in options:
            dub_on = bool(options["dubbing_mode"])
        else:
            dub_on = bool(get_setting("translation_dubbing_mode"))
        if dub_on:
            topts["dubbing_mode"] = True
            cps = options.get("dubbing_cps") or get_setting("translation_dubbing_cps")
            if cps:
                topts["dubbing_cps"] = float(cps)

        key = options.get("api_key") or (
            get_setting("openai_api_key") if provider == "openai" else None
        )
        if key:
            topts["api_key"] = key

        if fallback and fallback != provider:
            fb_key = options.get("fallback_api_key") or (
                get_setting("openai_api_key") if fallback == "openai" else None
            )
            if fb_key:
                topts["fallback_provider"] = fallback
                topts["fallback_api_key"] = fb_key

        return _pmod.subs_client(), {**base, "options": topts}, False

    if step_name == "dubbing":
        from api.settings import get_setting

        opts: dict[str, Any] = {"skip_translation": True, "tts_engine": "s2pro"}
        if options.get("force"):
            opts["force"] = True
        voice_basename = (
            options.get("s2_voice_profile")
            or get_setting("s2_voice_profile")
            or "voice_martin_osborne_24k.wav"
        )
        opts["s2_ref_audio_path"] = f"/voices/{voice_basename}"
        ref_text = options.get("s2_ref_text") or get_setting("s2_ref_text")
        if ref_text:
            opts["s2_ref_text"] = str(ref_text)
        for k in ("s2_temperature", "s2_top_p", "s2_top_k", "s2_max_tokens"):
            v = options.get(k)
            if v is None:
                v = get_setting(k)
            if v is not None:
                opts[k] = v
        quant = (
            options.get("s2_quantization")
            or get_setting("s2_quantization")
            or "q6_k"
        )
        opts["s2_gguf_path"] = f"/models/s2pro/s2-pro-{quant}.gguf"
        opts["s2_quantization"] = quant
        return _pmod.dubbing_client(), {**base, "options": opts}, False

    raise ValueError(f"Unknown step: {step_name}")
