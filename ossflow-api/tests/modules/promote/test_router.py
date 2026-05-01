"""Tests del router del módulo promote.

Combinamos:

* Tests directos sobre helpers puros (``resolve_inputs``,
  ``build_ffmpeg_argv``) — antes vivían como ``promote._resolve_inputs``
  / ``promote._build_ffmpeg_argv`` en ``api/promote.py``.
* Tests del flujo completo (``promote_one`` y ``/api/promote/season``)
  con un ``PromoteService`` inyectado vía ``app.dependency_overrides``
  para mantener la cobertura del ffmpeg-driver y el cleanup.

Para los tests que necesitan stubear ``subprocess.run`` y
``pipeline_active_for`` se monkeypatchean los símbolos del módulo
``service`` (mismo patrón que el legacy contra ``api/promote``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ossflow_api.modules.promote import promote_router
from ossflow_api.modules.promote import service as promote_service
from ossflow_api.modules.promote.dependencies import get_promote_service
from ossflow_api.modules.promote.service import PromoteService


def _make_chapter(season: Path, name: str, *, with_subs: bool = True) -> tuple[Path, Path]:
    """Crea un par original ``.mp4`` + ``doblajes/<name>.mkv``. Devuelve (orig, dubbed)."""
    season.mkdir(parents=True, exist_ok=True)
    (season / "doblajes").mkdir(exist_ok=True)
    orig = season / f"{name}.mp4"
    dubbed = season / "doblajes" / f"{name}.mkv"
    orig.write_bytes(b"\0" * 64)
    dubbed.write_bytes(b"\0" * 64)
    if with_subs:
        (season / f"{name}.es.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nhola\n", encoding="utf-8"
        )
        (season / f"{name}.en.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8"
        )
    return orig, dubbed


def _build_service() -> PromoteService:
    """Servicio sin loaders externos: el refresh-cache es no-op en tests."""
    return PromoteService()


def _build_client(svc: PromoteService) -> TestClient:
    app = FastAPI()
    app.include_router(promote_router)
    app.dependency_overrides[get_promote_service] = lambda: svc
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers puros: resolve_inputs / build_ffmpeg_argv
# ---------------------------------------------------------------------------


def test_resolve_inputs_collects_paths(tmp_path):
    season = tmp_path / "Season 01"
    orig, dubbed = _make_chapter(season, "S01E01")
    inp = promote_service.resolve_inputs(str(orig))
    assert inp.original == orig
    assert inp.dubbed == dubbed
    assert inp.output == season / "S01E01.mkv"
    assert inp.output_tmp == season / "S01E01.mkv.tmp"
    assert inp.es_srt == season / "S01E01.es.srt"
    assert inp.en_srt == season / "S01E01.en.srt"
    # Lista todos los sidecars (existan o no — el unlink es best-effort).
    paths = [str(p) for p in inp.sidecars_to_delete]
    assert any(p.endswith("_VOCALS.wav") for p in paths)
    assert any(p.endswith(".dub-qa.json") for p in paths)


def test_resolve_inputs_missing_original(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    with pytest.raises(HTTPException) as ei:
        promote_service.resolve_inputs(str(season / "ghost.mp4"))
    assert ei.value.status_code == 409
    assert ei.value.detail["code"] == "original_missing"


def test_resolve_inputs_missing_dubbed(tmp_path):
    season = tmp_path / "Season 01"
    season.mkdir()
    orig = season / "S01E01.mp4"
    orig.write_bytes(b"\0")
    with pytest.raises(HTTPException) as ei:
        promote_service.resolve_inputs(str(orig))
    assert ei.value.detail["code"] == "dubbed_missing"


def test_resolve_inputs_already_promoted_collision(tmp_path):
    """Si ya hay un ``<name>.mkv`` junto al ``.mp4`` original, rechaza."""
    season = tmp_path / "Season 01"
    orig, _ = _make_chapter(season, "S01E01")
    # ``.mkv`` pre-existente (p.ej. corrida parcial previa)
    (season / "S01E01.mkv").write_bytes(b"\0")
    with pytest.raises(HTTPException) as ei:
        promote_service.resolve_inputs(str(orig))
    assert ei.value.detail["code"] == "already_promoted"


def test_build_ffmpeg_argv_full(tmp_path):
    season = tmp_path / "Season 01"
    orig, _ = _make_chapter(season, "S01E01")
    argv = promote_service.build_ffmpeg_argv(promote_service.resolve_inputs(str(orig)))
    # Dos audios primero (dubbed, original), luego 2 srts.
    assert argv[0] == "ffmpeg"
    # Maps deben pedir vídeo del input 0 y los dos audios.
    assert "-map" in argv and "0:v:0" in argv and "0:a:0?" in argv and "1:a:0?" in argv
    # Español marcado como default.
    idx = argv.index("-disposition:a:0")
    assert argv[idx + 1] == "default"
    # Metadatos de título presentes.
    assert "title=Español (doblaje IA)" in argv
    assert "title=English (original)" in argv
    # Ambos subtítulos mapeados como ``-map 2:0?`` y ``-map 3:0?``.
    assert "2:0?" in argv and "3:0?" in argv


def test_build_ffmpeg_argv_no_subs(tmp_path):
    season = tmp_path / "Season 01"
    orig, _ = _make_chapter(season, "S01E01", with_subs=False)
    argv = promote_service.build_ffmpeg_argv(promote_service.resolve_inputs(str(orig)))
    # Sin maps de subtítulos.
    assert "2:0?" not in argv
    assert "3:0?" not in argv
    # El output sigue siendo ``.mkv.tmp``.
    assert argv[-1].endswith(".mkv.tmp")


# ---------------------------------------------------------------------------
# promote_one (flujo completo)
# ---------------------------------------------------------------------------


def test_promote_one_happy_path(tmp_path, monkeypatch):
    season = tmp_path / "Season 01"
    orig, dubbed = _make_chapter(season, "S01E01")

    def fake_run(argv, **kwargs):
        # ffmpeg escribe el output y devuelve 0
        out = Path(argv[-1])
        out.write_bytes(b"\0muxed")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(promote_service.subprocess, "run", fake_run)
    monkeypatch.setattr(promote_service, "pipeline_active_for", lambda *_: None)

    svc = _build_service()
    result = svc.promote_one(str(orig))

    assert result["ok"] is True
    final = season / "S01E01.mkv"
    assert final.exists()
    assert final.read_bytes() == b"\0muxed"
    # Original y dubbed eliminados.
    assert not orig.exists()
    assert not dubbed.exists()
    # Carpeta ``doblajes/`` barrida (estaba vacía).
    assert not (season / "doblajes").exists()
    # Sidecars (srts) eliminados.
    assert not (season / "S01E01.es.srt").exists()
    assert not (season / "S01E01.en.srt").exists()


def test_promote_one_ffmpeg_failure_keeps_inputs(tmp_path, monkeypatch):
    """Cuando ffmpeg devuelve no-cero, el ``.tmp`` se barre y los inputs intactos."""
    season = tmp_path / "Season 01"
    orig, dubbed = _make_chapter(season, "S01E01")

    def fake_run(argv, **kwargs):
        # Simula ffmpeg escribiendo un ``.tmp`` parcial antes de petar.
        Path(argv[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(argv, 1, "", "fake stderr line\n")

    monkeypatch.setattr(promote_service.subprocess, "run", fake_run)
    monkeypatch.setattr(promote_service, "pipeline_active_for", lambda *_: None)

    svc = _build_service()
    with pytest.raises(HTTPException) as ei:
        svc.promote_one(str(orig))
    assert ei.value.status_code == 500
    assert ei.value.detail["code"] == "ffmpeg_failed"

    # Inputs intactos.
    assert orig.exists()
    assert dubbed.exists()
    assert (season / "S01E01.es.srt").exists()
    # El ``.mkv`` final no aterrizó.
    assert not (season / "S01E01.mkv").exists()
    # ``.tmp`` barrido.
    assert not (season / "S01E01.mkv.tmp").exists()


def test_promote_one_blocks_when_pipeline_active(tmp_path, monkeypatch):
    season = tmp_path / "Season 01"
    orig, _ = _make_chapter(season, "S01E01")

    monkeypatch.setattr(
        promote_service, "pipeline_active_for", lambda paths: "abc1234"
    )

    svc = _build_service()
    with pytest.raises(HTTPException) as ei:
        svc.promote_one(str(orig))
    assert ei.value.status_code == 409
    assert ei.value.detail["code"] == "pipeline_active"
    # Nada se escribió/borró.
    assert orig.exists()


# ---------------------------------------------------------------------------
# Endpoint /api/promote/season vía TestClient
# ---------------------------------------------------------------------------


def test_promote_season_aggregates_results(tmp_path, monkeypatch):
    season = tmp_path / "Season 01"
    o1, _ = _make_chapter(season, "S01E01")
    o2, _ = _make_chapter(season, "S01E02")
    # Tercer candidato sin doblado: el iterador de season lo salta
    # silenciosamente (no aparece como failed ni skipped).
    (season / "S01E03.mp4").write_bytes(b"\0")

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"ok")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(promote_service.subprocess, "run", fake_run)
    monkeypatch.setattr(promote_service, "pipeline_active_for", lambda *_: None)

    svc = _build_service()
    client = _build_client(svc)

    r = client.post("/api/promote/season", json={"season_path": str(season)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["promoted_count"] == 2
    assert body["failed_count"] == 0
    # E03 nunca llegó al pool de promoción (no estaba en doblajes/).
    assert all("S01E03" not in p for p in body["promoted"])


def test_promote_chapter_endpoint_happy_path(tmp_path, monkeypatch):
    """End-to-end del endpoint ``POST /api/promote/chapter`` con ffmpeg stub."""
    season = tmp_path / "Season 02"
    orig, _ = _make_chapter(season, "S02E01")

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"\0muxed")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(promote_service.subprocess, "run", fake_run)
    monkeypatch.setattr(promote_service, "pipeline_active_for", lambda *_: None)

    svc = _build_service()
    client = _build_client(svc)

    r = client.post("/api/promote/chapter", json={"video_path": str(orig)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["output_path"].endswith("S02E01.mkv")
    assert "spa" in body["muxed_streams"]["audio"]


def test_promote_season_missing_doblajes_returns_empty(tmp_path):
    """Si la season no tiene ``doblajes/``, devolvemos listas vacías + mensaje."""
    season = tmp_path / "Season Empty"
    season.mkdir()

    svc = _build_service()
    client = _build_client(svc)

    r = client.post("/api/promote/season", json={"season_path": str(season)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["promoted"] == []
    assert body["skipped"] == []
    assert body["failed"] == []
    assert "doblajes" in body["message"]
