"""Quality tests on BJJ golden set.

Requires Ollama running at localhost:11434 with qwen2.5:7b-instruct-q4_K_M
pulled. Skipped by default — run with `pytest -m requires_ollama`.

Métricas:
- Glosario: % de ítems donde TODOS los must_keep_en aparecen literales en ES.
- Peninsular: % SIN latinismos prohibidos Y CON ≥1 marcador peninsular.
- Char budget: % en rango [80%, 110%] del budget cuando fill_budget=True.

Pasa: las 3 ≥ 85%. Falla si <70%. Warning entre 70-85%.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from subtitle_generator.translator import OllamaTranslator


GOLDEN_PATH = Path(__file__).parent / "data" / "bjj_golden_set.jsonl"

LATAM_RE = re.compile(
    r"\b(ahorita|ustedes|agarrar|agarra|agarro|agarras|chévere|chido|párate|parate|nomás|porfa)\b",
    re.IGNORECASE,
)
PENINSULAR_RE = re.compile(
    r"\b(coge|coges|fíjate|vale|tú|tus|hostia|joder|venga|aquí|ahora|mira)\b",
    re.IGNORECASE,
)


def _load_golden() -> list[dict]:
    items = []
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


@pytest.mark.requires_ollama
@pytest.mark.quality
def test_translation_quality_on_bjj_golden_set():
    golden = _load_golden()
    assert len(golden) >= 20, "golden set debe tener al menos 20 ítems"

    # Apunta al puerto publicado por docker compose (host → contenedor).
    t = OllamaTranslator(base_url="http://localhost:11434")
    items = [{"text": g["en"], "duration_ms": g["duration_ms"]} for g in golden]
    translated = t.translate_for_dubbing(items, cps=17.0, fill_budget=True)

    assert len(translated) == len(golden), "count mismatch"

    glossary_pass = 0
    peninsular_pass = 0
    budget_pass = 0
    failures = []

    for g, es in zip(golden, translated):
        es_lower = es.lower()

        # (a) Glosario: todos los must_keep_en presentes (case-insensitive)
        gloss_ok = all(term.lower() in es_lower for term in g["must_keep_en"])
        if gloss_ok:
            glossary_pass += 1
        else:
            missing = [t for t in g["must_keep_en"] if t.lower() not in es_lower]
            failures.append(f"GLOSS  | {g['en'][:60]}... | missing {missing} | got: {es!r}")

        # (b) Peninsular: SIN latam Y CON ≥1 marker
        latam_match = LATAM_RE.search(es)
        peninsular_match = PENINSULAR_RE.search(es)
        pen_ok = (latam_match is None) and (peninsular_match is not None)
        if pen_ok:
            peninsular_pass += 1
        elif latam_match:
            failures.append(f"LATAM  | {es!r} | latinismo: {latam_match.group(0)}")

        # (c) Char budget: en [80%, 110%] del target
        target = int((g["duration_ms"] / 1000.0) * 17.0)
        ratio = len(es) / max(1, target)
        if 0.80 <= ratio <= 1.10:
            budget_pass += 1

    n = len(golden)
    glossary_pct = 100 * glossary_pass / n
    peninsular_pct = 100 * peninsular_pass / n
    budget_pct = 100 * budget_pass / n

    print(f"\n=== Quality metrics on {n} BJJ items ===")
    print(f"Glossary respected:    {glossary_pct:.0f}% ({glossary_pass}/{n})")
    print(f"Peninsular Spanish:    {peninsular_pct:.0f}% ({peninsular_pass}/{n})")
    print(f"Char budget compliance:{budget_pct:.0f}% ({budget_pass}/{n})")

    if failures:
        print("\nFailures:")
        for f in failures[:10]:
            print(f"  {f}")

    # Hard fail si cualquier métrica < 70%
    assert glossary_pct >= 70, f"glossary {glossary_pct:.0f}% < 70%"
    assert peninsular_pct >= 70, f"peninsular {peninsular_pct:.0f}% < 70%"
    assert budget_pct >= 70, f"budget {budget_pct:.0f}% < 70%"
