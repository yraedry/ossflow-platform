"""Split a mixed ES/EN sentence into per-language spans."""

from __future__ import annotations

import re


def split_by_language(
    text: str,
    en_terms: frozenset[str] | set[str],
) -> list[tuple[str, str]]:
    """Return [(lang, span), ...] preserving order and original whitespace.

    - ``lang`` is ``"es"`` (default) or ``"en"`` (matched term).
    - Matching is case-insensitive on word boundaries; original casing is
      preserved in the returned span.
    - Longer terms take precedence (``"two on one"`` beats ``"one"``).
    - Empty input returns an empty list.
    """
    if not text:
        return []
    # Filtrar términos vacíos: una cadena "" en el set produciría un
    # pattern `\b(?:|...)\b` que matchea cadena vacía en cada posición
    # → bucle infinito en finditer. Defensivo: un usuario mal
    # configurado no debería colgar el servicio.
    clean_terms = [t for t in en_terms if t]
    if not clean_terms:
        return [("es", text)]

    sorted_terms = sorted(clean_terms, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in sorted_terms) + r")\b",
        re.IGNORECASE,
    )

    spans: list[tuple[str, str]] = []
    cursor = 0
    for match in pattern.finditer(text):
        start, end = match.start(), match.end()
        if start > cursor:
            spans.append(("es", text[cursor:start]))
        spans.append(("en", text[start:end]))
        cursor = end
    if cursor < len(text):
        spans.append(("es", text[cursor:]))
    return spans
