"""BJJ English terms — DEPRECADO en el path activo (ronda 8).

Con ``DubbingConfig.xtts_code_switching = False`` (default desde ronda
8) este set **no se consulta**: todo el texto se pasa como single-span
``language="es"`` al synthesizer.

Historia: hasta ronda 7 aquí vivían términos 1-palabra que queríamos
pronunciar con fonología inglesa mediante code-switching. Pero XTTS-v2
**no soporta code-switching real** — alternar ``language`` entre chunks
dentro del mismo discurso deja residuos del sampler anterior y colapsa
a chino/japonés (embeddings vecinos en el espacio de idiomas soportados).
Ninguna combinación de temperature/top_p/repetition_penalty en rondas
1-7 eliminó la alucinación al 100%. Único fix garantizado: no hacer
code-switching.

El set se mantiene como **frozenset vacío** para:

1. No romper imports existentes (synthesizer_xttsv2.py, tests).
2. Si alguien reactiva ``xtts_code_switching=True`` por error, el
   split_by_language devuelve ``[("es", text)]`` (igual que el path
   desactivado) — doble red de seguridad contra regresión.

Si en el futuro aparece XTTS-v3 con code-switching real, este archivo
puede volver a poblarse; hasta entonces, **no añadir términos aquí**.
Para términos BJJ, usar ``bjj_casting.py`` (castellaniza compuestos
2+ palabras y palabras sueltas problemáticas).
"""

from __future__ import annotations

# Explícitamente vacío. Ver docstring para contexto.
DEFAULT_BJJ_EN_TERMS: frozenset[str] = frozenset()
