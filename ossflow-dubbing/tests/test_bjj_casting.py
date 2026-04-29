"""Tests for BJJ compound castellanization."""

from dubbing_generator.tts.bjj_casting import castellanize


def test_empty_string():
    assert castellanize("") == ""


def test_no_match_passthrough():
    assert castellanize("esto no tiene jerga") == "esto no tiene jerga"


def test_seated_guard_replaced():
    out = castellanize("está en seated guard")
    assert out == "está en guardia sentada"


def test_two_on_one_replaced():
    out = castellanize("conseguir un two on one")
    assert out == "conseguir un dos contra uno"


def test_case_insensitive_match():
    out = castellanize("la Seated Guard es clave")
    # Capitalización preservada para la 1ª letra de la sustitución.
    assert out == "la Guardia sentada es clave"


def test_multiple_replacements_in_one_phrase():
    out = castellanize("desde seated guard paso a side control")
    assert out == "desde guardia sentada paso a control lateral"


def test_longest_match_wins():
    # "seated guard" debe ganar sobre "guard" sola.
    out = castellanize("en seated guard no en guard abierta")
    assert "guardia sentada" in out
    # "guard" suelta no se toca (queda EN — 1 palabra)
    assert "guard abierta" in out


def test_word_boundary_not_substring():
    # "seated guardian" no debe matchear "seated guard".
    out = castellanize("no seated guardian")
    assert out == "no seated guardian"


def test_single_word_passthrough():
    # "guard" sola NO está en el mapa → queda como está (XTTS la
    # pronunciará con acento EN vía code-switching).
    assert castellanize("la guard abierta") == "la guard abierta"


def test_grips_plural_replaced():
    # Plural EN en contexto ES dispara alucinación; castellanizar.
    assert castellanize("agarramos los grips") == "agarramos los agarres"


def test_grip_singular_replaced():
    assert castellanize("un grip fuerte") == "un agarre fuerte"


def test_frames_replaced():
    assert castellanize("uso frames para defender") == "uso marcos para defender"


def test_hooks_replaced_but_hook_passthrough():
    # Plural "hooks" problemático → castellaniza.
    assert castellanize("pon los hooks") == "pon los ganchos"
    # Singular "hook" queda en EN (estable aislado) → bjj_en_terms.
    assert castellanize("un hook profundo") == "un hook profundo"


def test_head_position_compound_replaced():
    assert (
        castellanize("controla la head position")
        == "controla la posición de cabeza"
    )


def test_cross_grip_compound_wins_over_grip():
    # "cross grip" debe ganar sobre "grip" suelto (longest-match).
    out = castellanize("busca un cross grip alto")
    assert out == "busca un agarre cruzado alto"
