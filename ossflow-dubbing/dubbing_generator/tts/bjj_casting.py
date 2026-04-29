"""Castellanización de términos BJJ antes de enviar al sintetizador.

XTTS-v2 aluciana (chino, japonés, mezclas fonéticas) cuando un mismo
prompt contiene un span ES seguido de un término EN — la transición
entre sub-inferencias dispara un reinicio prosódico donde el sampler
puede perderse. Cuanto más corto es el span EN (1 palabra) y más
frecuente aparece en el SRT, más probabilidad de alucinar.

Estrategia:

* Compuestos BJJ de 2+ palabras → castellanización obligatoria.
* Palabras sueltas *ambiguas* (grip/grips, frame/frames, head/hips…) →
  también castellanizadas, porque su aparición repetida dentro de un
  bloque ES largo es lo que desencadena "guardia [chino] seated guard"
  en caps largos (BJJ-density alta).
* Términos específicamente BJJ con acento EN bien marcado que XTTS
  pronuncia limpio (underhook, kimura, armbar…) → quedan en
  ``bjj_en_terms.py`` para code-switching puro.

El mapa es ordenado por longitud descendente de clave para que
"seated guard" se reemplace antes que "guard", y "grips" antes
que "grip".
"""

from __future__ import annotations

import re

# Compuestos BJJ + palabras sueltas problemáticas. Claves en minúscula; el
# matching es case-insensitive en word boundaries.
_COMPOUND_MAP: dict[str, str] = {
    # ------------------------------------------------------------------
    # Términos clave — pronunciación en TTS local (Piper) sin castellanizar
    # ------------------------------------------------------------------
    "jiu-jitsu": "yiu yitsu",
    "jiujitsu": "yiuyitsu",
    "jiu jitsu": "yiu yitsu",
    "bjj": "be ye ye",
    # ------------------------------------------------------------------
    # Compuestos (2+ palabras) — principal fuente de alucinación confirmada
    # ------------------------------------------------------------------
    # Guardias
    "seated guard": "guardia sentada",
    "standing guard": "guardia de pie",
    "closed guard": "guardia cerrada",
    "open guard": "guardia abierta",
    "half guard": "media guardia",
    "butterfly guard": "guardia mariposa",
    "spider guard": "guardia araña",
    "de la riva": "de la riva",          # sin cambio pero lo dejamos ES
    "reverse de la riva": "de la riva inversa",
    "x guard": "guardia x",
    "x-guard": "guardia x",
    "single leg x": "x de una pierna",
    "z guard": "guardia z",
    # Posiciones
    "side control": "control lateral",
    "back control": "control de espalda",
    "back take": "toma de espalda",
    "full mount": "montada completa",
    "north south": "norte sur",
    "knee on belly": "rodilla en el vientre",
    "knee cut": "corte de rodilla",
    "head position": "posición de cabeza",
    "head control": "control de cabeza",
    "hip position": "posición de cadera",
    # Grips y controles compuestos
    "two on one": "dos contra uno",
    "two-on-one": "dos contra uno",
    "wrist control": "control de muñeca",
    "collar tie": "agarre de collar",
    "russian tie": "agarre ruso",
    "cross grip": "agarre cruzado",
    "pistol grip": "agarre pistola",
    "c grip": "agarre en c",
    "c-grip": "agarre en c",
    # Frames y conceptos
    "inside frame": "marco interior",
    "outside frame": "marco exterior",
    "elbow frame": "marco de codo",
    # Sumisiones compuestas
    "heel hook": "heel hook",             # término específico, dejamos EN
    "rear naked choke": "mata león",
    "bow and arrow": "arco y flecha",
    "straight ankle lock": "llave de tobillo recta",
    # Acciones compuestas
    "stand up": "levantarse",
    "hip escape": "escape de cadera",
    "leg drag": "arrastre de pierna",
    # Cradles y ganchos de pierna — pronunciación EN mala en TTS local
    "leg hooks": "ganchos de pierna",
    "leg hook": "gancho de pierna",
    "cradles": "abrazos de pierna",
    "cradle": "abrazo de pierna",
    # NOTA: "single leg" y "double leg" NO se castellanizan — son
    # términos técnicos universales en BJJ/wrestling. "Una pierna"
    # suena raro al oyente BJJ-literate. XTTS los pronuncia bien
    # porque forman un compuesto fijo que el usuario espera en EN.
    # Si volvemos a ver alucinación en ellos, añadir 'leg' al set
    # EN terms (está implícito como parte del compuesto).
    # ------------------------------------------------------------------
    # Palabras sueltas genéricas (no BJJ-exclusivas) que XTTS pronuncia
    # mal o dispara alucinación cuando aparecen 5+ veces en un cap ES.
    # El plural inglés -s en contexto ES es especialmente tóxico: el
    # sampler mezcla el final /s/ con el siguiente morfema ES y colapsa
    # a ruido en otros idiomas.
    # ------------------------------------------------------------------
    "grips": "agarres",
    "grip": "agarre",
    "frames": "marcos",
    "frame": "marco",
    "hooks": "ganchos",
    # "hook" suelto: con TTS local (Piper) suena mal; con ElevenLabs sonaba
    # ok aislado. Castellanizamos siempre por seguridad — si el motor cloud
    # vuelve, el oyente igual entiende "gancho".
    "hook": "gancho",
    "levers": "palancas",
    "lever": "palanca",
    "posts": "postes",
    "post": "poste",
    "grippings": "agarres",
    # ------------------------------------------------------------------
    # Pasajes específicos
    # ------------------------------------------------------------------
    "tripod passing": "pase de trípode",
    "tripod pass": "pase de trípode",
    "tripod": "trípode",
    # ------------------------------------------------------------------
    # Más técnicas BJJ comunes que TTS local pronuncia mal
    # ------------------------------------------------------------------
    # Front headlock / chin strap (Craig Jones, Danaher style)
    "chin strap": "agarre de mentón",
    "chin straps": "agarres de mentón",
    "front headlock": "control frontal de cabeza",
    "guillotine choke": "guillotina",
    "guillotine": "guillotina",
    "darce": "darce",
    "d'arce": "darce",
    "anaconda": "anaconda",
    "ezekiel": "ezequiel",
    "north south choke": "estrangulación norte sur",
    "loop choke": "estrangulación loop",
    "bow and arrow choke": "estrangulación arco y flecha",
    "rear naked": "mata león",
    "lapel choke": "estrangulación con la solapa",
    "cross collar choke": "estrangulación de collar cruzado",
    # Leg lock system (Danaher)
    "leg locks": "llaves de pierna",
    "leg lock": "llave de pierna",
    "ashi garami": "ashi garami",
    "outside ashi": "ashi exterior",
    "inside ashi": "ashi interior",
    "saddle": "saddle",
    "honey hole": "honey hole",
    "411": "cuatro once",
    "50/50": "cincuenta cincuenta",
    "5050": "cincuenta cincuenta",
    "double trouble": "double trouble",
    "knee bar": "luxación de rodilla",
    "kneebar": "luxación de rodilla",
    "toe hold": "candado de pie",
    "toehold": "candado de pie",
    "calf slicer": "compresor de gemelo",
    "calf cutter": "compresor de gemelo",
    "estima lock": "candado estima",
    # Pasajes y entradas
    "leg drag pass": "paso de arrastre de pierna",
    "knee slice": "corte de rodilla",
    "smash pass": "paso a aplastamiento",
    "over under pass": "paso over under",
    "double under": "doble under",
    "long step": "paso largo",
    "back step": "paso atrás",
    "switch step": "paso de cambio",
    "leg pummel": "intercambio de piernas",
    # Sweeps y entradas a guardia
    "scissor sweep": "barrida de tijera",
    "flower sweep": "barrida de flor",
    "pendulum sweep": "barrida péndulo",
    "hip bump sweep": "barrida de cadera",
    "tripod sweep": "barrida tripod",
    "k guard": "guardia k",
    "k-guard": "guardia k",
    "lapel guard": "guardia de solapa",
    "worm guard": "guardia gusano",
    "lasso guard": "guardia lasso",
    "deep half": "media guardia profunda",
    "deep half guard": "media guardia profunda",
    "single leg": "una pierna",
    "single legs": "una pierna",
    "double leg": "doble pierna",
    "double legs": "doble pierna",
    # Conceptos
    "underhook": "underhook",
    "underhooks": "underhooks",
    "overhook": "overhook",
    "overhooks": "overhooks",
    "whizzer": "whizzer",
    "wedge": "cuña",
    "wedges": "cuñas",
    "shoulder pressure": "presión de hombro",
    "cross face": "cara cruzada",
    "crossface": "cara cruzada",
    "elbow control": "control de codo",
    "head and arm": "cabeza y brazo",
    # Acciones generales que aparecen sueltas
    "passes": "pasajes",
    "pass": "pasaje",
    "guards": "guardias",
    "submissions": "sumisiones",
    "submission": "sumisión",
    "transitions": "transiciones",
    "transition": "transición",
    "takedowns": "derribos",
    "takedown": "derribo",
    "sweeps": "barridas",
    "sweep": "barrida",
    "escapes": "escapes",
    "drills": "drills",
    "drilling": "drilling",
    "tap": "rendir",
    "tapping": "rindiendo",
    "tapped": "rindió",
    # ------------------------------------------------------------------
    # Nombres propios EN problemáticos en TTS local. Reemplazo fonético
    # ES (no traducción) para mantener identidad de la persona.
    # ------------------------------------------------------------------
    # Ortografía ES aproximada que Piper pronuncie como el original EN.
    "danaher": "Dánajer",
    "john danaher": "Yon Dánajer",
    "gordon ryan": "Górdon Ráian",
    "gordon": "Górdon",
    "ryan": "Ráian",
    "nicky ryan": "Niki Ráian",
    "nicky": "Niki",
    "craig jones": "Creig Yons",
    "craig": "Creig",
    "jones": "Yons",
    "garry tonon": "Gári Tónon",
    "tonon": "Tónon",
    "tom deblass": "Tom Diblás",
    "deblass": "Diblás",
    "lachlan giles": "Láklan Yails",
    "lachlan": "Láklan",
    "giles": "Yails",
    "josef chen": "Yósef Chen",
    "jozef chen": "Yósef Chen",
    "jozef": "Yósef",
    "josef": "Yósef",
    "chen": "Chen",
    "kade ruotolo": "Keid Ruótolo",
    "tye ruotolo": "Tai Ruótolo",
    "kade": "Keid",
    "tye": "Tai",
    "ruotolo": "Ruótolo",
    "mikey musumeci": "Maiki Musumechi",
    "mikey": "Maiki",
    "musumeci": "Musumechi",
    "marcelo garcia": "Marcelo García",
    "rickson gracie": "Rickson Greisi",
    "roger gracie": "Roger Greisi",
    "gracie": "Greisi",
    "renzo": "Renzo",
    "buchecha": "Bucheca",
    "leandro lo": "Leandro Lo",
    "andre galvao": "Andre Galváo",
    "galvao": "Galváo",
    "felipe pena": "Felipe Pena",
    "rafa mendes": "Rafa Mendes",
    "rafael mendes": "Rafael Mendes",
    "guilherme mendes": "Guilerme Mendes",
    "lucas lepri": "Lucas Lepri",
    "keenan cornelius": "Kínan Cornelius",
    "keenan": "Kínan",
    "cornelius": "Cornelius",
    "tyler": "Táiler",
    "tainan": "Tainan",
    "diogo reis": "Diogo Reis",
    "matheus diniz": "Mateus Diniz",
    "ffion davies": "Fion Deivis",
    "tackett": "Táket",
    "nicholas meregali": "Nicolás Meregali",
    "meregali": "Meregali",
    "haisam rida": "Háisam Rida",
}


def _build_pattern() -> re.Pattern:
    # Orden descendente por longitud para que matches largos ganen.
    keys = sorted(_COMPOUND_MAP.keys(), key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b",
        re.IGNORECASE,
    )


_PATTERN = _build_pattern()


def castellanize(text: str) -> str:
    """Reemplaza compuestos BJJ EN por su equivalente castellano.

    Preserva la capitalización aproximada: si el match original empieza
    en mayúscula, la sustitución también (solo la primera letra de la
    primera palabra). El resto es minúscula — no intentamos respetar
    MAYÚSCULA COMPLETA porque en SRT normal no aparece.
    """
    if not text:
        return text

    def _sub(m: re.Match) -> str:
        original = m.group(0)
        replacement = _COMPOUND_MAP[original.lower()]
        if original[0].isupper() and replacement:
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return _PATTERN.sub(_sub, text)
