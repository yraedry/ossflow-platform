"""Configuration dataclasses and default constants for the subtitle generator."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ROOT_DIR = r"Z:\instruccionales\Arm Drags - John Danaher\Season 01"

EXTENSIONES = (".mp4", ".mkv", ".avi", ".mov")

DEFAULT_INITIAL_PROMPT_TEMPLATE = (
    "The following is a technical Brazilian Jiu-Jitsu instructional"
    " by {instructor} on {topic}."
)

DEFAULT_INITIAL_PROMPT_GENERIC = (
    "A Brazilian Jiu-Jitsu coach explains techniques step by step."
    " He uses English terms like guard, half-guard, mount, side control,"
    " armbar, kimura, triangle, sweep, underhook, overhook, tripod."
)

# Keep the old name as an alias pointing to the generic prompt for backward compat
DEFAULT_INITIAL_PROMPT = DEFAULT_INITIAL_PROMPT_GENERIC

DEFAULT_HOTWORDS = "\n".join([
    # ------------------------------------------------------------------
    # Guard positions
    # ------------------------------------------------------------------
    "guard", "half guard", "half-guard", "closed guard", "full guard",
    "open guard", "butterfly guard", "butterfly hooks",
    "de la Riva", "de la Riva guard", "reverse de la Riva",
    "reverse de la Riva guard", "de la X", "de la X guard",
    "x-guard", "single leg x", "single leg x-guard", "single leg x worm guard",
    "50/50", "50/50 guard", "spider guard", "lasso guard", "lasso spider",
    "worm guard", "worm hole", "worm wrestling",
    "rubber guard", "z-guard", "knee shield", "knee shield half guard",
    "quarter guard", "dog fight", "dogfight", "lockdown",
    "squid guard", "gubber guard", "coyote guard", "lucas leite guard",
    "k-guard", "octopus guard", "williams guard", "reverse x guard",
    "honey hole", "saddle", "inside sankaku", "outside sankaku",
    "ashi garami", "cross ashi garami", "cross ashi",
    "411", "four eleven", "leg entanglement", "leg lock position",
    "truck", "the truck", "lapel guard", "lapel lasso", "demi guard",
    "seated guard", "supine guard", "standing guard",
    # ------------------------------------------------------------------
    # Top positions
    # ------------------------------------------------------------------
    "mount", "full mount", "high mount", "low mount", "s-mount",
    "technical mount", "twisted mount",
    "side control", "side mount", "cross side", "cross body",
    "kesa gatame", "scarf hold", "reverse kesa gatame",
    "knee on belly", "knee-on-belly", "knee ride",
    "north-south", "north south", "headquarters", "HQ",
    "back mount", "back control", "seatbelt", "body triangle",
    "crucifix", "crucifix position",
    # ------------------------------------------------------------------
    # Bottom / defensive positions
    # ------------------------------------------------------------------
    "turtle", "turtle position", "turtling",
    # ------------------------------------------------------------------
    # Passes
    # ------------------------------------------------------------------
    "pass", "passing", "guard pass", "pass the guard",
    "toreando", "toreando pass", "torreando",
    "tripod", "tripod pass", "tripod passing",
    "knee cut", "knee cut pass", "knee slice",
    "leg drag", "smash pass", "over-under pass", "over-under",
    "stack pass", "long step", "longstep", "backstep", "back step",
    "body lock pass", "body-lock pass", "double under pass",
    "double unders", "x-pass", "folding pass",
    "cartwheel pass", "gravedigger pass", "throwby pass",
    "pressure pass", "platinum worm passing",
    # ------------------------------------------------------------------
    # Submissions — armlocks
    # ------------------------------------------------------------------
    "armbar", "arm bar", "juji gatame", "flying armbar",
    "kimura", "ude garami", "reverse kimura",
    "americana", "keylock",
    "omoplata", "baratoplata", "monoplata", "manoplata",
    "lapelaplata", "gogoplata", "mir lock",
    "wrist lock", "wristlock", "mão de vaca",
    # ------------------------------------------------------------------
    # Submissions — chokes
    # ------------------------------------------------------------------
    "rear naked choke", "RNC", "mata leão", "mata leao",
    "triangle", "triangle choke", "sankaku", "sankaku jime",
    "inverted triangle", "reverse triangle", "side triangle",
    "arm triangle", "arm triangle choke", "kata gatame",
    "guillotine", "high elbow guillotine", "ten finger guillotine",
    "flying guillotine", "marcelotine",
    "darce", "d'arce", "d'arce choke",
    "anaconda", "anaconda choke",
    "ezekiel", "ezequiel", "ezekiel choke",
    "bow and arrow", "bow and arrow choke",
    "cross collar choke", "cross-collar choke",
    "loop choke", "Von Flue choke", "Von Flue",
    "north-south choke", "clock choke", "baseball bat choke",
    "paper cutter choke", "peruvian necktie", "japanese necktie",
    "neck crank", "can opener", "twister", "chin strap", "chinstrap",
    "gi choke",
    # ------------------------------------------------------------------
    # Submissions — leg locks
    # ------------------------------------------------------------------
    "heel hook", "inside heel hook", "outside heel hook",
    "toe hold", "kneebar", "knee bar",
    "ankle lock", "straight ankle lock",
    "calf slicer", "calf crusher", "estima lock",
    "banana split", "electric chair",
    # ------------------------------------------------------------------
    # Sweeps / reversals / movements
    # ------------------------------------------------------------------
    "sweep", "sweeping", "reversal",
    "berimbolo", "kiss of the dragon", "granby", "granby roll",
    "tornado sweep", "scissor sweep", "hook sweep", "butterfly sweep",
    "flower sweep", "pendulum sweep", "lumberjack sweep",
    "hip bump", "hip bump sweep", "elevator sweep",
    "technical stand-up", "technical stand up",
    "bridge", "bridging", "bridge and roll", "upa", "upa escape",
    "shrimp", "shrimping", "hip escape",
    "imanari", "imanari roll", "rolling back take",
    # ------------------------------------------------------------------
    # Takedowns / stand-up
    # ------------------------------------------------------------------
    "takedown", "take down",
    "single leg", "single leg takedown", "high crotch",
    "double leg", "double leg takedown",
    "arm drag", "2-on-1", "2 on 1", "russian tie",
    "fireman's carry", "fireman carry",
    "ouchi gari", "kouchi gari", "osoto gari", "kosoto gari",
    "o soto gari", "o uchi gari", "ko uchi gari", "ko soto gari",
    "harai goshi", "uchi mata", "uchimata",
    "seoi nage", "ippon seoi nage", "drop seoi nage",
    "tai otoshi", "tomoe nage", "sumi gaeshi", "sumi-gaeshi",
    "tani otoshi", "sasae", "sasae tsurikomi ashi",
    "kouchi makikomi", "makikomi",
    "sprawl", "sprawling", "whizzer",
    # ------------------------------------------------------------------
    # Grips / hand fighting
    # ------------------------------------------------------------------
    "grip", "grips", "grip fighting", "hand fighting",
    "gable grip", "s-grip", "s grip", "ten finger grip",
    "pistol grip", "pocket grip", "monkey grip", "spider grip",
    "seat belt grip", "seatbelt grip",
    "collar sleeve grip", "same side collar grip",
    "cross collar grip", "cross grip", "same side grip",
    "collar tie", "2-on-1 grip", "wrist control",
    "underhook", "overhook", "under hook", "over hook",
    "pummel", "pummeling", "pommeling",
    "cross face", "crossface", "frame", "framing", "frames",
    # ------------------------------------------------------------------
    # Body parts / anatomy-in-BJJ
    # ------------------------------------------------------------------
    "post", "posting", "base", "hook", "hooks",
    "lapel", "sleeve", "collar", "belt", "pants", "cuff",
    "hip", "hips", "shoulder", "elbow", "knee", "shin",
    "inside position", "inside space", "outside position",
    "knee line", "hip line", "center line", "centerline",
    # ------------------------------------------------------------------
    # Concepts / coaching speak
    # ------------------------------------------------------------------
    "posture", "posturing up", "broken posture",
    "connection", "disconnection", "alignment", "misalignment",
    "pressure", "heavy pressure", "weight distribution",
    "leverage", "timing", "isolation", "control",
    "angle", "angles", "off-balance", "off-balancing", "kuzushi", "kazushi",
    "entry", "entries", "setup", "set-up", "transition", "transitions",
    "chain", "chaining", "combo", "combination",
    "retention", "recovery", "guard retention", "guard recovery",
    "escape", "escaping", "submission", "submissions", "finish", "finishing",
    "defense", "offense", "counter", "counters", "countering",
    "top", "bottom", "top pressure", "top position", "bottom position",
    "stack", "stacking", "stuff", "stuffing",
    "break", "breaking grips", "grip break", "posture break",
    "reset", "resetting",
    # ------------------------------------------------------------------
    # Training / terms
    # ------------------------------------------------------------------
    "drill", "drilling", "drill it", "drilling partner",
    "roll", "rolling", "sparring", "positional sparring",
    "positional training", "specific sparring", "specific training",
    "live rounds", "live training", "flow roll", "flow rolling",
    "randori", "uchikomi", "ukemi", "break fall", "breakfall",
    "tap", "tap out", "tapping", "tapping out", "submit",
    "roll over", "bail", "give up the back",
    "gi", "no-gi", "nogi", "no gi", "kimono", "belt",
    "faixa", "faixa preta", "faixa roxa", "faixa azul", "faixa marrom",
    "white belt", "blue belt", "purple belt", "brown belt", "black belt",
    "stripe", "stripes", "tips",
    # ------------------------------------------------------------------
    # Japanese technique terminology (judo-derived)
    # ------------------------------------------------------------------
    "waza", "newaza", "tachi-waza", "ne-waza", "tachi waza",
    "jime", "gatame", "garami", "nage", "hishigi",
    "ude hishigi", "ude hishigi juji gatame",
    "ude garami", "kata ha jime", "hadaka jime",
    "juji gatame", "sankaku jime", "kata gatame", "kesa gatame",
    "yoko sankaku", "ushiro sankaku",
    # ------------------------------------------------------------------
    # Portuguese / Brazilian terms
    # ------------------------------------------------------------------
    "oss", "osu", "porrada", "porra", "combate", "pegada",
    "jogo", "tatame", "vai", "vai pegar", "parou", "luta", "lute",
    "rola", "vale tudo", "jiu-jitsu", "jiu jitsu", "jiujitsu",
    "BJJ", "gentle art", "arte suave",
    "professor", "mestre", "sensei", "coach",
    "passador", "guardeiro", "guard player", "guard passer", "guard puller",
    "guard pull", "pulling guard",
    # ------------------------------------------------------------------
    # Referees / competition
    # ------------------------------------------------------------------
    "referee", "ref", "advantage", "advantages", "points",
    "penalty", "penalties", "disqualification", "DQ",
    "submission only", "EBI overtime", "overtime",
    "double gold", "absolute", "open weight",
    "round timer", "shark tank",
    # ------------------------------------------------------------------
    # People — instructors and athletes commonly referenced
    # ------------------------------------------------------------------
    "Danaher", "John Danaher", "Gordon Ryan", "Craig Jones",
    "Marcelo Garcia", "Lachlan Giles", "Mikey Musumeci",
    "Jozef Chen", "Nicky Rod", "Nicky Ryan", "Kade Ruotolo",
    "Tye Ruotolo", "Roger Gracie", "Buchecha", "Marcus Almeida",
    "Rafael Lovato", "Lovato Jr",
    "Keenan Cornelius", "Xande Ribeiro", "Saulo Ribeiro",
    "Andre Galvao", "Leandro Lo", "Rodolfo Vieira",
    "Caio Terra", "Rafa Mendes", "Gui Mendes", "Mendes brothers",
    "Cobrinha", "Rubens Charles", "Bernardo Faria",
    "Helio Gracie", "Carlos Gracie", "Rickson Gracie",
    "Royce Gracie", "Royler Gracie", "Renzo Gracie",
    "Kron Gracie", "Roger Gracie",
    "Tom DeBlass", "Garry Tonon", "Eddie Bravo", "10th Planet",
    "Keith Krikorian", "Nick Rodriguez", "Giancarlo Bodoni",
    "Dante Leon", "Lucas Lepri", "Lucas Leite",
    "Marcelotine", "Marcelo",
    # ------------------------------------------------------------------
    # Competitions / federations / brands
    # ------------------------------------------------------------------
    "ADCC", "IBJJF", "EBI", "CJJ", "Combat Jiu-Jitsu",
    "ADCC Worlds", "ADCC Trials", "IBJJF Worlds", "IBJJF Pans",
    "Worlds", "Mundials", "Mundial", "Pans", "Europeans", "No-Gi Worlds",
    "Who's Number One", "WNO", "UFC Fight Pass Invitational", "FPI",
    "Polaris", "Grappling Industries", "Submission Only",
    "BJJ Fanatics", "BJJFanatics",
])


def generate_prompt(
    instructor: str | None = None,
    topic: str | None = None,
    template: str | None = None,
) -> str:
    """Build a WhisperX initial prompt from instructor/topic metadata.

    If neither *instructor* nor *topic* are provided, returns a generic BJJ
    prompt.  A custom *template* may be supplied (must contain ``{instructor}``
    and/or ``{topic}`` placeholders).
    """
    if not instructor and not topic:
        return DEFAULT_INITIAL_PROMPT_GENERIC

    tpl = template or DEFAULT_INITIAL_PROMPT_TEMPLATE
    return tpl.format(
        instructor=instructor or "the instructor",
        topic=topic or "Brazilian Jiu-Jitsu techniques",
    )


@dataclass
class TranscriptionConfig:
    """Parameters controlling WhisperX transcription and alignment."""

    model_name: str = "large-v3"
    compute_type: str = "float16"
    device: str = "cuda"
    language: str = "en"
    batch_size: int = 2
    beam_size: int = 5
    initial_prompt: str = DEFAULT_INITIAL_PROMPT

    # VAD parameters — slightly more sensitive to catch speech in noisy audio
    # (audio is denoised before VAD, so lower thresholds are safe)
    vad_onset: float = 0.300
    vad_offset: float = 0.200

    # Gap-fill: re-transcribe gaps larger than this (seconds) with boosted audio
    gap_fill_min_gap: float = 3.0
    gap_fill_audio_boost_db: float = 10.0
    gap_fill_vad_onset: float = 0.200
    gap_fill_vad_offset: float = 0.150

    # Transcription quality
    condition_on_previous_text: bool = False
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.7

    # BJJ-specific hotwords for WhisperX — improves recognition of domain terms
    # WhisperX expects a single string (newline-separated)
    hotwords: str | None = None

    # Prompt template for dynamic generation
    initial_prompt_template: str = DEFAULT_INITIAL_PROMPT_TEMPLATE

    # OpenAI post-process: cleans WhisperX artifacts (syllable duplication,
    # broken mid-clause boundaries) preserving block count and timestamps.
    # Runs after writer.write_srt; on failure it leaves the original SRT.
    postprocess_openai: bool = False
    postprocess_model: str = "gpt-4o-mini"
    postprocess_api_key: str | None = None


@dataclass
class SubtitleConfig:
    """Parameters controlling subtitle formatting and validation."""

    max_chars_per_line: int = 42
    max_lines: int = 2
    min_duration: float = 0.5
    max_duration: float = 7.0
    gap_fill_threshold: float = 0.100   # Fill gaps smaller than 100ms
    gap_warn_threshold: float = 5.0     # Warn about gaps larger than 5s

    # Hallucination filter thresholds
    similarity_threshold: float = 0.80
    similarity_lookback: int = 15
    repeated_ngram_size: int = 3
    repeated_ngram_max: int = 5       # BJJ instructors repeat phrases a lot ("from here", "what we need to")
    low_confidence_word_threshold: float = 0.25  # only flag very low confidence
    low_confidence_segment_ratio: float = 0.90   # only drop if almost all words are very low-conf
    max_chars_per_second: float = 60.0  # very relaxed — alignment jitter can produce high CPS on short segs

    # Silence filter threshold (RMS in dBFS below which a segment is silence)
    silence_threshold_db: float = -40.0  # title cards / silence typically above -40 dBFS

    # Synthetic score filter: ratio of words with exact 0.5 score to trigger stricter filtering
    synthetic_score_ratio: float = 0.80  # only flag near-fully-synthetic segments
    synthetic_score_strict_ratio: float = 0.60  # relaxed strict pass
