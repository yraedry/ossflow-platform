"""SRT translation with pluggable providers (Ollama local, OpenAI cloud).

Preserves timestamps and subtitle structure. Writes ``{base}_ES.srt`` next
to the source file.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Protocol

import httpx

from .srt_io import parse_srt, serialize_srt

log = logging.getLogger("subtitler")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

_TIMEOUT = 120.0

_RATE_LIMIT_MAX_RETRIES = 5
_RATE_LIMIT_MAX_WAIT = 60.0
_TRY_AGAIN_RE = re.compile(r"try again in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def _retry_delay_from_response(resp: httpx.Response, attempt: int) -> float:
    """Pick a wait time for 429/5xx based on headers, body, or exponential backoff."""
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header:
        try:
            return min(float(header), _RATE_LIMIT_MAX_WAIT)
        except ValueError:
            pass
    match = _TRY_AGAIN_RE.search(resp.text or "")
    if match:
        try:
            return min(float(match.group(1)) + 0.5, _RATE_LIMIT_MAX_WAIT)
        except ValueError:
            pass
    # Exponential backoff with jitter: 1, 2, 4, 8, 16 s (capped).
    base = min(2 ** attempt, _RATE_LIMIT_MAX_WAIT)
    return base + random.uniform(0, 0.5)


def _post_with_retry(
    url: str,
    *,
    headers: dict,
    json_body: dict | None = None,
    data: dict | None = None,
    provider_label: str,
) -> httpx.Response:
    """POST that retries 429 and 5xx with backoff. Other errors bubble up via caller."""
    last_resp: httpx.Response | None = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES):
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(url, headers=headers, json=json_body, data=data)
        if resp.status_code < 400:
            return resp
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            last_resp = resp
            if attempt == _RATE_LIMIT_MAX_RETRIES - 1:
                break
            wait = _retry_delay_from_response(resp, attempt)
            log.warning(
                "%s %d (attempt %d/%d) — sleeping %.1fs before retry",
                provider_label, resp.status_code, attempt + 1,
                _RATE_LIMIT_MAX_RETRIES, wait,
            )
            time.sleep(wait)
            continue
        return resp
    assert last_resp is not None  # loop above always enters at least once
    return last_resp


# ---------------------------------------------------------------------------
# Language code helpers
# ---------------------------------------------------------------------------

_LANG_NAMES = {
    "EN": "English",
    "ES": "Spanish",
    "PT": "Portuguese",
    "FR": "French",
    "IT": "Italian",
    "DE": "German",
}


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code.upper(), code)


# ---------------------------------------------------------------------------
# Protocol + base
# ---------------------------------------------------------------------------

class Translator(Protocol):
    """Minimum contract a translation provider must fulfil."""

    def translate_texts(self, texts: list[str]) -> list[str]: ...

    def translate_srt(self, src_path: Path, dst_path: Path | None = None) -> Path: ...


class _BaseTranslator:
    """Shared SRT read/write logic. Subclasses implement ``translate_texts``."""

    def __init__(self, source_lang: str = "EN", target_lang: str = "ES") -> None:
        self.source_lang = source_lang.upper()
        self.target_lang = target_lang.upper()

    def translate_texts(self, texts: list[str]) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def translate_srt(
        self,
        src_path: Path,
        dst_path: Path | None = None,
    ) -> Path:
        """Translate ``src_path`` SRT, writing ``dst_path`` (default ``*_ES.srt``)."""
        src_path = Path(src_path)
        if dst_path is None:
            dst_path = src_path.with_name(f"{src_path.stem}.es.srt")
        else:
            dst_path = Path(dst_path)

        subs = parse_srt(src_path)
        if not subs:
            log.warning("No subtitles found in %s", src_path)
            dst_path.write_text("", encoding="utf-8")
            return dst_path

        texts = [s["text"] for s in subs]
        translated = self.translate_texts(texts)
        if len(translated) != len(texts):
            raise RuntimeError(
                f"Provider returned {len(translated)} items, expected {len(texts)}"
            )

        for sub, new_text in zip(subs, translated):
            sub["text"] = new_text

        dst_path.write_text(serialize_srt(subs), encoding="utf-8")
        log.info("Wrote %d translated subtitles to %s", len(subs), dst_path.name)
        return dst_path


# ---------------------------------------------------------------------------
# OpenAI provider (BJJ-aware, keeps technique names in English)
# ---------------------------------------------------------------------------

_BJJ_SYSTEM_PROMPT = """You translate Brazilian Jiu-Jitsu instructional subtitles from {src_name} to {tgt_name}.

TERMINOLOGY — CRITICAL, DO NOT CONFUSE THESE WORDS:
- "grip" = the HAND grabbing fabric/wrist. Keep as "grip" in Spanish, never translate to "agarre", "gancho", "enganche".
- "hook" = a LEG hooking (butterfly hook, inside hook, heel hook). Keep as "hook" in Spanish, never translate to "gancho" (except when the English itself literally says "hook" as a noun for a submission like "heel hook" — still keep in English).
- "underhook" / "overhook" = arm positions. Keep in English, never "sub-gancho" or similar invention.
- "frame" = rigid bone structure against the opponent. Keep as "frame".
- "post" = supporting limb on the ground. Keep as "post" (verb: "hacer post", "postear").
- "base" = body stability / supporting leg. Keep as "base".
- "pressure" = weight / pressure applied. Can be "presión" OR kept as "pressure"; prefer "presión" in narration, keep "pressure" if it's a noun referring to a technique ("pressure pass" -> "pressure pass").

Rules, non-negotiable:
1. Keep BJJ technique names and positions in English: guard, half-guard, closed guard, open guard, butterfly guard, de la Riva, reverse de la Riva, x-guard, spider guard, lasso guard, rubber guard, worm guard, knee shield, mount, side control, back mount, north-south, turtle, kimura, armbar, triangle, omoplata, heel hook, toe hold, kneebar, ankle lock, rear naked choke, guillotine, darce, anaconda, ezekiel, bow and arrow, cross collar choke, arm triangle, gogoplata, americana, kimura grip, gable grip, s-grip, pistol grip, berimbolo, knee cut, knee slice, leg drag, smash pass, tripod pass, toreando, stack pass, body lock pass, over-under pass, long step, backstep, sweep, bridge, shrimp, hip escape, technical stand-up, sprawl, scramble. Portuguese and Japanese technique names also stay as-is (juji gatame, sankaku, kesa gatame, mata leão, ashi garami, imanari roll, etc.).
2. Keep grappling English terms untranslated: grip, frame, framing, post, base, hook, lapel, sleeve, collar, gi, no-gi, tap, pin, pummel, pummeling, crossface, whizzer, reset, setup, entry, transition, top, bottom, stack, drill, roll, sparring.
3. Translate ordinary narration, explanations, transitions and body descriptions into informal {tgt_name} de España (castellano peninsular) as spoken by a coach. Use "tú" (nunca "vos" ni "usted"), vocabulario y giros de España ("vale", "coger", "ahora", "vamos a ver", "fíjate"), NUNCA latinoamericanismos ("agarrar" por "coger", "ustedes", "ahorita", "chévere", "pararse" por "ponerse de pie"). Conjugación 2ª persona singular en presente/imperativo siempre peninsular ("coges", "coge", "fíjate", "mira"). Evita el voseo y el "ustedeo".
4. Preserve meaning; do NOT add, shorten or merge content. One input item = one output item, same order.
5. Preserve line breaks inside a subtitle block exactly.
6. Output MUST be a JSON object of the exact shape: {{"t": ["translated item 1", "translated item 2", ...]}} with the same number of items as the input, in the same order.

EXAMPLES (illustrative, follow this style):
EN: "Establish your grips on the sleeve and collar before you open the guard."
ES: "Establece tus grips en la manga y el collar antes de abrir la guard."

EN: "I'm going to hook under his leg with my butterfly hook and sweep."
ES: "Voy a meter el hook bajo su pierna con el butterfly hook y barrer con un sweep."

EN: "Get your underhook, pummel through, and establish frames on the hips."
ES: "Consigue el underhook, haz pummel por dentro y coloca frames en las caderas."

EN: "From half guard, you threaten the kimura to force the pass."
ES: "Desde half guard, amenazas con la kimura para forzar el pass."

EN: "Control the ankle, then break the grip on your collar."
ES: "Controla el tobillo, luego rompe el grip en tu collar."
"""


# Dubbing mode: industry-standard iso-synchronous translation.
# The translator MUST keep each line within a character budget so the TTS
# comes out close to the original slot duration without needing audio stretch.
# This avoids the classic "ES is 40% longer than EN → robotic acceleration".
_BJJ_DUBBING_SYSTEM_PROMPT = """You adapt Brazilian Jiu-Jitsu instructional subtitles from {src_name} to {tgt_name} FOR DUBBING (voice-over).

[STRUCTURAL CONSTRAINT — JSON ARRAY MAPPING]
Input: JSON array of N strings. Output: JSON object {{"t": [...]}} with EXACTLY N strings.
- One input string → one output string. No merging, no splitting, no dropping.
- A very short input ("in order") must produce a very short output ("en este orden").
- Do not borrow text from adjacent items.
Failure to maintain 1:1 mapping breaks audio sync.

[TERMINOLOGY — DO NOT TRANSLATE THESE]
- "camping" (BJJ) = resting body weight on opponent's frames while passing guard. Output: "camping". NEVER "acampar", "campamento", "acampada".
- "grip" = hand grabbing fabric or wrist. Output: "grip". NEVER "agarre", "enganche".
- "hook" = leg hooking. Output: "hook". NEVER "gancho".
- "underhook", "overhook", "frame", "post", "base", "pass", "guard" → keep in English.

[PRIORITIES]
1. Preserve 1:1 array mapping (see structural constraint above).
2. Preserve all technical content — BJJ terms, instructions, steps.
3. Fit within `max_chars` budget per item. Soft overflow ≤20% accepted.
4. Compact connectors and drop pure filler, but never at the cost of technical information.

[LANGUAGE REGISTER]
Castellano peninsular de España, register of a BJJ coach.
- Second person: "tú" — imperativo "coge", "mira", "fíjate", "pon".
- FORBIDDEN: "ustedes" (use "vosotros"), "agarrar" (use "coger"), "ahorita", "pararse" (use "ponerse de pie").
- Use: "vale", "venga", "fíjate", "mira", "ahora".

[BJJ TERMS — KEEP IN ENGLISH]
camping, outside camping, inside camping, guard, half-guard, closed guard, open guard, butterfly guard, de la Riva, x-guard, spider guard, lasso guard, rubber guard, worm guard, knee shield, mount, side control, back mount, north-south, turtle, armbar, kimura, triangle, omoplata, heel hook, toe hold, kneebar, ankle lock, rear naked choke, guillotine, darce, anaconda, ezekiel, bow and arrow, americana, berimbolo, knee cut, knee slice, leg drag, smash pass, tripod pass, toreando, stack pass, body lock pass, sweep, bridge, shrimp, hip escape, technical stand-up, underhook, overhook, grip, frame, framing, post, base, hook, lapel, sleeve, collar, gi, no-gi, crossface, whizzer, pummel, sprawl, scramble, drill, setup, entry, transition, top, bottom, stack. Also Portuguese/Japanese: juji gatame, sankaku, ashi garami, mata leão, kesa gatame.

[OUTPUT FORMAT]
Return ONLY valid JSON: {{"t": ["item 1", "item 2", ...]}} — same count as input, same order.

EXAMPLES:
EN: "Get your grip on the sleeve first."
ES: "Coge el grip en la manga primero."

EN: "Use the butterfly hook to sweep him."
ES: "Usa el butterfly hook para barrerlo."

EN: "Underhook deep, then frame on his hip."
ES: "Underhook profundo, luego frame en su cadera."

EN: "So we do outside camping"
ES: "Así que hacemos outside camping"

EN: "in order"
ES: "en este orden"
"""


# Fill-budget variant — used when the source timing is speech-anchored
# (nivel 3 dub track). Here each slot equals real speaker talk time, so
# the ES adaptation must roughly match that duration; otherwise the TTS
# finishes early and leaves audible silence in the middle of the speech.
_BJJ_DUBBING_FILL_SYSTEM_PROMPT = """You adapt Brazilian Jiu-Jitsu instructional subtitles from {src_name} to {tgt_name} FOR DUBBING (voice-over).

[STRUCTURAL CONSTRAINT — JSON ARRAY MAPPING]
Input: JSON array of N strings. Output: JSON object {{"t": [...]}} with EXACTLY N strings.
- One input string → one output string. No merging, no splitting, no dropping.
- A very short input ("in order") must produce a very short output ("en este orden").
Failure to maintain 1:1 mapping breaks audio sync.

[TIME BUDGET CONSTRAINT]
Each item has a `target_chars` value representing real speaker talk time.
The TTS engine speaks at a fixed rate — character count = audio duration.
- Target: 95–108% of `target_chars`.
- Undershoot (< 95%) = silence gap in the middle of speech (worst failure).
- Hard ceiling: do not exceed 110% (TTS would speed up and sound robotic).
- If the literal translation is short, add natural connective phrases: "fíjate bien", "en este caso", "la idea es", "como ves", "de esta forma", "observa cómo". Do not invent technical content.

[TERMINOLOGY — DO NOT TRANSLATE THESE]
- "camping" (BJJ) = resting body weight on opponent's frames while passing guard. Output: "camping". NEVER "acampar", "campamento".
- "grip" = hand grabbing. Output: "grip". NEVER "agarre", "enganche".
- "hook" = leg hooking. Output: "hook". NEVER "gancho".
- "underhook", "overhook", "frame", "post", "base" → keep in English.

[LANGUAGE REGISTER]
Castellano peninsular de España, register of a BJJ coach.
- Second person: "tú" — imperativo "coge", "mira", "fíjate", "pon".
- FORBIDDEN: "ustedes" (use "vosotros"), "agarrar" (use "coger"), "ahorita", "pararse" (use "ponerse de pie").
- Use: "vale", "venga", "fíjate", "mira", "ahora".

[BJJ TERMS — KEEP IN ENGLISH]
camping, outside camping, inside camping, guard, half-guard, closed guard, open guard, butterfly guard, de la Riva, x-guard, spider guard, lasso guard, rubber guard, worm guard, knee shield, mount, side control, back mount, north-south, turtle, armbar, kimura, triangle, omoplata, heel hook, toe hold, kneebar, rear naked choke, guillotine, darce, anaconda, ezekiel, bow and arrow, sweep, pass, underhook, overhook, grip, frame, base, hook, lapel, sleeve, collar, mount, knee cut, knee slice, smash pass, leg drag, toreando, berimbolo, tripod pass, stack pass, body lock pass, crossface, whizzer, tap, pummel, sprawl, scramble, drill, setup, entry, transition, top, bottom, pressure, stack. Also: juji gatame, sankaku, ashi garami, mata leão, kesa gatame.

[OUTPUT FORMAT]
Return ONLY valid JSON: {{"t": ["item 1", "item 2", ...]}} — same count as input, same order.

EXAMPLES:
EN: "Get your grip on the sleeve first."
ES: "Coge el grip en la manga primero, fíjate."

EN: "Use the butterfly hook to sweep."
ES: "Usa el butterfly hook para hacer el sweep."

EN: "So we do outside camping"
ES: "Así que hacemos outside camping"

EN: "in order"
ES: "en este orden"
"""


class _BaseChatTranslator(_BaseTranslator, abc.ABC):
    """Shared logic for chat-completion-style providers (OpenAI, Ollama).

    Subclasses implement 4 abstract methods that diverge per provider dialect.
    The retry/feedback/prompt-construction body is shared here.
    """

    # Number of subtitle items per request. 40 is a sweet spot:
    # - keeps latency low
    # - keeps context coherent
    # - well under any token limit even for long lines
    _BATCH_SIZE = 40
    _MAX_RETRIES = 2

    # ---- HTTP dialect — subclasses MUST implement ----
    @abc.abstractmethod
    def _endpoint_url(self) -> str:
        """Return the absolute chat-completion endpoint URL for the provider."""
        ...

    @abc.abstractmethod
    def _request_headers(self) -> dict[str, str]:
        """Return the HTTP headers (auth + content-type) for the provider."""
        ...

    @abc.abstractmethod
    def _wrap_chat_body(self, messages: list[dict[str, Any]], json_mode: bool) -> dict[str, Any]:
        """Build the provider-specific HTTP request body for a chat completion.

        Args:
            messages: list of ``{"role": ..., "content": ...}`` dicts.
            json_mode: if True, instruct the provider to enforce strict JSON
                output (OpenAI: ``response_format={"type":"json_object"}``;
                Ollama: ``format="json"``). Subclasses may ignore the flag if
                their provider does not support strict JSON, but must document
                the deviation.

        Returns:
            Body dict ready for ``httpx.Client.post(json=...)``.
        """
        ...

    @abc.abstractmethod
    def _extract_message_content(self, resp_json: dict[str, Any]) -> str:
        """Pull the assistant message text out of a parsed provider response."""
        ...

    def _do_post(self, body: dict, headers: dict) -> httpx.Response:
        """POST to the provider endpoint. Subclasses may override (e.g. auto-pull)."""
        return _post_with_retry(
            self._endpoint_url(),
            headers=headers,
            json_body=body,
            provider_label=self.provider_label,
        )

    # ---- Optional hook (default to module-level timeout) ----
    def _request_timeout(self) -> float:
        """Override per-provider HTTP timeout. Default = module-level ``_TIMEOUT`` (120s).

        Hook reserved for subclasses. The current implementation of
        ``_post_with_retry`` reads the timeout from the module-level
        ``_TIMEOUT`` constant, so overriding this method has no effect today.
        Override + a future task that wires this into the post helper is
        required before tuning per-provider timeouts. Ollama local with
        qwen2.5-7b can take 30-120s for large batches; OpenAI gpt-4o-mini
        typically <5s.
        """
        return _TIMEOUT

    # ---- Lógica compartida ----
    def translate_texts(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        out: list[str] = []
        for chunk in _chunks(texts, self._BATCH_SIZE):
            try:
                out.extend(self._translate_batch(chunk))
            except RuntimeError:
                # Batch failed after retries — fall back to one-by-one
                log.warning("Batch of %d failed, translating one-by-one", len(chunk))
                for item in chunk:
                    out.extend(self._translate_batch([item]))
        return out

    def translate_for_dubbing(
        self,
        items: list[dict],
        cps: float = 17.0,
        fill_budget: bool = False,
    ) -> list[str]:
        """Translate ES-dub style: enforces char budget per item from slot duration.

        Each ``item`` must be ``{"text": str, "duration_ms": int}``.
        Budget per item = (duration_ms / 1000) * cps, clamped to [12, 220].

        ``fill_budget=True`` switches to the speech-anchored mode (nivel 3):
        the prompt asks the model to actively reach ~85-105% of the budget
        instead of treating it as a ceiling. Use it when the input slots
        represent real speaker talk time — otherwise undershooting produces
        audible silence in the middle of the dub.
        """
        if not items:
            return []
        # Fill-budget prompt is stricter (per-item targets, upper/lower bounds)
        # and the model occasionally merges/splits items when the batch is
        # large. Smaller batches give cleaner 1:1 counts.
        batch_size = min(15, self._BATCH_SIZE) if fill_budget else self._BATCH_SIZE
        out: list[str] = []
        for chunk in _chunks(items, batch_size):
            try:
                out.extend(self._translate_dubbing_batch(chunk, cps, fill_budget))
            except RuntimeError:
                log.warning(
                    "Dubbing batch of %d failed, falling back one-by-one",
                    len(chunk),
                )
                for item in chunk:
                    out.extend(self._translate_dubbing_batch([item], cps, fill_budget))
        return out

    def _translate_batch(self, texts: list[str]) -> list[str]:
        system = _BJJ_SYSTEM_PROMPT.format(
            src_name=_lang_name(self.source_lang),
            tgt_name=_lang_name(self.target_lang),
        )
        user_payload = {"items": texts}
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Translate each item in `items`. Return JSON "
                    '{"t": [...]} with the same number of items in the same order.\n'
                    + json.dumps(user_payload, ensure_ascii=False)
                ),
            },
        ]
        body = self._wrap_chat_body(messages, json_mode=True)
        headers = self._request_headers()

        last_err: Exception | None = None
        for attempt in range(1 + self._MAX_RETRIES):
            r = self._do_post(body, headers)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"{self.provider_label} error {r.status_code}: {r.text[:300]}"
                )

            data = r.json()
            try:
                content = self._extract_message_content(data)
                parsed = json.loads(content)
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{self.provider_label} response parse failed: {exc}"
                ) from exc

            items = parsed.get("t")
            if not isinstance(items, list):
                items = parsed.get("items")
            if isinstance(items, list) and len(items) == len(texts):
                translated = [_coerce_to_str(x) for x in items]
                foreign_idx = _foreign_script_indices(translated)
                if foreign_idx and attempt < self._MAX_RETRIES:
                    log.warning(
                        "Foreign script in %d items, retrying batch", len(foreign_idx),
                    )
                    body["messages"].append(
                        {"role": "assistant", "content": json.dumps({"t": translated}, ensure_ascii=False)},
                    )
                    body["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                "Some items contain non-Spanish characters. "
                                "Translate ALL items into Spanish (castellano peninsular) only. "
                                'Return JSON {"t": [...]} with the same number of items.\n'
                                "Offenders: "
                                + json.dumps(
                                    [{"index": i, "current": translated[i], "source": texts[i]} for i in foreign_idx],
                                    ensure_ascii=False,
                                )
                            ),
                        },
                    )
                    last_err = RuntimeError(f"Foreign script in items {foreign_idx}")
                    continue
                return translated

            last_err = RuntimeError(
                f"{self.provider_label} returned {len(items) if isinstance(items, list) else 'non-list'} items, "
                f"expected {len(texts)}"
            )
            log.warning("%s count mismatch (attempt %d/%d), retrying…",
                        self.provider_label, attempt + 1, 1 + self._MAX_RETRIES)

        raise last_err  # type: ignore[misc]

    def _translate_dubbing_batch(
        self, items: list[dict], cps: float, fill_budget: bool = False,
    ) -> list[str]:
        """Iso-synchronous batch: sends items with max_chars budget.

        Retries items that exceed budget by asking the model to shorten them.
        Budget floor = 12 chars (tiny slots still need something sayable),
        ceiling = 220 (Chatterbox chunk limit).

        ``fill_budget=True`` uses the speech-anchored prompt and enforces a
        LOWER bound (output must reach ≥80% of budget) in addition to the
        usual upper bound; retries items that undershoot.
        """
        def _budget(d_ms: int) -> int:
            raw = int((d_ms / 1000.0) * cps)
            return max(15, min(260, raw))

        budget_key = "target_chars" if fill_budget else "max_chars"
        payload_items = [
            {
                "text": it.get("text", ""),
                budget_key: _budget(int(it.get("duration_ms", 1500))),
            }
            for it in items
        ]
        system_tpl = _BJJ_DUBBING_FILL_SYSTEM_PROMPT if fill_budget else _BJJ_DUBBING_SYSTEM_PROMPT
        system = system_tpl.format(
            src_name=_lang_name(self.source_lang),
            tgt_name=_lang_name(self.target_lang),
            cps=f"{cps:.0f}",
        )
        user_instruction = (
            "Translate each item in `items` into an ADAPTED dubbing line "
            "that LANDS BETWEEN 95% AND 108% of its `target_chars`. "
            "Undershooting produces silence gaps in the dub (worst failure). "
            "Prefer a slight overshoot (≤108%) over any undershoot. "
            'Return JSON {"t": [...]} with the same number of items in the '
            "same order.\n"
            if fill_budget else
            "Translate each item in `items` into an ADAPTED dubbing line "
            "that fits within its `max_chars` budget. "
            'Return JSON {"t": [...]} with the same number of items in the '
            "same order.\n"
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    user_instruction
                    + json.dumps({"items": payload_items}, ensure_ascii=False)
                ),
            },
        ]
        body = self._wrap_chat_body(messages, json_mode=True)
        headers = self._request_headers()

        last_err: Exception | None = None
        for attempt in range(1 + self._MAX_RETRIES):
            r = self._do_post(body, headers)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"{self.provider_label} error {r.status_code}: {r.text[:300]}"
                )
            try:
                content = self._extract_message_content(r.json())
                parsed = json.loads(content)
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{self.provider_label} response parse failed: {exc}"
                ) from exc

            result = parsed.get("t")
            if not isinstance(result, list):
                result = parsed.get("items")
            if not (isinstance(result, list) and len(result) == len(items)):
                got = len(result) if isinstance(result, list) else "non-list"
                last_err = RuntimeError(
                    f"{self.provider_label} returned non-matching count "
                    f"({got} vs {len(items)})"
                )
                # Feedback retry: tell the model the count is wrong so it
                # stops merging/splitting items. Without this the retry
                # loop just hits the same mistake 3 times.
                body["messages"].append(
                    {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
                )
                body["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            f"Your response had {got} items but the input had "
                            f"{len(items)}. You MUST output EXACTLY "
                            f"{len(items)} items in the same order as the "
                            "input. Never merge, split, or drop items — a "
                            "short input item gets a short translation, not "
                            "a merge with its neighbour. Re-translate now "
                            'and return JSON {"t": [...]} with exactly '
                            f"{len(items)} strings."
                        ),
                    },
                )
                continue

            result = [_coerce_to_str(x) for x in result]

            # Detect items where the model mixed in non-Latin script (e.g. Chinese).
            # Treat them like budget offenders: request a targeted retry.
            foreign_idx = _foreign_script_indices(result)
            if foreign_idx and attempt < self._MAX_RETRIES:
                log.warning(
                    "Foreign script detected in %d items (indices %s); requesting retranslation",
                    len(foreign_idx), foreign_idx,
                )
                foreign_feedback = [
                    {
                        "index": i,
                        "problem": "contains non-Spanish characters (e.g. Chinese/Arabic/Cyrillic) — retranslate into Spanish only",
                        "current": result[i],
                        "source": payload_items[i]["text"],
                    }
                    for i in foreign_idx
                ]
                body["messages"].append(
                    {"role": "assistant", "content": json.dumps({"t": result}, ensure_ascii=False)},
                )
                body["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            "Some items contain non-Spanish characters. "
                            "Rewrite ALL items in Spanish only (castellano peninsular). "
                            'Return JSON {"t": [...]} with the same number of items.\n'
                            "Offending items: "
                            + json.dumps(foreign_feedback, ensure_ascii=False)
                        ),
                    },
                )
                continue

            budgets = [p[budget_key] for p in payload_items]
            # Upper bound: fill_budget allows +10% overshoot (TTS stretch
            # absorbs softly). Reader mode is lenient (1.25x).
            upper_ratio = 1.10 if fill_budget else 1.25
            over_idx: list[int] = [
                i for i, (txt, b) in enumerate(zip(result, budgets))
                if len(txt) > int(b * upper_ratio)
            ]
            # Undershoot check: the model tends to sacrifice content to
            # respect the budget — produces ES that drops whole clauses
            # ("and we'll first talk about the advantages" → "ventajas.").
            # Reject anything below 0.70x of the EN budget in slot mode
            # (0.90x in fill mode, stricter because slot = real talk time).
            lower_ratio = 0.90 if fill_budget else 0.70
            under_idx = [
                i for i, (txt, b) in enumerate(zip(result, budgets))
                if len(txt) < int(b * lower_ratio)
            ]
            offenders = sorted(set(over_idx + under_idx))

            if not offenders or attempt == self._MAX_RETRIES:
                if offenders:
                    log.warning(
                        "Budget mismatch on %d items after retries "
                        "(over=%d, under=%d); accepting",
                        len(offenders), len(over_idx), len(under_idx),
                    )
                return result

            # Feedback loop: tell the model which items missed budget and
            # in which direction, then request a fresh pass.
            feedback = [
                {
                    "index": i,
                    "current_len": len(result[i]),
                    "budget": payload_items[i][budget_key],
                    "direction": (
                        "TOO LONG — shorten"
                        if i in over_idx else
                        "TOO SHORT — expand without changing meaning"
                    ),
                    "current": result[i],
                    "source": payload_items[i]["text"],
                }
                for i in offenders
            ]
            retry_msg = (
                "These items did not land within budget. Rewrite ALL items "
                "(not only offenders). Items marked TOO SHORT must be "
                "expanded with natural connective phrasing so the dub "
                "covers the speaker's talk time without silence. "
                'Return JSON {"t": [...]} with every item rewritten.\n'
                "Details: "
                if fill_budget else
                "These items missed budget. Rewrite ALL items (not only "
                "offenders). Items marked TOO LONG: shorten without losing "
                "technical content. Items marked TOO SHORT: you dropped "
                "meaning from the source — add the missing clauses back in "
                "natural coach Spanish. Content preservation is priority #1, "
                "even if a rewritten item lands slightly over budget. "
                'Return JSON {"t": [...]} with every item rewritten.\n'
                "Details: "
            )
            body["messages"].append(
                {"role": "assistant", "content": json.dumps({"t": result}, ensure_ascii=False)},
            )
            body["messages"].append(
                {
                    "role": "user",
                    "content": retry_msg + json.dumps(feedback, ensure_ascii=False),
                },
            )

        if last_err:
            raise last_err
        return result


class OpenAITranslator(_BaseChatTranslator):
    """Translate SRT files via OpenAI Chat Completions (gpt-4o-mini default).

    Batches subtitle items into a single JSON request for coherence and cost.
    Uses ``response_format=json_object`` for reliable parsing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        source_lang: str = "EN",
        target_lang: str = "ES",
        temperature: float = 0.2,
        base_url: str | None = None,
    ) -> None:
        super().__init__(source_lang, target_lang)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not provided (env or constructor)")
        self.api_key = key
        self.model = model
        self.temperature = temperature
        self.provider_label = "OpenAI"
        url = (base_url or OPENAI_URL).rstrip("/")
        # If base_url was passed without the /chat/completions suffix, fix it.
        if not url.endswith("/chat/completions"):
            if url.endswith("/v1"):
                url = f"{url}/chat/completions"
        self.url = url

    def _endpoint_url(self) -> str:
        return self.url

    def _request_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _wrap_chat_body(self, messages: list[dict], json_mode: bool) -> dict:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _extract_message_content(self, resp_json: dict) -> str:
        return resp_json["choices"][0]["message"]["content"]


class OllamaTranslator(_BaseChatTranslator):
    """Translate via Ollama native /api/chat with strict JSON mode."""

    # Qwen2.5-7b loses item count on large batches — keep small for reliable 1:1 mapping
    _BATCH_SIZE = 8

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        source_lang: str = "EN",
        target_lang: str = "ES",
        temperature: float = 0.2,
        base_url: str | None = None,
    ) -> None:
        super().__init__(source_lang, target_lang)
        self.model = model
        self.temperature = temperature
        self.provider_label = "Ollama"
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        ).rstrip("/")

    def _endpoint_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _request_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _wrap_chat_body(self, messages: list[dict], json_mode: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            body["format"] = "json"
        return body

    def _extract_message_content(self, resp_json: dict[str, Any]) -> str:
        return resp_json["message"]["content"]

    def _ensure_model(self) -> None:
        """Pull the model if Ollama returns 404. Uses stream=True to survive CDN retries."""
        pull_url = f"{self.base_url}/api/pull"
        log.info("Ollama model '%s' not found — pulling now (may take several minutes)…", self.model)
        with httpx.Client(timeout=1800.0) as client:
            with client.stream("POST", pull_url, json={"name": self.model, "stream": True}) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"Ollama pull failed ({resp.status_code}): {resp.read().decode()[:200]}")
                last_status = ""
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        last_status = data.get("status", last_status)
                        if data.get("error"):
                            raise RuntimeError(f"Ollama pull error: {data['error']}")
                    except (json.JSONDecodeError, RuntimeError):
                        raise
                    except Exception:
                        pass
        if last_status != "success":
            raise RuntimeError(f"Ollama pull ended with status: {last_status!r}")
        log.info("Ollama pull complete for '%s'", self.model)

    def _do_post(self, body: dict, headers: dict) -> httpx.Response:
        """POST to Ollama; if 404 pull the model and retry once."""
        r = _post_with_retry(
            self._endpoint_url(),
            headers=headers,
            json_body=body,
            provider_label=self.provider_label,
        )
        if r.status_code == 404:
            self._ensure_model()
            r = _post_with_retry(
                self._endpoint_url(),
                headers=headers,
                json_body=body,
                provider_label=self.provider_label,
            )
        return r


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_translator(
    provider: str,
    *,
    api_key: str | None = None,
    source_lang: str = "EN",
    target_lang: str = "ES",
    model: str | None = None,
    formality: str | None = None,
) -> _BaseTranslator:
    """Build a translator for the requested provider name."""
    p = (provider or "").lower().strip()
    if p == "ollama":
        return OllamaTranslator(
            model=model or "qwen2.5:7b-instruct-q4_K_M",
            source_lang=source_lang,
            target_lang=target_lang,
        )
    if p in ("openai", "gpt", "chatgpt"):
        return OpenAITranslator(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            source_lang=source_lang,
            target_lang=target_lang,
        )
    raise ValueError(f"Unknown translation provider: {provider!r}")


_FOREIGN_SCRIPT_RE = re.compile(
    r"[Ѐ-ӿ"   # Cyrillic
    r"؀-ۿ"   # Arabic
    r"ऀ-ॿ"   # Devanagari
    r"一-鿿"   # CJK Unified (Chinese/Japanese kanji)
    r"぀-ゟ"   # Hiragana
    r"゠-ヿ"   # Katakana
    r"가-힯"   # Korean
    r"]"
)


def _foreign_script_indices(items: list[str]) -> list[int]:
    """Return indices of items that contain non-Latin/non-Spanish script characters."""
    return [i for i, txt in enumerate(items) if _FOREIGN_SCRIPT_RE.search(txt)]


def _coerce_to_str(x: Any) -> str:
    """Extract a plain string from whatever the LLM returned in a translation slot.

    The model occasionally echoes back the input payload structure instead of
    a plain string.  Handle the common shapes gracefully before falling back
    to str().
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # {"t": "..."} or {"text": "..."} — take whichever is present
        for key in ("t", "text", "translated"):
            if key in x and isinstance(x[key], str):
                return x[key]
        # Any string value as last resort
        for v in x.values():
            if isinstance(v, str):
                return v
    if isinstance(x, list) and x and isinstance(x[0], str):
        # ["translated text", "49"] — first element is the translation
        return x[0]
    return str(x)


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
