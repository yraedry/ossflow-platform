"""OpenAI-powered post-processing of WhisperX-generated SRT.

Fixes two classes of artifacts that leak through WhisperX + alignment:

1. Syllable duplication on word boundaries (``instructionalal``,
   ``primarilyarily``) — a known beam-search artifact.
2. Phrase boundaries cut mid-clause so a line ends in a preposition or
   conjunction (``...if you're one`` / ``one seated...``) — VAD timing
   splits words off their neighbours.

Constraints for the model:
  - Keep the same number of blocks (1:1 with input).
  - Keep timestamps untouched (we only send text).
  - Only reassign words across adjacent blocks to fix broken boundaries;
    never invent content, never merge/split blocks.

The cleanup runs in batches of 40 blocks, the same batch size as the
translator, so latency stays bounded on long instructionals.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable

import httpx

log = logging.getLogger("subtitler")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_TIMEOUT = 120.0
_BATCH_SIZE = 40
_MAX_RETRIES = 2

_SYSTEM_PROMPT = (
    "You clean up raw English subtitles produced by automatic speech "
    "recognition (WhisperX) on Brazilian Jiu-Jitsu instructional videos. "
    "You receive an ordered list of subtitle blocks and must return the "
    "same list with the text cleaned up.\n\n"
    "Rules — follow strictly:\n"
    "1. Return EXACTLY the same number of items in the same order.\n"
    "2. Never invent content. If you are unsure, leave the original text.\n"
    "3. Fix obvious syllable duplication on word boundaries introduced by "
    "the ASR (e.g. 'instructionalal' -> 'instructional', "
    "'primarilyarily' -> 'primarily', 'handfightt' -> 'hand fight').\n"
    "4. Fix mis-transcribed BJJ terms when the evidence is strong. Common "
    "ASR errors and their correct form:\n"
    "   - 'butterflip' / 'butterflick' / 'butter flip' -> 'butterfly'\n"
    "   - 'butter fly' -> 'butterfly'\n"
    "   - 'grip fighting' misheard as 'grip fighten' -> 'grip fighting'\n"
    "   - 'de la river' / 'de la rea' / 'delaRiva' -> 'De La Riva'\n"
    "   - 'half guard' misheard as 'hall guard' / 'hal guard' -> 'half guard'\n"
    "   - 'x guard' / 'ex guard' / 'x-guard' -> 'X-guard'\n"
    "   - 'kimora' / 'kamira' / 'kimurra' -> 'kimura'\n"
    "   - 'arm bar' / 'armar' / 'arm bar' -> 'armbar'\n"
    "   - 'torando' / 'torrando' / 'toreado' -> 'toreando'\n"
    "   - 'heal hook' / 'heal-hook' -> 'heel hook'\n"
    "   - 'underhook' misheard as 'under hoop' -> 'underhook'\n"
    "   - 'berimbolo' misheard as 'berimboro' / 'berinbolo' -> 'berimbolo'\n"
    "   - 'wizer' / 'whiz her' -> 'whizzer'\n"
    "   - 'omaplata' / 'omoprata' -> 'omoplata'\n"
    "   - 'darse' / 'darshe' / 'd'arce' variations -> 'D'Arce'\n"
    "   Only apply these if the surrounding context is clearly BJJ technique "
    "talk — do NOT over-correct generic English words.\n"
    "5. If a block ends mid-clause (trailing preposition/conjunction like "
    "'to', 'and', 'so', 'but', 'of', 'from', 'if you're one') and the next "
    "block continues the same sentence awkwardly, you MAY move a few words "
    "between those two adjacent blocks so each reads as a natural fragment. "
    "You may not move words across more than one boundary, and the total "
    "word content of the two blocks must stay the same.\n"
    "6. Never merge two blocks into one. Never split one block into two.\n"
    "7. Preserve BJJ terminology exactly: 'grip' (the hand grabbing) and "
    "'hook' (the leg hooking) are DIFFERENT things — never conflate them. "
    "Keep all technique names in English ('guard', 'hand fight', 'toreando', "
    "'De La Riva', 'kimura', 'armbar', 'grip', 'hook', 'underhook', "
    "'overhook', 'whizzer', 'omoplata'). Do not translate.\n"
    "8. Keep punctuation style close to the original. Do not add emoji or "
    "formatting.\n"
    "9. Keep line breaks ('\\n') only if they were present in the input."
)


def _chunks(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class SrtPostprocessor:
    """OpenAI-based SRT cleanup. Idempotent on already-clean input."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        base_url: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not provided (env or constructor)")
        self.api_key = key
        self.model = model
        self.temperature = temperature
        self.url = (base_url or OPENAI_URL).rstrip("/")
        if not self.url.endswith("/chat/completions"):
            if self.url.endswith("/v1"):
                self.url = f"{self.url}/chat/completions"

    def clean_subtitles(self, subtitles: list[dict]) -> list[dict]:
        """Return a new list with text cleaned. Timestamps are untouched.

        On any failure the original block is kept as-is so a partial failure
        never corrupts the SRT.
        """
        if not subtitles:
            return subtitles

        cleaned: list[dict] = []
        for batch in _chunks(subtitles, _BATCH_SIZE):
            texts = [str(s.get("text", "")) for s in batch]
            try:
                new_texts = self._clean_batch(texts)
            except RuntimeError as exc:
                log.warning("SRT postprocess batch failed (%s); keeping originals", exc)
                new_texts = texts
            if len(new_texts) != len(batch):
                log.warning(
                    "SRT postprocess count mismatch (%d vs %d); keeping originals",
                    len(new_texts), len(batch),
                )
                new_texts = texts
            for sub, new in zip(batch, new_texts):
                out = dict(sub)
                out["text"] = new if isinstance(new, str) and new.strip() else sub.get("text", "")
                cleaned.append(out)
        return cleaned

    def _clean_batch(self, texts: list[str]) -> list[str]:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Clean each item in `items`. Return JSON "
                        '{"t": [...]} with the same number of items in the '
                        "same order.\n"
                        + json.dumps({"items": texts}, ensure_ascii=False)
                    ),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for _ in range(1 + _MAX_RETRIES):
            try:
                with httpx.Client(timeout=_TIMEOUT) as client:
                    r = client.post(self.url, headers=headers, json=body)
                if r.status_code >= 400:
                    raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:300]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                items = parsed.get("t")
                if not isinstance(items, list):
                    raise RuntimeError("OpenAI response missing 't' list")
                return [str(x) for x in items]
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(f"OpenAI postprocess failed: {last_err}")
