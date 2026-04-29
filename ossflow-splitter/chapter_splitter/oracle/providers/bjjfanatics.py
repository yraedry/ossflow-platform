from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher
from typing import Optional

import httpx
from selectolax.parser import HTMLParser

from ..errors import (
    HTMLChangedError,
    ProviderScrapeError,
    ProviderSearchError,
    ProviderTimeoutError,
)
from ..models import Candidate, OracleChapter, OracleResult, OracleVolume

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_TIMEOUT_S = 15.0
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 1.0


def _http_get_with_retry(
    client: httpx.Client, url: str, *, params: dict | None = None
) -> httpx.Response:
    """GET with exponential backoff on timeouts and 5xx. Raises last error."""
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = client.get(url, params=params)
            if resp.status_code < 500:
                return resp
            last_exc = httpx.HTTPStatusError(
                f"server returned {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        except httpx.TimeoutException as e:
            last_exc = e
        except httpx.HTTPError as e:
            last_exc = e
            break
        if attempt < _MAX_ATTEMPTS:
            sleep_s = _BACKOFF_BASE_S * (2 ** (attempt - 1))
            logger.warning(
                "GET %s attempt %d/%d failed (%s); retrying in %.1fs",
                url, attempt, _MAX_ATTEMPTS, type(last_exc).__name__, sleep_s,
            )
            time.sleep(sleep_s)
    raise last_exc  # type: ignore[misc]
_SEARCH_ENDPOINT = (
    "https://bjjfanatics-msigw.ondigitalocean.app/v4/products/search"
)
_PRODUCT_BASE = "https://bjjfanatics.com/products/"

_VOLUME_NUM_RE = re.compile(r"Volume\s+(\d+)", re.IGNORECASE)
_TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})\s*$")
_RANGE_RE = re.compile(
    r"^\s*(?P<start>[\d:]+)\s*-\s*\(\s*(?P<total>[\d:]+)\s*\)\s*$"
)
# Extracts time tokens like "1:35", "25:10", "1:09:38" from any cell text.
# Used as a robust fallback when the cell doesn't fit the strict patterns
# above (e.g. "0:00 - 1:35" on intermediate rows, or missing parentheses
# on the last row).
_TIME_TOKEN_RE = re.compile(r"\d+(?::\d+)+")


def _field_score(query: str, candidate: str) -> float:
    """Score a single field. Substring match = 1.0; else SequenceMatcher ratio."""
    if not query:
        return 1.0
    if not candidate:
        return 0.0
    q = query.lower().strip()
    c = candidate.lower().strip()
    if q in c:
        return 1.0
    return SequenceMatcher(None, q, c).ratio()


def _score_candidate(
    title: str, author: str | None, cand_title: str, cand_vendor: str
) -> float:
    """Weighted score: 70% title, 30% author. Substring on title gives 1.0."""
    title_score = _field_score(title, cand_title)
    if author:
        author_score = _field_score(author, cand_vendor)
        return max(0.0, min(1.0, 0.7 * title_score + 0.3 * author_score))
    return max(0.0, min(1.0, title_score))


def _title_case(s: str) -> str:
    """Capitalize first letter of each word, preserve short acronyms (≤4 all-caps).

    Safer than ``str.title()`` which mangles apostrophes ("don't" → "Don'T").
    Tokens like "BJJ", "MMA", "DLR", "ADCC" are kept as-is.
    """
    def cap(word: str) -> str:
        if not word:
            return word
        if word.isupper() and len(word) <= 4 and word.isalpha():
            return word
        return word[0].upper() + word[1:].lower()

    return " ".join(cap(w) for w in s.split())


def _parse_time(raw: str) -> int:
    """Parse a time string into total seconds.

    Accepts ``"0"``, ``"M:SS"``, ``"MM:SS"``, ``"H:MM:SS"``.
    """
    if raw is None:
        raise ValueError("time string is None")
    s = raw.strip()
    if not s:
        raise ValueError("empty time string")
    if s.isdigit():
        return int(s)
    parts = s.split(":")
    if not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"invalid time string: {raw!r}")
    nums = [int(p) for p in parts]
    if len(nums) == 2:
        m, sec = nums
        return m * 60 + sec
    if len(nums) == 3:
        h, m, sec = nums
        return h * 3600 + m * 60 + sec
    raise ValueError(f"invalid time string: {raw!r}")


def _parse_range(raw: str) -> tuple[int, int]:
    """Parse the last-chapter cell ``"MM:SS - (MM:SS)"`` into (start, total)."""
    if raw is None:
        raise ValueError("range string is None")
    m = _RANGE_RE.match(raw)
    if not m:
        raise ValueError(f"not a range expression: {raw!r}")
    return _parse_time(m.group("start")), _parse_time(m.group("total"))


class BJJFanaticsProvider:
    id = "bjjfanatics"
    display_name = "BJJ Fanatics"
    domains = ["bjjfanatics.com"]

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self._client = client

    # ------------------------------------------------------------------ search
    def search(
        self, title: str, author: str | None = None
    ) -> list[Candidate]:
        query = title.strip()
        if author:
            query = f"{query} {author}".strip()
        params = {"term": query}
        try:
            with httpx.Client(
                timeout=_TIMEOUT_S,
                headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            ) as client:
                resp = _http_get_with_retry(client, _SEARCH_ENDPOINT, params=params)
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                f"timeout searching bjjfanatics for {query!r} after "
                f"{_MAX_ATTEMPTS} attempts"
            ) from e
        except httpx.HTTPError as e:
            raise ProviderSearchError(
                f"network error searching bjjfanatics: {e}"
            ) from e

        if resp.status_code >= 400:
            raise ProviderSearchError(
                f"bjjfanatics search returned HTTP {resp.status_code}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderSearchError(
                f"bjjfanatics search returned non-JSON response: {e}"
            ) from e

        items = self._extract_search_items(data)

        candidates: list[Candidate] = []
        for it in items:
            cand_title = (
                it.get("title")
                or it.get("name")
                or it.get("product_title")
                or ""
            )
            authors_field = it.get("authors")
            if isinstance(authors_field, list) and authors_field:
                vendor = str(authors_field[0])
            else:
                vendor = (
                    it.get("vendor")
                    or it.get("author")
                    or it.get("brand")
                    or ""
                )
            handle = (
                it.get("url")
                or it.get("handle")
                or it.get("slug")
                or ""
            )
            if not cand_title or not handle:
                continue

            if handle.startswith("http"):
                url = handle
            else:
                url = _PRODUCT_BASE + handle.lstrip("/").replace(
                    "products/", ""
                )

            score = _score_candidate(title, author, cand_title, vendor)
            candidates.append(
                Candidate(
                    url=url,
                    title=cand_title,
                    author=vendor or None,
                    score=max(0.0, min(1.0, score)),
                    provider_id=self.id,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    @staticmethod
    def _extract_search_items(data: object) -> list[dict]:
        """Best-effort extraction of product list from the JSON envelope."""
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("videos", "products", "items", "results", "hits", "data"):
                v = data.get(key)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
                if isinstance(v, dict):
                    for k2 in ("videos", "products", "items", "results", "hits"):
                        v2 = v.get(k2)
                        if isinstance(v2, list):
                            return [x for x in v2 if isinstance(x, dict)]
        return []

    # ------------------------------------------------------------------ scrape
    def scrape(self, url: str) -> OracleResult:
        try:
            with httpx.Client(
                timeout=_TIMEOUT_S,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
            ) as client:
                resp = _http_get_with_retry(client, url)
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                f"timeout scraping {url} after {_MAX_ATTEMPTS} attempts"
            ) from e
        except httpx.HTTPError as e:
            raise ProviderScrapeError(
                f"network error scraping {url}: {e}"
            ) from e

        if resp.status_code >= 400:
            raise ProviderScrapeError(
                f"scrape {url} returned HTTP {resp.status_code}"
            )

        return self._parse_html(resp.text, url)

    # ------------------------------------------------------------- parsing
    def _parse_html(self, html: str, url: str) -> OracleResult:
        tree = HTMLParser(html)

        title_node = tree.css_first("h1.product-title")
        if title_node is None:
            raise HTMLChangedError(
                "h1.product-title not found - bjjfanatics layout changed"
            )

        poster_url = self._pick_best_poster(tree, url)

        accordion = tree.css_first("div.product__course-content-accordion")
        if accordion is None:
            raise HTMLChangedError(
                "div.product__course-content-accordion not found - "
                "no COURSE CONTENT section"
            )

        # Walk children of the accordion; pair each h3.product__course-title
        # with the next sibling div.product__course-content.
        volumes: list[OracleVolume] = []
        children = list(accordion.iter(include_text=False))
        i = 0
        while i < len(children):
            node = children[i]
            tag = node.tag
            classes = (node.attributes.get("class") or "") if node.attributes else ""
            if tag == "h3" and "product__course-title" in classes:
                # Find the next sibling div.product__course-content
                content = None
                j = i + 1
                while j < len(children):
                    nxt = children[j]
                    nxt_classes = (
                        nxt.attributes.get("class") or ""
                    ) if nxt.attributes else ""
                    if (
                        nxt.tag == "div"
                        and "product__course-content" in nxt_classes
                        and "product__course-content-header" not in nxt_classes
                    ):
                        content = nxt
                        break
                    if nxt.tag == "h3" and "product__course-title" in nxt_classes:
                        break
                    j += 1

                if content is not None:
                    vol = self._parse_volume(node, content)
                    if vol is not None:
                        volumes.append(vol)
                i = j
            else:
                i += 1

        if not volumes:
            raise HTMLChangedError(
                "no volumes parsed from accordion - structure changed"
            )

        try:
            return OracleResult(
                product_url=url,
                provider_id=self.id,
                poster_url=poster_url,
                volumes=volumes,
            )
        except Exception as e:  # pydantic ValidationError, etc.
            raise ProviderScrapeError(
                f"failed building OracleResult: {e}"
            ) from e

    # ------------------------------------------------------------- poster

    # Prefer portraits but accept anything at least slightly taller than wide.
    # BJJFanatics ships both 4:3 hero shots and 2:3 cover variants; the
    # portrait variant is the one that doesn't need letterbox in the UI.
    _POSTER_RATIO_MIN = 0.60  # width/height — lower = more portrait
    _POSTER_RATIO_PREFERRED = 0.70  # anything <= this is considered portrait
    _POSTER_MIN_WIDTH = 400  # ignore thumbnails/icons

    def _pick_best_poster(self, tree: HTMLParser, url: str) -> str | None:
        """Find the most poster-like image for a BJJFanatics product page.

        Tries, in order:
          1. Shopify /products/<handle>.json — images come with width/height
          2. Gallery <img> tags in the HTML (parsed for width attrs)
          3. og:image / link[rel=image_src] as final fallback

        Picks the image with the most portrait aspect ratio ≥ _POSTER_RATIO_MIN.
        Returns ``None`` only if nothing at all matched.
        """
        candidates: list[tuple[str, int, int]] = []  # (url, w, h)

        # 1. Shopify product JSON — authoritative dimensions
        try:
            candidates.extend(self._poster_candidates_from_json(url))
        except Exception as exc:  # pragma: no cover — network/parse noise
            logger.debug("shopify json poster lookup failed: %s", exc)

        # 2. HTML gallery fallback
        if not candidates:
            candidates.extend(self._poster_candidates_from_html(tree))

        # 3. og:image with og:image:width/height meta hints
        og_candidate = self._poster_candidate_from_og(tree)
        if og_candidate is not None:
            candidates.append(og_candidate)

        best = self._select_portrait(candidates)
        if best:
            return self._strip_shopify_size(best)

        # Last-resort: raw og:image / image_src, no dimensions known
        og = tree.css_first('meta[property="og:image"]')
        if og is not None and og.attributes:
            content = og.attributes.get("content")
            if content and content.strip():
                return self._strip_shopify_size(content.strip())
        link = tree.css_first('link[rel="image_src"]')
        if link is not None and link.attributes:
            href = link.attributes.get("href")
            if href and href.strip():
                return self._strip_shopify_size(href.strip())
        return None

    def _poster_candidate_from_og(
        self, tree: HTMLParser
    ) -> tuple[str, int, int] | None:
        og = tree.css_first('meta[property="og:image"]')
        if og is None or not og.attributes:
            return None
        src = (og.attributes.get("content") or "").strip()
        if not src:
            return None
        w_node = tree.css_first('meta[property="og:image:width"]')
        h_node = tree.css_first('meta[property="og:image:height"]')
        try:
            w = int((w_node.attributes.get("content") if w_node and w_node.attributes else "0") or 0)
            h = int((h_node.attributes.get("content") if h_node and h_node.attributes else "0") or 0)
        except (TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        return (src, w, h)

    @staticmethod
    def _strip_shopify_size(url: str) -> str:
        """Remove Shopify CDN size mutations from both query and path.

        Shopify ships two kinds of size variants:

        * Query-param form: ``image.jpg?crop=center&height=300&width=300`` —
          strip those params and we get the full-res original.
        * In-path form: ``image_480x480.jpg`` — Shopify also encodes the
          size as a ``_WIDTHxHEIGHT`` suffix before the extension. Newer
          BJJFanatics products (e.g. "Engaging without regrets") only ship
          the size this way in the Shopify JSON candidates, so query
          stripping leaves us with a cropped thumbnail. We remove that
          suffix too — Shopify serves the un-suffixed URL as the original.
        """
        if "cdn.shop" not in url and "cdn.shopify" not in url:
            return url
        try:
            base, _, query = url.partition("?")
            # Strip `_NNNxMMM` right before the final extension. Anchored
            # to a file-extension so we don't accidentally mutate unrelated
            # substrings inside the path.
            base = re.sub(r"_\d+x\d+(?=\.[A-Za-z0-9]+$)", "", base)
            if not query:
                return base
            keep = []
            for part in query.split("&"):
                key = part.split("=", 1)[0].lower()
                if key in {"width", "height", "crop", "pad_color"}:
                    continue
                keep.append(part)
            return base + ("?" + "&".join(keep) if keep else "")
        except Exception:
            return url

    def _poster_candidates_from_json(
        self, product_url: str
    ) -> list[tuple[str, int, int]]:
        m = re.search(r"/products/([^/?#]+)", product_url)
        if not m:
            return []
        handle = m.group(1)
        json_url = f"{_PRODUCT_BASE}{handle}.json"
        resp = _http_get_with_retry(self._client, json_url)
        if resp.status_code >= 400:
            return []
        try:
            payload = resp.json()
        except ValueError:
            return []
        product = payload.get("product") if isinstance(payload, dict) else None
        if not isinstance(product, dict):
            return []
        images = product.get("images") or []
        out: list[tuple[str, int, int]] = []
        for img in images:
            if not isinstance(img, dict):
                continue
            src = img.get("src") or ""
            w = img.get("width") or 0
            h = img.get("height") or 0
            if not isinstance(w, int) or not isinstance(h, int):
                continue
            if src and w > 0 and h > 0:
                out.append((src, w, h))
        return out

    def _poster_candidates_from_html(
        self, tree: HTMLParser
    ) -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        for img in tree.css("img"):
            attrs = img.attributes or {}
            src = attrs.get("src") or attrs.get("data-src") or ""
            if not src or "cdn.shopify.com" not in src:
                continue
            w_raw = attrs.get("width") or "0"
            h_raw = attrs.get("height") or "0"
            try:
                w, h = int(w_raw), int(h_raw)
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                out.append((src.strip(), w, h))
        return out

    def _select_portrait(
        self, candidates: list[tuple[str, int, int]]
    ) -> str | None:
        """Pick the most portrait candidate that also meets the min-width bar.

        Three tiers — preferred portrait first, relaxed portrait, then the
        biggest image of any aspect ratio as a last resort. BJJFanatics
        started shipping landscape 16:9 promo posters for newer instructionals
        (e.g. "Engaging without regrets"); forcing portrait here returned
        None and left us with a 300×300 og:image thumbnail that cropped
        the title. Accepting landscape in the final tier keeps Oracle
        usable on those products — the frontend handles non-portrait
        aspect ratios with ``object-contain``.
        """
        viable = [
            (url, w, h)
            for (url, w, h) in candidates
            if w >= self._POSTER_MIN_WIDTH and (w / h) <= self._POSTER_RATIO_MIN
        ]
        if not viable:
            viable = [
                (url, w, h)
                for (url, w, h) in candidates
                if w >= self._POSTER_MIN_WIDTH
                and (w / h) <= self._POSTER_RATIO_PREFERRED
            ]
        if viable:
            viable.sort(key=lambda t: (t[1] / t[2], -t[2]))
            return viable[0][0]

        # Final tier: no portrait candidate at all — accept any landscape
        # image as long as it's bigger than the og:image thumbnail. Pick
        # the widest one (more likely to show the full cover art).
        landscape = [
            (url, w, h)
            for (url, w, h) in candidates
            if w >= self._POSTER_MIN_WIDTH
        ]
        if not landscape:
            return None
        landscape.sort(key=lambda t: -t[1])
        return landscape[0][0]

    def _parse_volume(self, h3_node, content_node) -> OracleVolume | None:
        h3_text = h3_node.text(strip=True) or ""
        m = _VOLUME_NUM_RE.search(h3_text)
        if not m:
            logger.debug("skipping h3 without Volume N: %r", h3_text)
            return None
        number = int(m.group(1))

        rows = content_node.css("figure.table > table > tbody > tr")
        if not rows:
            # Some pages might omit <figure> wrapper.
            rows = content_node.css("table > tbody > tr")
        if not rows:
            raise HTMLChangedError(
                f"no <tr> rows for Volume {number}"
            )

        parsed: list[tuple[str, int, int | None]] = []  # (title, start_s, end_hint)
        last_total: int | None = None

        for idx, tr in enumerate(rows):
            tds = tr.css("td")
            if len(tds) < 2:
                continue
            title = _title_case((tds[0].text(strip=True) or "").strip())
            time_raw = (tds[1].text(strip=True) or "").strip()
            if not title or not time_raw:
                continue

            # Robust parsing: split on ' - ' to separate start from optional
            # end/total. _parse_time handles "0", "M:SS", "MM:SS", "H:MM:SS".
            # Examples seen across products:
            #   "0"                → start=0
            #   "1:35"             → start=95
            #   "0 - 00:57"        → start=0, end_hint=57
            #   "0:00 - 1:35"      → start=0, end_hint=95
            #   "25:10 - (29:45)"  → start=1510, total=1785 (last row)
            #   "1:05:00 - 1:09:38"→ start=3900, end_hint=4178
            #   "1:09:38"          → start=4178
            # Normalize common HTML typos: semicolon instead of colon, Unicode
            # dashes, and flexible spacing around the " - " separator so that
            # "0 -14:40" and "0- 14:40" also split correctly.
            normalized = (
                time_raw.replace(";", ":")
                .replace("–", "-")
                .replace("—", "-")
            )
            # Only treat " - " as separator when it's between two time-ish
            # tokens (avoid splitting inside a single negative time, which
            # shouldn't happen anyway).
            halves = [h.strip() for h in re.split(r"\s*-\s*", normalized, maxsplit=1)]
            end_hint: int | None = None
            try:
                start_s = _parse_time(halves[0])
                if len(halves) == 2 and halves[1]:
                    second_raw = halves[1]
                    has_parens = "(" in second_raw and ")" in second_raw
                    second_clean = second_raw.strip("()").strip()
                    second_val = _parse_time(second_clean)
                    end_hint = second_val
                    if has_parens:
                        last_total = second_val
            except ValueError as e:
                raise ProviderScrapeError(
                    f"Volume {number} row {idx}: bad time {time_raw!r} ({e})"
                ) from e
            parsed.append((title, start_s, end_hint))

        if not parsed:
            raise HTMLChangedError(
                f"Volume {number}: no chapter rows parseable"
            )

        # Monotonic repair: if any start_s < previous, the cell likely had a
        # typo (e.g. "52:59:00" parsed as H:MM:SS instead of M:SS, or a glued
        # number). Nudge to prev+1s with warning so pydantic validation passes
        # and the user can fix it in the editor.
        for k in range(1, len(parsed)):
            title_k, start_k, end_k = parsed[k]
            _, prev_start, _ = parsed[k - 1]
            if start_k < prev_start:
                logger.warning(
                    "Volume %d chapter %d %r: start_s %.0f < previous %.0f; "
                    "repairing to %.0f. Review manually in the UI.",
                    number, k, title_k, start_k, prev_start, prev_start + 1,
                )
                parsed[k] = (title_k, prev_start + 1, end_k)

        # Build chapters: prefer the explicit end_hint from the cell (range
        # format), else use next chapter's start_s, else fallback +3600s.
        _LAST_FALLBACK_S = 3600.0
        chapters: list[OracleChapter] = []
        n = len(parsed)
        for k, (title, start_s, end_hint) in enumerate(parsed):
            if end_hint is not None and end_hint > start_s:
                end_s = end_hint
            elif k < n - 1:
                end_s = parsed[k + 1][1]
            else:
                if last_total is not None and last_total > start_s:
                    end_s = last_total
                else:
                    end_s = start_s + _LAST_FALLBACK_S
                    logger.warning(
                        "Volume %d: last chapter has no end hint; using "
                        "fallback end_s=%.0fs (ffmpeg will cap at EOF). "
                        "Edit manually in the UI if the cut is wrong.",
                        number, end_s,
                    )
            if end_s <= start_s:
                # Nudge by 1s so pydantic accepts it. Log warning so the user
                # knows this chapter needs manual review in the UI.
                logger.warning(
                    "Volume %d chapter %d %r: end_s %s <= start_s %s; "
                    "nudging to start+1s. Review manually in the UI.",
                    number, k, title, end_s, start_s,
                )
                end_s = start_s + 1.0
            chapters.append(
                OracleChapter(title=title, start_s=start_s, end_s=end_s)
            )

        total_duration_s = (
            last_total if last_total is not None else chapters[-1].end_s
        )

        return OracleVolume(
            number=number,
            chapters=chapters,
            total_duration_s=total_duration_s,
        )


# Auto-register on import.
from ..registry import registry  # noqa: E402

registry.register(BJJFanaticsProvider())
