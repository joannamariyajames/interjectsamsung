"""Natural-language fact extraction module.

Phase 2A: deterministic rule-based extraction for canonical travel facts
(destination, party_size, budget, cabin_class). Operates offline without
requiring external API keys.

Phase 2B: Gemini-backed structured semantic extraction as a fallback layer.
When deterministic extraction cannot confidently resolve facts from an
utterance that appears to contain travel intent, Gemini is called for
structured JSON extraction. The deterministic layer always runs first; Gemini
supplements it only when needed.

Lightweight and decoupled: produces internal ExtractedFact objects that the
runtime layer can assert into BackspaceCore as needed. Does not duplicate or
replace the BACKSPACE Fact data model.

Destination extraction uses linguistic context patterns only — no allowlist of
city, country, or airport names is used.  Any properly-named place that appears
in a recognised travel phrase (e.g. "flights to X", "trip to X", "visit X",
"hotels in X") can be extracted, making the extractor fully open-vocabulary.
A conservative stop-word guard prevents common English words (articles,
infinitive verbs, generic nouns) from being mistaken for destinations.
Cases that cannot be resolved confidently by the deterministic rules return
None, leaving them for Phase 2B (Gemini structured extraction).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Stop-word guard for destination validation
# ---------------------------------------------------------------------------
# This is a small, targeted set of common English words that must never be
# treated as destination names.  It is NOT an allowlist of destinations — it
# is a blocklist of words that are syntactically ambiguous but semantically
# unambiguous non-destinations.
_DEST_STOP: frozenset[str] = frozenset(
    {
        # Articles / determiners
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "any",
        "some",
        "my",
        "your",
        "our",
        "their",
        "its",
        "his",
        "her",
        # Pronouns
        "me",
        "you",
        "we",
        "they",
        "he",
        "she",
        "it",
        "us",
        "them",
        "him",
        # Infinitive verbs that commonly follow "to" (false-positive destinations)
        "be",
        "do",
        "go",
        "get",
        "see",
        "say",
        "try",
        "use",
        "add",
        "ask",
        "buy",
        "run",
        "eat",
        "pay",
        "set",
        "put",
        "cut",
        "let",
        "sit",
        "give",
        "fly",
        "confirm",
        "cancel",
        "check",
        "find",
        "make",
        "take",
        "change",
        "update",
        "help",
        "look",
        "know",
        "tell",
        "show",
        "plan",
        "search",
        "talk",
        "call",
        "handle",
        "discuss",
        "explain",
        "verify",
        "work",
        "start",
        "stop",
        "keep",
        "leave",
        "come",
        "send",
        "return",
        "book",
        "rebook",
        "modify",
        "contact",
        "reach",
        "connect",
        "proceed",
        "meet",
        "want",
        "need",
        "wish",
        "hope",
        "think",
        "wait",
        "rest",
        "visit",
        "stay",
        "pick",
        "drop",
        "bring",
        "carry",
        "hold",
        "read",
        "write",
        "open",
        "close",
        "join",
        "enter",
        "exit",
        "save",
        "share",
        "select",
        "choose",
        "apply",
        "review",
        "process",
        "complete",
        # Travel-domain nouns that are not destinations
        "hotel",
        "hotels",
        "airport",
        "airports",
        "booking",
        "ticket",
        "tickets",
        "agent",
        "support",
        "service",
        "desk",
        "counter",
        "terminal",
        "gate",
        "lounge",
        "home",
        "office",
        "destination",
        "origin",
        "departure",
        "arrival",
        "seat",
        "seats",
        "baggage",
        "luggage",
        "visa",
        "passport",
        "policy",
        "refund",
        "flight",
        "flights",
        "trip",
        "trips",
        "journey",
        "travel",
        # Number words (avoid colliding with party-size extraction)
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        # Cabin-class words (extracted independently)
        "business",
        "economy",
        "first",
        "premium",
    }
)


def _valid_dest(candidate: str) -> bool:
    """Return True iff *candidate* looks like a plausible destination name.

    Rules:
    - 1–3 words (covers "Goa", "New York", "Kuala Lumpur")
    - No word is in the stop-word guard
    - A single-word candidate must be at least 3 characters long
      (filters articles such as "a" and common two-letter abbreviations)
    """
    words = candidate.lower().split()
    if not words or len(words) > 3:
        return False
    if any(w in _DEST_STOP for w in words):
        return False
    if len(words) == 1 and len(words[0]) < 3:
        return False
    return True


# ---------------------------------------------------------------------------
# Number-word lookup (shared by party-size extraction)
# ---------------------------------------------------------------------------
_NUM_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class ExtractedFact:
    """Internal extraction outcome representing one typed key-value observation."""

    key: str
    value: Any
    confidence: float = 1.0


def _parse_int(raw: str) -> int:
    cleaned = raw.lower().strip()
    if cleaned in _NUM_WORDS:
        return _NUM_WORDS[cleaned]
    return int(cleaned)


# ---------------------------------------------------------------------------
# Party-size extraction (unchanged)
# ---------------------------------------------------------------------------


def _extract_party_size(text: str) -> int | None:
    # Solo / individual passenger
    if re.search(r"\b(just me|by myself|alone|solo|myself|single passenger)\b", text, re.I):
        return 1

    # "party of 5", "party size 4", "table for 2", "for 2 people"
    m = re.search(
        r"\b(?:party of|party size|table for|group of|for)\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"(?:\s*(?:people|passengers|adults|guests|persons|travelers|travellers|tickets))?\b",
        text,
        re.I,
    )
    if m:
        return _parse_int(m.group(1))

    # "3 adults", "5 people", "2 passengers"
    m = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:people|passengers|adults|guests|persons|travelers|travellers|tickets)\b",
        text,
        re.I,
    )
    if m:
        return _parse_int(m.group(1))

    # "make it 5", "change to 5"
    m = re.search(
        r"\b(?:make it|change to|increase to|actually)\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        text,
        re.I,
    )
    if m:
        return _parse_int(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Budget extraction (unchanged)
# ---------------------------------------------------------------------------


def _extract_budget(text: str) -> int | float | None:
    # Explicit budget keyword: "budget 20000", "budget of 15000", "budget: 9000"
    m = re.search(
        r"\b(?:budget|max budget|price limit)\s*(?:of|is|:)?\s*(?:rs\.?|inr|\$|usd)?\s*"
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\b",
        text,
        re.I,
    )
    if m:
        num_str = m.group(1).replace(",", "")
        return float(num_str) if "." in num_str else int(num_str)

    # Comparative constraint: "under 9000", "less than 5000", "below 8000", "within 10000"
    # Negative lookahead ensures we do NOT match durations or weights.
    m = re.search(
        r"\b(?:under|below|less than|up to|max|no more than|within|cheaper than)"
        r"\s*(?:rs\.?|inr|\$|usd)?\s*"
        r"(\d+(?:,\d{3})*(?:\.\d+)?)"
        r"(?!\s*(?:kg|kilograms?|hours?|hrs?|h\b|m\b|mins?|minutes?|percent|%|am|pm|days?|nights?))\b",
        text,
        re.I,
    )
    if m:
        num_str = m.group(1).replace(",", "")
        return float(num_str) if "." in num_str else int(num_str)

    return None


# ---------------------------------------------------------------------------
# Cabin-class extraction (unchanged)
# ---------------------------------------------------------------------------


def _extract_cabin_class(text: str) -> str | None:
    if re.search(r"\bpremium\s+economy\b", text, re.I):
        return "premium_economy"
    if re.search(r"\beconomy\s+saver\b", text, re.I):
        return "economy_saver"
    if re.search(r"\beconomy\s+flex\b", text, re.I):
        return "economy_flex"
    if re.search(r"\bbusiness(?:\s+class)?\b", text, re.I):
        return "business"
    if re.search(r"\bfirst(?:\s+class)?\b", text, re.I):
        return "first"
    if re.search(r"\beconomy(?:\s+class)?\b", text, re.I):
        return "economy"
    return None


# ---------------------------------------------------------------------------
# Destination extraction — context-pattern approach, no city allowlist
# ---------------------------------------------------------------------------

# Multi-word proper noun: each word starts with an uppercase letter (up to 3 words).
# Examples: "Goa", "New York", "Kuala Lumpur"
_DEST_PROPER = r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"

# Single word, any case, at least 3 characters.
# Used as a fallback in strong travel-verb contexts where lowercase is common
# in casual typed speech (e.g. "heading to jaipur").
_DEST_ANY = r"([a-zA-Z]{3,})"

# Travel-context signals that reliably precede "to <destination>".
# This list covers verbs and nouns that naturally take a directional complement.
_TRAVEL_VERB = (
    r"(?:fly(?:ing)?"
    r"|flights?"
    r"|trip"
    r"|journey"
    r"|holiday"
    r"|vacation"
    r"|travel(?:l?ing)?"
    r"|heading"
    r"|going"
    r"|book(?:ing)?\s+a\s+(?:flight|ticket|trip)"
    r"|depart(?:ing)?)"
)


def _extract_destination(text: str) -> str | None:
    """Extract destination using linguistic context patterns only.

    No allowlist of cities, countries, or airports is used.  Any place name
    that appears in a recognised travel phrase can be extracted regardless of
    whether it was seen during development.  Ambiguous cases that cannot be
    resolved confidently by the deterministic rules return None so that
    Phase 2B (Gemini structured extraction) can handle them.

    Pattern priority (highest confidence first):
      1. Route phrase:  "from <origin> to <dest>"
      2. Travel-context verb/noun + "to":  "flights to X", "heading to X"
      3. Visit/tour/explore without "to":  "visit X", "visiting X"
      4. Accommodation + "in" + ProperNoun:  "hotels in X"
      5. Conservative bare "to <ProperNoun>":  "to Singapore" (capital required)
    """

    # ── 1. Route phrase: "from <origin> to <dest>" ───────────────────────
    # Try multi-word proper-noun destination first (most precise).
    m = re.search(
        r"\bfrom\s+[A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)?\s+to\s+"
        + _DEST_PROPER
        + r"\b",
        text,
    )
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()
    # Fall back to single-word any-case (e.g. "from mumbai to bengaluru").
    m = re.search(
        r"\bfrom\s+[A-Za-z]{3,}(?:\s+[A-Za-z]{3,})?\s+to\s+" + _DEST_ANY + r"\b",
        text,
        re.I,
    )
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()

    # ── 2. Strong travel-context verb/noun + "to" + dest ─────────────────
    # Multi-word proper noun: "Flights to New York", "trip to Kuala Lumpur"
    m = re.search(r"\b" + _TRAVEL_VERB + r"\s+to\s+" + _DEST_PROPER + r"\b", text)
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()
    # Single-word any-case: "heading to jaipur", "fly to goa"
    m = re.search(r"\b" + _TRAVEL_VERB + r"\s+to\s+" + _DEST_ANY + r"\b", text, re.I)
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()

    # ── 3. Visit / tour / explore + dest (no "to" required) ──────────────
    # Multi-word proper noun: "visit Singapore", "visiting Nairobi"
    m = re.search(
        r"\b(?:visit(?:ing)?|tour(?:ing)?|explor(?:e|ing))\s+" + _DEST_PROPER + r"\b",
        text,
    )
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()
    # Single-word any-case
    m = re.search(
        r"\b(?:visit(?:ing)?|tour(?:ing)?|explor(?:e|ing))\s+" + _DEST_ANY + r"\b",
        text,
        re.I,
    )
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()

    # ── 4. Accommodation noun + "in" + ProperNoun ────────────────────────
    # Capital letter is required here: bare "in X" is too weak without a travel verb.
    # Multi-word proper noun: "hotels in New Delhi"
    m = re.search(
        r"\b(?:hotels?|accommodation|stay(?:ing)?|rooms?)\s+in\s+" + _DEST_PROPER + r"\b",
        text,
    )
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()
    # Single capitalized word: "hotels in Zanzibar", "hotels in Delhi"
    m = re.search(
        r"\b(?:hotels?|accommodation|stay(?:ing)?|rooms?)\s+in\s+([A-Z][a-zA-Z]{2,})\b",
        text,
    )
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()

    # ── 5. Conservative bare "to <ProperNoun>" ────────────────────────────
    # Highest precision, lowest recall: only fire when the destination is
    # explicitly title-cased by the writer (strong proper-noun signal).
    m = re.search(r"\bto\s+" + _DEST_PROPER + r"\b", text)
    if m and _valid_dest(m.group(1)):
        return m.group(1).strip().title()

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_facts(utterance: str, goal_text: str | None = None) -> list[ExtractedFact]:
    """Extract structured facts from an utterance (and optional goal context).

    Returns a list of ExtractedFact instances in canonical form:
    - destination: normalized title-case string (e.g. "Goa", "Reykjavik")
    - party_size:  integer (e.g. 5)
    - budget:      int or float (e.g. 9000)
    - cabin_class: normalized lowercase string (e.g. "business")
    """
    if not utterance or not utterance.strip():
        return []

    facts: list[ExtractedFact] = []

    dest = _extract_destination(utterance)
    if dest is not None:
        facts.append(ExtractedFact(key="destination", value=dest, confidence=0.95))

    size = _extract_party_size(utterance)
    if size is not None:
        facts.append(ExtractedFact(key="party_size", value=size, confidence=0.95))

    budget = _extract_budget(utterance)
    if budget is not None:
        facts.append(ExtractedFact(key="budget", value=budget, confidence=0.95))

    cabin = _extract_cabin_class(utterance)
    if cabin is not None:
        facts.append(ExtractedFact(key="cabin_class", value=cabin, confidence=0.95))

    return facts


# ===========================================================================
# Phase 2B — Gemini structured semantic extraction
# ===========================================================================
#
# When the deterministic extractor (Phase 2A) cannot confidently resolve all
# facts from an utterance that appears to contain travel intent, the system
# may call Gemini for structured JSON extraction.
#
# Design invariants:
#   • Phase 2A always runs first and is never skipped.
#   • Gemini supplements deterministic results — it does NOT replace them.
#   • If Gemini is unavailable or returns garbage, deterministic results
#     are used as-is; the system never crashes.
#   • No domain-specific vocabulary (city lists, airport codes, etc.) is
#     used anywhere in the Gemini layer.
#   • Output is the same ExtractedFact representation used by Phase 2A.
#   • Gemini does NOT determine change/invalidation — only the current
#     utterance is interpreted.
# ===========================================================================

_log = logging.getLogger(__name__)

# Keys that the deterministic extractor can produce.
_DETERMINISTIC_KEYS: frozenset[str] = frozenset(
    {"destination", "party_size", "budget", "cabin_class"}
)

# Keys that the Gemini extractor can produce (superset of deterministic keys,
# plus dates which only Gemini handles).
_GEMINI_KEYS: frozenset[str] = _DETERMINISTIC_KEYS | {"dates"}

# ---------------------------------------------------------------------------
# Gemini extraction prompt
# ---------------------------------------------------------------------------
# This is a general semantic instruction — no hardcoded city names, budget
# tiers, cabin-class aliases, or other domain-specific vocabulary.

_GEMINI_EXTRACTION_PROMPT = """\
You are a structured-data extraction engine for a travel assistant.

Given a user utterance, extract any travel-related facts you can confidently \
identify. Return a JSON object with ONLY the keys for which you have \
confident values. Do NOT guess or hallucinate values.

Possible keys (include only those clearly present in the utterance):
- "destination": The travel destination as a string (title-case). Resolve \
indirect references like "the Japanese capital" to the actual place name.
- "party_size": The number of travelers as an integer.
- "budget": The budget amount as a number (integer or float). Extract the \
numeric value only, without currency symbols.
- "cabin_class": One of "economy", "premium_economy", "business", "first", \
"economy_saver", "economy_flex" — normalised to lowercase with underscores.
- "dates": Travel dates as a string in a normalised form \
(e.g. "2025-03-15", "next weekend", "March 15-20"). Preserve relative \
references if an exact date cannot be determined.

Rules:
1. Return ONLY valid JSON. No markdown, no commentary, no code fences.
2. If the user expresses uncertainty ("maybe", "I'm not sure", "possibly"), \
do NOT include that fact.
3. If no travel facts are present, return an empty JSON object: {}
4. For budget, extract only the numeric amount. Interpret colloquial \
expressions like "80 grand" as 80000, "50k" as 50000, etc.
5. For party_size, interpret natural language: "just the two of us" → 2, \
"a family of six" → 6, "solo" → 1, etc.
6. For cabin_class, normalise variants: "biz class" → "business", \
"coach" → "economy", etc.
"""


def _parse_gemini_response(raw_text: str) -> list[ExtractedFact]:
    """Parse Gemini's JSON response into ExtractedFact objects.

    Tolerant of minor formatting issues (markdown fences, trailing commas).
    Returns an empty list if parsing fails entirely.
    """
    text = raw_text.strip()

    # Strip markdown code fences if Gemini wrapped its response.
    if text.startswith("```"):
        # Remove opening fence (possibly ```json)
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        _log.warning("Gemini extraction returned unparseable JSON: %.200s", text)
        return []

    if not isinstance(data, dict):
        _log.warning("Gemini extraction returned non-object JSON: %s", type(data).__name__)
        return []

    facts: list[ExtractedFact] = []

    for key in _GEMINI_KEYS:
        value = data.get(key)
        if value is None:
            continue

        # Type coercion / validation per key
        try:
            if key == "destination":
                if not isinstance(value, str) or not value.strip():
                    continue
                value = value.strip().title()

            elif key == "party_size":
                value = int(value)
                if value < 1:
                    continue

            elif key == "budget":
                if isinstance(value, str):
                    value = value.replace(",", "")
                value = float(value)
                if value <= 0:
                    continue
                # Normalise to int when there's no fractional part
                if value == int(value):
                    value = int(value)

            elif key == "cabin_class":
                if not isinstance(value, str) or not value.strip():
                    continue
                value = value.strip().lower().replace(" ", "_")

            elif key == "dates":
                if not isinstance(value, str) or not value.strip():
                    continue
                value = value.strip()

        except (ValueError, TypeError):
            _log.debug("Gemini fact %r had unconvertible value %r, skipping", key, value)
            continue

        facts.append(ExtractedFact(key=key, value=value, confidence=0.85))

    return facts


def _is_deterministic_sufficient(
    utterance: str,
    deterministic_facts: list[ExtractedFact],
) -> bool:
    """Heuristic: is the deterministic result good enough, or should we try Gemini?

    Returns True if we should use deterministic results as-is (skip Gemini).
    Returns False if Gemini should be consulted for supplementary extraction.

    The heuristic looks for signs that the utterance contains travel-related
    semantic content that the deterministic extractor may have missed.
    """
    det_keys = {f.key for f in deterministic_facts}

    # If deterministic extraction found at least one fact, it's likely adequate
    # for straightforward utterances.  But if the utterance is complex
    # (long / multi-clause) and we're missing facts that seem present, Gemini
    # might help.
    if det_keys:
        # Quick check: does the utterance contain hints of facts we missed?
        # These are very light syntactic heuristics, not domain vocabulary.
        hint_patterns: list[tuple[str, re.Pattern[str]]] = [
            ("destination", re.compile(
                r"\b(?:capital|city|town|place|country|region|island|coast|continent)\b", re.I)),
            ("party_size", re.compile(
                r"\b(?:family|couple|us|both|all\s+of\s+us|the\s+two|the\s+three)\b", re.I)),
            ("budget", re.compile(
                r"\b(?:grand|k\b|lakh|afford|cheap|expensive|affordable|pricey|costly)\b", re.I)),
            ("dates", re.compile(
                r"\b(?:weekend|month|january|february|march|april|may|june|july|august"
                r"|september|october|november|december|next\s+week|tomorrow|tonight"
                r"|next\s+month|this\s+weekend)\b", re.I)),
            ("cabin_class", re.compile(
                r"\b(?:coach|biz\s+class|upper\s+class|club\s+class)\b", re.I)),
        ]
        missed_hints = any(
            key not in det_keys and pat.search(utterance)
            for key, pat in hint_patterns
        )
        if not missed_hints:
            return True  # Deterministic result looks complete enough
        return False  # Hints suggest Gemini might find something

    # No deterministic facts at all.  Only consult Gemini if the utterance
    # looks like it might contain travel intent we couldn't parse.
    # Very conservative: if the utterance is short and doesn't contain
    # obvious travel signals, don't waste an API call.
    travel_signal = re.search(
        r"\b(?:travel|trip|flight|fly|book|hotel|stay|visit|vacation"
        r"|holiday|destination|going|heading|weekend|budget|spend"
        r"|passenger|people|family|class|seat|ticket|capital|city)\b",
        utterance,
        re.I,
    )
    return travel_signal is None  # No signal → sufficient (skip Gemini)


async def gemini_extract(
    utterance: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    _client: Any = None,
) -> list[ExtractedFact]:
    """Call Gemini for structured fact extraction.

    Uses the existing google-genai SDK (same package as GeminiProvider).
    Does NOT create a second provider instance — operates at the SDK level
    with the same api_key and model configuration.

    Parameters
    ----------
    utterance : str
        The user's raw utterance.
    api_key : str | None
        Override the Gemini API key.  Defaults to ``settings.gemini_api_key``.
    model : str | None
        Override the model name.  Defaults to ``settings.gemini_model``.
    _client : Any
        Injectable client for testing (avoids real API calls).

    Returns
    -------
    list[ExtractedFact]
        Parsed facts from Gemini's structured response, or [] on any error.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        _log.debug("google-genai not installed; skipping Gemini extraction")
        return []

    from .config import settings

    resolved_key = api_key or settings.gemini_api_key
    resolved_model = model or settings.gemini_model

    if _client is None:
        if not resolved_key:
            _log.debug("No GEMINI_API_KEY; skipping Gemini extraction")
            return []
        _client = genai.Client(api_key=resolved_key)

    prompt = (
        f"{_GEMINI_EXTRACTION_PROMPT}\n\n"
        f"User utterance: \"{utterance}\"\n\n"
        f"Respond with ONLY the JSON object."
    )

    try:
        config = types.GenerateContentConfig(
            temperature=0.0,  # Deterministic extraction — minimise creativity
        )
        response = await _client.aio.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=config,
        )
        raw_text = response.text or ""
    except Exception as exc:
        _log.warning("Gemini extraction call failed: %s", exc)
        return []

    return _parse_gemini_response(raw_text)


def _merge_extracted_facts(
    det_facts: list[ExtractedFact],
    gemini_facts: list[ExtractedFact],
    utterance: str,
) -> list[ExtractedFact]:
    """Merge deterministic and Gemini extraction outcomes safely.

    Design rules:
    1. Cross-key semantic collision resolution:
       If Gemini classifies a token under its true semantic key
       (e.g. Gemini identifies "March" as ``dates``), any deterministic
       fact that misclassified that same token under another key
       (e.g. deterministic ``destination="March"``) is recognized as a
       false positive and superseded by Gemini's semantic classification.
    2. Same-key conflict resolution:
       - If both extractors agree on the value: preserve the fact with
         the higher confidence (preferring deterministic on tie).
       - If they disagree:
         * If the deterministic value is explicitly grounded in the utterance
           while Gemini's value is not found in the utterance, Gemini's value
           is treated as an ungrounded hallucination and deterministic wins.
         * If Gemini's value is grounded in the utterance while deterministic
           was not, Gemini wins.
         * If Gemini has strictly higher confidence, Gemini wins.
         * Otherwise, deterministic fact (higher confidence) takes precedence.
    3. Supplementation:
       Facts found only by Gemini (e.g. ``dates`` or complex party sizes)
       are added to the final fact list.
    """
    if not gemini_facts:
        return list(det_facts)

    gem_by_key = {f.key: f for f in gemini_facts}
    det_by_key = {f.key: f for f in det_facts}

    # Step 1: Detect cross-key semantic collisions (e.g. dates vs destination).
    # If a deterministic fact's value is claimed by Gemini under a different key,
    # the deterministic extraction was a false-positive syntactic misclassification.
    suppressed_det_keys: set[str] = set()
    for det_key, det_fact in det_by_key.items():
        det_val_str = str(det_fact.value).strip().lower()
        if not det_val_str:
            continue
        for gem_key, gem_fact in gem_by_key.items():
            if gem_key == det_key:
                continue
            gem_val_str = str(gem_fact.value).strip().lower()
            if det_val_str == gem_val_str or det_val_str in gem_val_str:
                suppressed_det_keys.add(det_key)
                _log.debug(
                    "Deterministic fact %s=%r superseded by Gemini %s=%r (semantic reclassification)",
                    det_key, det_fact.value, gem_key, gem_fact.value,
                )
                break

    merged: list[ExtractedFact] = []
    all_keys = list(dict.fromkeys(
        [f.key for f in det_facts if f.key not in suppressed_det_keys]
        + [f.key for f in gemini_facts]
    ))

    utt_lower = utterance.lower()

    for key in all_keys:
        in_det = key in det_by_key and key not in suppressed_det_keys
        in_gem = key in gem_by_key

        if in_det and not in_gem:
            merged.append(det_by_key[key])
        elif in_gem and not in_det:
            merged.append(gem_by_key[key])
        else:
            df = det_by_key[key]
            gf = gem_by_key[key]

            df_val = str(df.value).strip().lower()
            gf_val = str(gf.value).strip().lower()

            if df_val == gf_val:
                # Same value: take higher confidence (or deterministic on tie)
                chosen = df if df.confidence >= gf.confidence else gf
                merged.append(chosen)
            else:
                # Disagreement on value:
                df_grounded = df_val in utt_lower
                gf_grounded = gf_val in utt_lower

                if df_grounded and not gf_grounded:
                    # Deterministic is directly in the text, Gemini is not -> hallucination guard
                    merged.append(df)
                elif gf_grounded and not df_grounded:
                    # Gemini is grounded in text, deterministic was not -> Gemini wins
                    merged.append(gf)
                elif gf.confidence > df.confidence:
                    # Gemini has strictly higher confidence
                    merged.append(gf)
                else:
                    # Deterministic default precedence
                    merged.append(df)

    return merged


async def extract_facts_with_fallback(
    utterance: str,
    goal_text: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    _gemini_client: Any = None,
) -> list[ExtractedFact]:
    """Extract facts using deterministic rules first, then Gemini if needed.

    This is the Phase 2B public entry point.  It orchestrates the two-tier
    extraction pipeline:

    1. Run deterministic extraction (Phase 2A) — always.
    2. Decide whether the result is sufficient.
    3. If insufficient and Gemini is available, run Gemini extraction.
    4. Merge results using safe confidence & cross-key collision resolution:
       - Cross-key semantic collisions: Gemini's semantic typing supersedes
         deterministic misclassifications (e.g. "March" as dates vs destination).
       - Same-key conflicts: grounded explicit facts win over ungrounded
         hallucinations; higher-confidence facts win on ties.
       - Missing facts: Gemini facts supplement deterministic facts.
    5. On any Gemini error, return deterministic results as-is.

    Parameters
    ----------
    utterance : str
        The user's raw utterance.
    goal_text : str | None
        Optional goal context (passed to deterministic extractor).
    api_key, model, _gemini_client
        Forwarded to ``gemini_extract`` for testing / configuration.

    Returns
    -------
    list[ExtractedFact]
        Combined facts from both extractors.
    """
    # Step 1: Always run deterministic extraction
    det_facts = extract_facts(utterance, goal_text)

    # Step 2: Check sufficiency
    if _is_deterministic_sufficient(utterance, det_facts):
        return det_facts

    # Step 3: Call Gemini for supplementary extraction
    gemini_facts = await gemini_extract(
        utterance,
        api_key=api_key,
        model=model,
        _client=_gemini_client,
    )

    if not gemini_facts:
        return det_facts

    # Step 4: Merge results safely
    return _merge_extracted_facts(det_facts, gemini_facts, utterance)

