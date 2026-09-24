"""Natural-language fact extraction module.

Phase 2A scope: deterministic rule-based extraction for canonical travel facts
(destination, party_size, budget, cabin_class). Operates offline without
requiring external API keys.

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
