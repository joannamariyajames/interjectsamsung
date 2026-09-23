"""Deterministic extraction of session-scoped facts from user utterances.

Pure functions with zero external API/LLM dependencies.
Initially supports:
- budget: integer amounts with currency/k-suffix normalization
- people: party size from word numbers or digits
- destination: recognized travel destinations
"""

from __future__ import annotations

import re
from typing import Any

# Word numbers supported for people count (one through ten)
_WORD_NUMBERS: dict[str, int] = {
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

_NUM_PATTERN = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"

# Time units and non-budget units that should not be parsed as budgets
_NON_BUDGET_UNITS = re.compile(
    r"^(?:hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|percent|%|am|pm)\b",
    re.IGNORECASE,
)

# Budget cues and currency markers
_BUDGET_CUES = re.compile(
    r"\b(?:budget\s*(?:is|of|now|limit)?\s*(?:is|now)?|"
    r"under|below|max|maximum|up\s+to|within|cheaper\s+than)\s*"
    r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\b|"
    r"\b(?:₹|rs\.?|inr)\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\b|"
    r"\b(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:₹|rs\.?|inr|rupees?)\b",
    re.IGNORECASE,
)

# People / party size cues
_PEOPLE_WE_ARE = re.compile(
    rf"\bwe(?:'re|\s+are)\s+({_NUM_PATTERN})\b",
    re.IGNORECASE,
)
_PEOPLE_NOUN = re.compile(
    rf"\b({_NUM_PATTERN})\s+(?:people|guests|passengers|pax|persons|travellers|travelers|adults)\b",
    re.IGNORECASE,
)
_PEOPLE_FOR = re.compile(
    rf"\b(?:for|party\s+of|table\s+for)\s+({_NUM_PATTERN})\b",
    re.IGNORECASE,
)

# Units that disqualify a "for <number>" from being a party size
_NON_PEOPLE_FOLLOWING = re.compile(
    r"^(?:hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|nights?|"
    r"inr|rs\.?|rupees?|₹|k|lakh|lac|thousand|hundred|fee|window)\b",
    re.IGNORECASE,
)

# Canonical travel destinations relevant to existing corpus
_CITY_MAP: dict[str, str] = {
    "goa": "Goa",
    "bangalore": "Bangalore",
    "bengaluru": "Bengaluru",
    "mumbai": "Mumbai",
    "delhi": "Delhi",
}

_CITY_REGEX = re.compile(
    r"\b(goa|bangalore|bengaluru|mumbai|delhi)\b",
    re.IGNORECASE,
)

_FROM_TO_REGEX = re.compile(
    r"\bfrom\s+[a-z]+\s+to\s+(goa|bangalore|bengaluru|mumbai|delhi)\b",
    re.IGNORECASE,
)

_TO_CITY_REGEX = re.compile(
    r"\b(?:to|in|for|visiting|going\s+to|trip\s+to|book)\s+(goa|bangalore|bengaluru|mumbai|delhi)\b",
    re.IGNORECASE,
)


def _parse_int_or_word(raw: str) -> int | None:
    cleaned = raw.strip().lower()
    if cleaned in _WORD_NUMBERS:
        return _WORD_NUMBERS[cleaned]
    try:
        return int(cleaned)
    except ValueError:
        return None


def _extract_budget(text: str) -> int | None:
    for m in _BUDGET_CUES.finditer(text):
        raw_val = m.group(1) or m.group(3) or m.group(5)
        k_suffix = m.group(2) or m.group(4) or m.group(6)
        if not raw_val:
            continue

        # Check what immediately follows the match to avoid matching "under 2 hours"
        end_pos = m.end()
        remaining = text[end_pos:].strip()
        if _NON_BUDGET_UNITS.match(remaining):
            continue

        num_str = raw_val.replace(",", "")
        try:
            val = float(num_str)
        except ValueError:
            continue

        if k_suffix:
            val *= 1000

        return int(val) if val.is_integer() else int(round(val))
    return None


def _extract_people(text: str) -> int | None:
    # 1. "we're 5" / "we are five"
    m_we = _PEOPLE_WE_ARE.search(text)
    if m_we:
        val = _parse_int_or_word(m_we.group(1))
        if val is not None and 1 <= val <= 100:
            return val

    # 2. "5 people" / "5 guests" / "two passengers"
    m_noun = _PEOPLE_NOUN.search(text)
    if m_noun:
        val = _parse_int_or_word(m_noun.group(1))
        if val is not None and 1 <= val <= 100:
            return val

    # 3. "for two" / "party of 4"
    for m_for in _PEOPLE_FOR.finditer(text):
        raw = m_for.group(1)
        end_pos = m_for.end()
        remaining = text[end_pos:].strip()
        if _NON_PEOPLE_FOLLOWING.match(remaining):
            continue

        val = _parse_int_or_word(raw)
        # Avoid treating large numbers (e.g. "for 2400") as people
        if val is not None and 1 <= val <= 20:
            return val

    return None


def _extract_destination(text: str) -> str | None:
    # Check for "from X to Y" destination
    m_from_to = _FROM_TO_REGEX.search(text)
    if m_from_to:
        city_key = m_from_to.group(1).lower()
        return _CITY_MAP.get(city_key)

    # Check for "to X" / "book X" / "trip to X"
    m_to = _TO_CITY_REGEX.search(text)
    if m_to:
        city_key = m_to.group(1).lower()
        return _CITY_MAP.get(city_key)

    # Check for general single city mention
    matches = _CITY_REGEX.findall(text)
    if len(matches) == 1:
        return _CITY_MAP.get(matches[0].lower())
    if len(matches) > 1:
        # If multiple cities mentioned, prefer the last one as destination
        return _CITY_MAP.get(matches[-1].lower())

    return None


def extract_facts(text: str) -> dict[str, Any]:
    """Deterministically extract structured facts (budget, people, destination) from text."""
    facts: dict[str, Any] = {}

    destination = _extract_destination(text)
    if destination is not None:
        facts["destination"] = destination

    people = _extract_people(text)
    if people is not None:
        facts["people"] = people

    budget = _extract_budget(text)
    if budget is not None:
        facts["budget"] = budget

    return facts
