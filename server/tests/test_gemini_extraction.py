"""Unit tests for Phase 2B: Gemini structured fact extraction.

All tests mock Gemini API calls. No real GEMINI_API_KEY is needed.

Test coverage:
1. Destination from arbitrary/unseen destination (via Gemini)
2. Party size (semantic)
3. Budget (colloquial expressions)
4. Dates (Gemini-only key)
5. Cabin class (semantic variants)
6. Multiple facts in one utterance
7. Ambiguous / uncertain / missing facts
8. Malformed Gemini output
9. Gemini unavailable / error
10. Deterministic fallback still works
11. No hardcoded destination vocabulary
12. Merge behavior (deterministic priority)
13. Sufficiency heuristic
14. Response parser edge cases
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.extraction import (
    ExtractedFact,
    _is_deterministic_sufficient,
    _parse_gemini_response,
    extract_facts,
    extract_facts_with_fallback,
    gemini_extract,
)


# ---------------------------------------------------------------------------
# Helpers: mock Gemini client
# ---------------------------------------------------------------------------


class FakeGeminiResponse:
    """Simulates the response object from genai.Client.aio.models.generate_content."""

    def __init__(self, text: str) -> None:
        self.text = text


def _make_fake_client(response_text: str) -> Any:
    """Build a fake genai.Client whose .aio.models.generate_content returns *response_text*."""
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=FakeGeminiResponse(response_text)
    )
    return client


def _make_failing_client(error: Exception) -> Any:
    """Build a fake genai.Client whose .aio.models.generate_content raises *error*."""
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=error)
    return client


# ===========================================================================
# 1. Destination extraction from arbitrary / unseen destinations (via Gemini)
# ===========================================================================


async def test_gemini_extracts_arbitrary_destination():
    """Gemini should extract destinations that are not in any predefined list."""
    response = json.dumps({"destination": "Ulaanbaatar"})
    client = _make_fake_client(response)

    facts = await gemini_extract("I want to visit the Mongolian capital", _client=client)
    fact_map = {f.key: f.value for f in facts}

    assert fact_map.get("destination") == "Ulaanbaatar"


async def test_gemini_extracts_novel_unseen_destinations():
    """Multiple unseen destinations that no allowlist would contain."""
    cases = [
        (json.dumps({"destination": "Antananarivo"}), "Antananarivo"),
        (json.dumps({"destination": "Ouagadougou"}), "Ouagadougou"),
        (json.dumps({"destination": "Bishkek"}), "Bishkek"),
        (json.dumps({"destination": "Thimphu"}), "Thimphu"),
        (json.dumps({"destination": "Nuuk"}), "Nuuk"),
    ]
    for response_text, expected in cases:
        client = _make_fake_client(response_text)
        facts = await gemini_extract("Trip to somewhere", _client=client)
        fact_map = {f.key: f.value for f in facts}
        assert fact_map.get("destination") == expected, (
            f"Expected {expected!r}, got {fact_map.get('destination')!r}"
        )


# ===========================================================================
# 2. Party size extraction (semantic)
# ===========================================================================


async def test_gemini_extracts_party_size():
    response = json.dumps({"party_size": 5})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "I've changed my mind, make it five people instead.", _client=client
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("party_size") == 5


async def test_gemini_extracts_party_size_from_family():
    response = json.dumps({"party_size": 6})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "Can you find something affordable for a family of six?", _client=client
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("party_size") == 6


async def test_gemini_extracts_party_size_couple():
    response = json.dumps({"party_size": 2})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "Let's make this a weekend trip for just the two of us.", _client=client
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("party_size") == 2


# ===========================================================================
# 3. Budget extraction (colloquial)
# ===========================================================================


async def test_gemini_extracts_budget_grand():
    response = json.dumps({"budget": 80000})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "I don't want to spend more than 80 grand.", _client=client
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("budget") == 80000


async def test_gemini_extracts_budget_k_notation():
    response = json.dumps({"budget": 50000})
    client = _make_fake_client(response)

    facts = await gemini_extract("Keep it under 50k", _client=client)
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("budget") == 50000


async def test_gemini_extracts_float_budget():
    response = json.dumps({"budget": 1500.50})
    client = _make_fake_client(response)

    facts = await gemini_extract("Budget is about $1500.50", _client=client)
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("budget") == 1500.50


# ===========================================================================
# 4. Dates extraction (Gemini-only key)
# ===========================================================================


async def test_gemini_extracts_dates():
    response = json.dumps({"dates": "next weekend"})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "Let's make this a weekend trip for just the two of us.", _client=client
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("dates") == "next weekend"


async def test_gemini_extracts_specific_dates():
    response = json.dumps({"dates": "March 15-20"})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "We're planning to go from March 15 to March 20.", _client=client
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("dates") == "March 15-20"


# ===========================================================================
# 5. Cabin class extraction (semantic variants)
# ===========================================================================


async def test_gemini_extracts_cabin_class_business():
    response = json.dumps({"cabin_class": "business"})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "Actually, we're flying business class.", _client=client
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("cabin_class") == "business"


async def test_gemini_normalises_cabin_class():
    """Gemini might return 'premium economy'; parser should normalise."""
    response = json.dumps({"cabin_class": "premium economy"})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "Premium economy would be nice.", _client=client
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("cabin_class") == "premium_economy"


# ===========================================================================
# 6. Multiple facts in one utterance (via Gemini)
# ===========================================================================


async def test_gemini_extracts_multiple_facts():
    response = json.dumps({
        "destination": "Reykjavik",
        "party_size": 2,
        "budget": 80000,
        "cabin_class": "business",
        "dates": "next weekend",
    })
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "Forget the previous destination, we're going to Reykjavik, "
        "business class, two of us, budget 80 grand, next weekend.",
        _client=client,
    )
    fact_map = {f.key: f.value for f in facts}

    assert fact_map.get("destination") == "Reykjavik"
    assert fact_map.get("party_size") == 2
    assert fact_map.get("budget") == 80000
    assert fact_map.get("cabin_class") == "business"
    assert fact_map.get("dates") == "next weekend"

    # All Gemini facts have 0.85 confidence
    for f in facts:
        assert f.confidence == 0.85


# ===========================================================================
# 7. Ambiguous / uncertain / missing facts
# ===========================================================================


async def test_gemini_returns_empty_for_uncertain():
    """If the user is uncertain, Gemini should not include the fact."""
    response = json.dumps({})
    client = _make_fake_client(response)

    facts = await gemini_extract(
        "Maybe Tokyo, I'm not sure.", _client=client
    )
    assert facts == []


async def test_gemini_returns_empty_for_no_travel_content():
    response = json.dumps({})
    client = _make_fake_client(response)

    facts = await gemini_extract("How's the weather today?", _client=client)
    assert facts == []


async def test_gemini_skips_null_values():
    """Null values in the Gemini response should be ignored."""
    response = json.dumps({
        "destination": "Paris",
        "party_size": None,
        "budget": None,
    })
    client = _make_fake_client(response)

    facts = await gemini_extract("Going to Paris", _client=client)
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("destination") == "Paris"
    assert "party_size" not in fact_map
    assert "budget" not in fact_map


# ===========================================================================
# 8. Malformed Gemini output
# ===========================================================================


async def test_gemini_handles_invalid_json():
    client = _make_fake_client("This is not JSON at all")
    facts = await gemini_extract("Going somewhere", _client=client)
    assert facts == []


async def test_gemini_handles_json_array_instead_of_object():
    client = _make_fake_client('[{"destination": "Paris"}]')
    facts = await gemini_extract("Going to Paris", _client=client)
    assert facts == []


async def test_gemini_handles_markdown_wrapped_json():
    """Gemini sometimes wraps JSON in markdown fences."""
    response = '```json\n{"destination": "Lisbon", "party_size": 3}\n```'
    client = _make_fake_client(response)

    facts = await gemini_extract("Trip to Lisbon with 3 people", _client=client)
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("destination") == "Lisbon"
    assert fact_map.get("party_size") == 3


async def test_gemini_handles_empty_string_response():
    client = _make_fake_client("")
    facts = await gemini_extract("Going somewhere", _client=client)
    assert facts == []


async def test_gemini_handles_none_text_response():
    """If Gemini response.text is None."""
    client = MagicMock()
    response = MagicMock()
    response.text = None
    client.aio.models.generate_content = AsyncMock(return_value=response)

    facts = await gemini_extract("Going somewhere", _client=client)
    assert facts == []


async def test_gemini_handles_invalid_party_size():
    """Negative or zero party_size should be ignored."""
    response = json.dumps({"party_size": -1, "destination": "Lima"})
    client = _make_fake_client(response)

    facts = await gemini_extract("Trip to Lima", _client=client)
    fact_map = {f.key: f.value for f in facts}
    assert "party_size" not in fact_map
    assert fact_map.get("destination") == "Lima"


async def test_gemini_handles_invalid_budget():
    """Zero or negative budget should be ignored."""
    response = json.dumps({"budget": 0, "destination": "Lima"})
    client = _make_fake_client(response)

    facts = await gemini_extract("Trip to Lima", _client=client)
    fact_map = {f.key: f.value for f in facts}
    assert "budget" not in fact_map


async def test_gemini_handles_non_string_destination():
    """Destination must be a string."""
    response = json.dumps({"destination": 12345})
    client = _make_fake_client(response)

    facts = await gemini_extract("Going somewhere", _client=client)
    fact_map = {f.key: f.value for f in facts}
    assert "destination" not in fact_map


async def test_gemini_handles_unknown_keys_gracefully():
    """Extra keys in Gemini response should be ignored."""
    response = json.dumps({
        "destination": "Berlin",
        "meal_preference": "vegetarian",
        "airline": "Lufthansa",
    })
    client = _make_fake_client(response)

    facts = await gemini_extract("Going to Berlin", _client=client)
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("destination") == "Berlin"
    assert "meal_preference" not in fact_map
    assert "airline" not in fact_map


# ===========================================================================
# 9. Gemini unavailable / error
# ===========================================================================


async def test_gemini_returns_empty_on_api_error():
    client = _make_failing_client(RuntimeError("Rate limit exceeded"))
    facts = await gemini_extract("Going to Paris", _client=client)
    assert facts == []


async def test_gemini_returns_empty_on_timeout():
    import asyncio

    client = _make_failing_client(asyncio.TimeoutError("Timeout"))
    facts = await gemini_extract("Going to Paris", _client=client)
    assert facts == []


async def test_gemini_returns_empty_on_connection_error():
    client = _make_failing_client(ConnectionError("No internet"))
    facts = await gemini_extract("Going to Paris", _client=client)
    assert facts == []


async def test_gemini_extract_no_api_key():
    """When no API key is configured and no client is injected, return []."""
    with patch("app.config.settings", MagicMock(gemini_api_key=None, gemini_model="gemini-2.0-flash")):
        facts = await gemini_extract("Going to Paris")
    assert facts == []


# ===========================================================================
# 10. Deterministic fallback still works (Phase 2A untouched)
# ===========================================================================


def test_deterministic_extraction_unchanged():
    """Phase 2A deterministic extraction must still work exactly as before."""
    facts = {f.key: f.value for f in extract_facts("Fly to Goa for 2 people under 9000 in business class")}
    assert facts.get("destination") == "Goa"
    assert facts.get("party_size") == 2
    assert facts.get("budget") == 9000
    assert facts.get("cabin_class") == "business"


def test_deterministic_empty_utterance():
    assert extract_facts("") == []
    assert extract_facts("   ") == []


async def test_fallback_uses_deterministic_when_gemini_fails():
    """If Gemini fails, extract_facts_with_fallback returns deterministic results."""
    failing_client = _make_failing_client(RuntimeError("Gemini down"))

    facts = await extract_facts_with_fallback(
        # This utterance has a travel signal ("capital") that will trigger Gemini attempt,
        # but deterministic can't parse it, and Gemini fails → empty result.
        "We're looking for something in the Japanese capital.",
        _gemini_client=failing_client,
    )
    # The deterministic extractor can't parse "Japanese capital", and Gemini
    # failed, so we get whatever deterministic found (likely nothing for destination).
    # But importantly: no crash.
    assert isinstance(facts, list)
    for f in facts:
        assert isinstance(f, ExtractedFact)


async def test_fallback_deterministic_sufficient_skips_gemini():
    """When deterministic extraction is sufficient, Gemini should NOT be called."""
    # Build a client that would fail if called
    spy_client = _make_failing_client(AssertionError("Should not be called"))

    facts = await extract_facts_with_fallback(
        "Flights to Paris for 2 people in business class under 9000",
        _gemini_client=spy_client,
    )
    fact_map = {f.key: f.value for f in facts}
    # All facts come from deterministic (confidence 0.95)
    assert fact_map.get("destination") == "Paris"
    assert fact_map.get("party_size") == 2
    assert fact_map.get("budget") == 9000
    assert fact_map.get("cabin_class") == "business"
    for f in facts:
        assert f.confidence == 0.95  # Deterministic confidence, not Gemini's 0.85


# ===========================================================================
# 11. No hardcoded destination vocabulary
# ===========================================================================


async def test_no_allowlist_gemini_extracts_any_destination():
    """Gemini extraction does not depend on a predefined destination vocabulary."""
    import app.extraction as m

    # Verify no destination-related allowlists exist
    for attr_name in dir(m):
        attr = getattr(m, attr_name)
        if isinstance(attr, (set, frozenset, list, tuple, dict)):
            # _DEST_STOP is a stop-word blocklist (allowed)
            # _NUM_WORDS is a number-word map (allowed)
            # _DETERMINISTIC_KEYS / _GEMINI_KEYS are structural key sets (allowed)
            if attr_name in ("_DEST_STOP", "_NUM_WORDS", "_DETERMINISTIC_KEYS", "_GEMINI_KEYS"):
                continue
            # Check no attr contains city/destination names
            stringified = str(attr).lower()
            for city in ["goa", "mumbai", "delhi", "tokyo", "paris", "london",
                         "reykjavik", "nairobi", "ulaanbaatar"]:
                assert city not in stringified, (
                    f"Production attribute {attr_name!r} contains city name {city!r}: "
                    f"this violates the no-hardcoding requirement"
                )


def test_no_known_destinations_attribute():
    """Verify that KNOWN_DESTINATIONS / similar allowlists don't exist."""
    import app.extraction as m

    forbidden = [
        "KNOWN_DESTINATIONS", "KNOWN_CITIES", "DESTINATION_LIST",
        "AIRPORT_LIST", "CITY_NAMES", "LOCATION_VOCABULARY",
        "TRAVEL_LOCATIONS", "KNOWN_PLACES", "LOCATION_MAP",
        "SUPPORTED_CITIES", "VALID_DESTINATIONS", "DESTINATION_TERMS",
    ]
    for name in forbidden:
        assert not hasattr(m, name), (
            f"{name} must not exist — extraction must be allowlist-free"
        )


# ===========================================================================
# 12. Merge behavior (deterministic priority, false positives, hallucinations)
# ===========================================================================


async def test_merge_deterministic_takes_priority():
    """Deterministic facts override Gemini facts for the same key when values agree."""
    gemini_response = json.dumps({
        "destination": "Tokyo",
        "dates": "next weekend",
    })
    client = _make_fake_client(gemini_response)

    facts = await extract_facts_with_fallback(
        "flights to Tokyo this weekend",
        _gemini_client=client,
    )
    fact_map = {f.key: f.value for f in facts}

    assert fact_map.get("destination") == "Tokyo"
    assert fact_map.get("dates") == "next weekend"

    dest_fact = next(f for f in facts if f.key == "destination")
    assert dest_fact.confidence == 0.95

    dates_fact = next(f for f in facts if f.key == "dates")
    assert dates_fact.confidence == 0.85


async def test_merge_deterministic_plus_gemini_missing_fact():
    """Requirement 1: Deterministic extracts destination, Gemini supplies missing dates."""
    gemini_response = json.dumps({
        "destination": "Reykjavik",
        "dates": "next weekend",
    })
    client = _make_fake_client(gemini_response)

    facts = await extract_facts_with_fallback(
        "Flights to Reykjavik next weekend",
        _gemini_client=client,
    )
    fact_map = {f.key: f.value for f in facts}

    assert fact_map.get("destination") == "Reykjavik"
    assert fact_map.get("dates") == "next weekend"


async def test_merge_deterministic_plus_gemini_same_fact():
    """Requirement 2: Both extractors identify the same fact value."""
    gemini_response = json.dumps({
        "destination": "Reykjavik",
        "party_size": 2,
    })
    client = _make_fake_client(gemini_response)

    facts = await extract_facts_with_fallback(
        "Flights to Reykjavik for 2 people",
        _gemini_client=client,
    )
    fact_map = {f.key: f.value for f in facts}

    assert fact_map.get("destination") == "Reykjavik"
    assert fact_map.get("party_size") == 2
    # Verify higher deterministic confidence is preserved
    for f in facts:
        assert f.confidence == 0.95


async def test_merge_deterministic_false_positive_resolved_by_gemini():
    """Requirement 3: Deterministic false-positive destination is corrected by Gemini.

    Utterance: 'Fly to March with 2 people'
    Deterministic regex misclassifies 'March' as destination (from 'Fly to <ProperNoun>').
    Gemini semantically understands that 'March' is a date, returning dates='March', party_size=2.
    The false positive destination='March' must be dropped.
    """
    gemini_response = json.dumps({
        "dates": "March",
        "party_size": 2,
    })
    client = _make_fake_client(gemini_response)

    facts = await extract_facts_with_fallback(
        "Fly to March with 2 people",
        _gemini_client=client,
    )
    fact_map = {f.key: f.value for f in facts}

    # The false positive destination must NOT be present
    assert "destination" not in fact_map
    assert fact_map.get("dates") == "March"
    assert fact_map.get("party_size") == 2


async def test_merge_gemini_hallucination_cannot_override_explicit_fact():
    """Requirement 4: Ungrounded Gemini hallucination cannot override explicit user fact.

    Utterance explicitly says 'Fly to Goa for 2 people this weekend'.
    Deterministic extracts destination='Goa'.
    Gemini hallucinates destination='Mumbai' (which does not appear in the utterance).
    Deterministic 'Goa' must win.
    """
    gemini_response = json.dumps({
        "destination": "Mumbai",  # Hallucination! Not in utterance
        "dates": "this weekend",
        "party_size": 2,
    })
    client = _make_fake_client(gemini_response)

    facts = await extract_facts_with_fallback(
        "Fly to Goa for 2 people this weekend",
        _gemini_client=client,
    )
    fact_map = {f.key: f.value for f in facts}

    assert fact_map.get("destination") == "Goa"
    assert fact_map.get("dates") == "this weekend"
    assert fact_map.get("party_size") == 2


async def test_merge_gemini_supplements_deterministic():
    """Gemini adds facts that deterministic missed entirely."""
    gemini_response = json.dumps({
        "destination": "Tokyo",
        "party_size": 2,
        "budget": 80000,
        "dates": "March 15-20",
    })
    client = _make_fake_client(gemini_response)

    facts = await extract_facts_with_fallback(
        "We're looking for something in the Japanese capital for a family trip",
        _gemini_client=client,
    )
    fact_map = {f.key: f.value for f in facts}

    assert fact_map.get("destination") == "Tokyo"
    assert fact_map.get("dates") == "March 15-20"


def test_dates_contract_compatibility_with_backspace_core():
    """Requirement 5: Dates fact key and value format are compatible with BACKSPACE Core Fact."""
    from app.backspace.facts import Fact

    extracted = ExtractedFact(key="dates", value="2025-06-15", confidence=0.85)

    # Convert to Core Fact representation
    core_fact = Fact(
        key=extracted.key,
        value=extracted.value,
        confidence=extracted.confidence,
        source="gemini_extraction",
    )

    assert core_fact.key == "dates"
    assert core_fact.value == "2025-06-15"
    assert core_fact.confidence == 0.85
    assert core_fact.source == "gemini_extraction"


# ===========================================================================
# 13. Sufficiency heuristic
# ===========================================================================


def test_sufficiency_no_facts_no_signal():
    """No deterministic facts + no travel signal → sufficient (don't call Gemini)."""
    assert _is_deterministic_sufficient("Hello, how are you?", []) is True


def test_sufficiency_no_facts_with_signal():
    """No facts + travel signal → insufficient (call Gemini)."""
    assert _is_deterministic_sufficient(
        "I need a flight to the capital city", []
    ) is False


def test_sufficiency_facts_complete():
    """Deterministic found facts, no missed hints → sufficient."""
    facts = [
        ExtractedFact(key="destination", value="Paris", confidence=0.95),
        ExtractedFact(key="party_size", value=2, confidence=0.95),
    ]
    assert _is_deterministic_sufficient(
        "Flights to Paris for 2 people", facts
    ) is True


def test_sufficiency_facts_with_missed_date_hint():
    """Deterministic found destination, but utterance mentions 'weekend' → insufficient."""
    facts = [
        ExtractedFact(key="destination", value="Paris", confidence=0.95),
    ]
    assert _is_deterministic_sufficient(
        "Flights to Paris this weekend", facts
    ) is False


def test_sufficiency_facts_with_missed_budget_hint():
    """Deterministic found destination, but 'affordable' hints at budget → insufficient."""
    facts = [
        ExtractedFact(key="destination", value="Paris", confidence=0.95),
    ]
    assert _is_deterministic_sufficient(
        "Flights to Paris, something affordable", facts
    ) is False


def test_sufficiency_facts_with_missed_capital_hint():
    """'capital' hints at a destination Gemini should resolve."""
    assert _is_deterministic_sufficient(
        "I want to visit the capital", []
    ) is False


# ===========================================================================
# 14. Response parser edge cases
# ===========================================================================


def test_parse_gemini_response_valid_json():
    facts = _parse_gemini_response('{"destination": "Rome", "party_size": 4}')
    fact_map = {f.key: f.value for f in facts}
    assert fact_map["destination"] == "Rome"
    assert fact_map["party_size"] == 4


def test_parse_gemini_response_markdown_fences():
    raw = '```json\n{"destination": "Rome"}\n```'
    facts = _parse_gemini_response(raw)
    assert any(f.key == "destination" and f.value == "Rome" for f in facts)


def test_parse_gemini_response_empty_json():
    facts = _parse_gemini_response("{}")
    assert facts == []


def test_parse_gemini_response_garbage():
    facts = _parse_gemini_response("Absolutely! Here is your answer...")
    assert facts == []


def test_parse_gemini_response_all_keys():
    raw = json.dumps({
        "destination": "Kyoto",
        "party_size": 3,
        "budget": 5000,
        "cabin_class": "first",
        "dates": "2025-06-01",
    })
    facts = _parse_gemini_response(raw)
    keys = {f.key for f in facts}
    assert keys == {"destination", "party_size", "budget", "cabin_class", "dates"}


def test_parse_gemini_response_string_budget_with_commas():
    """Budget can come as a string with commas — parser should handle."""
    raw = json.dumps({"budget": "1,500,000"})
    facts = _parse_gemini_response(raw)
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("budget") == 1500000


def test_parse_gemini_response_confidence_is_085():
    """All Gemini-parsed facts must have confidence 0.85."""
    raw = json.dumps({"destination": "Lagos"})
    facts = _parse_gemini_response(raw)
    assert all(f.confidence == 0.85 for f in facts)


# ===========================================================================
# 15. Gemini output maps correctly to ExtractedFact
# ===========================================================================


async def test_gemini_facts_are_extractedfact_instances():
    response = json.dumps({"destination": "Valletta", "party_size": 2})
    client = _make_fake_client(response)

    facts = await gemini_extract("Trip to Valletta for 2", _client=client)
    for f in facts:
        assert isinstance(f, ExtractedFact)
        assert hasattr(f, "key")
        assert hasattr(f, "value")
        assert hasattr(f, "confidence")


async def test_gemini_facts_have_correct_types():
    response = json.dumps({
        "destination": "Zagreb",
        "party_size": 4,
        "budget": 3000,
        "cabin_class": "economy",
        "dates": "December 20-25",
    })
    client = _make_fake_client(response)

    facts = await gemini_extract("Trip details", _client=client)
    fact_map = {f.key: f.value for f in facts}

    assert isinstance(fact_map["destination"], str)
    assert isinstance(fact_map["party_size"], int)
    assert isinstance(fact_map["budget"], (int, float))
    assert isinstance(fact_map["cabin_class"], str)
    assert isinstance(fact_map["dates"], str)


# ===========================================================================
# 16. Adversarial hard-coding check
# ===========================================================================


async def test_adversarial_unseen_destinations():
    """Destinations that were NEVER used during development should work."""
    unseen = [
        "Phnom Penh", "Tashkent", "Ashgabat", "Maputo", "Port Moresby",
        "Dushanbe", "Belmopan", "Sucre", "Ngerulmud", "Funafuti",
    ]
    for city in unseen:
        response = json.dumps({"destination": city})
        client = _make_fake_client(response)
        facts = await gemini_extract(f"Travel to {city}", _client=client)
        fact_map = {f.key: f.value for f in facts}
        assert fact_map.get("destination") == city.title(), (
            f"Failed for unseen destination: {city}"
        )


async def test_adversarial_unusual_budget_expressions():
    """Budget expressions that are not standard."""
    cases = [
        (json.dumps({"budget": 250000}), 250000),  # "2.5 lakh"
        (json.dumps({"budget": 75000}), 75000),     # "75 grand"
        (json.dumps({"budget": 999.99}), 999.99),   # fractional
        (json.dumps({"budget": 1}), 1),             # minimal
    ]
    for response_text, expected in cases:
        client = _make_fake_client(response_text)
        facts = await gemini_extract("Some budget query", _client=client)
        fact_map = {f.key: f.value for f in facts}
        assert fact_map.get("budget") == expected


async def test_adversarial_cabin_class_variants():
    """Cabin class wording that the deterministic extractor wouldn't catch."""
    cases = [
        (json.dumps({"cabin_class": "business"}), "business"),
        (json.dumps({"cabin_class": "economy"}), "economy"),
        (json.dumps({"cabin_class": "premium economy"}), "premium_economy"),
        (json.dumps({"cabin_class": "first"}), "first"),
    ]
    for response_text, expected in cases:
        client = _make_fake_client(response_text)
        facts = await gemini_extract("Some class query", _client=client)
        fact_map = {f.key: f.value for f in facts}
        assert fact_map.get("cabin_class") == expected


# ===========================================================================
# 17. End-to-end: full pipeline with fallback
# ===========================================================================


async def test_full_pipeline_semantic_destination():
    """Full pipeline: deterministic can't parse → Gemini resolves."""
    gemini_response = json.dumps({"destination": "Tokyo"})
    client = _make_fake_client(gemini_response)

    facts = await extract_facts_with_fallback(
        "We're looking for something in the Japanese capital.",
        _gemini_client=client,
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("destination") == "Tokyo"


async def test_full_pipeline_deterministic_only():
    """Full pipeline: deterministic handles everything, Gemini not called."""
    # The spy client would blow up if called
    spy_client = _make_failing_client(AssertionError("Gemini should not be called"))

    facts = await extract_facts_with_fallback(
        "Fly to Goa for 3 adults in economy under 5000",
        _gemini_client=spy_client,
    )
    fact_map = {f.key: f.value for f in facts}
    assert fact_map.get("destination") == "Goa"
    assert fact_map.get("party_size") == 3
    assert fact_map.get("budget") == 5000
    assert fact_map.get("cabin_class") == "economy"


async def test_full_pipeline_empty_utterance():
    """Empty utterance returns no facts, no Gemini call."""
    facts = await extract_facts_with_fallback("")
    assert facts == []


async def test_full_pipeline_unrelated_utterance():
    """Non-travel utterance returns no facts, no Gemini call."""
    spy_client = _make_failing_client(AssertionError("Should not be called"))
    facts = await extract_facts_with_fallback(
        "What's the weather like?",
        _gemini_client=spy_client,
    )
    assert facts == []
