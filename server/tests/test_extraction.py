"""Unit tests for natural-language fact extraction (app.extraction)."""

from __future__ import annotations

import pytest
from app.extraction import ExtractedFact, extract_facts


# ---------------------------------------------------------------------------
# Party-size extraction
# ---------------------------------------------------------------------------


def test_extract_party_size_numeric_and_words():
    # "for 2 people"
    facts = {f.key: f.value for f in extract_facts("Find flights for 2 people")}
    assert facts.get("party_size") == 2

    # "party of 5"
    facts = {f.key: f.value for f in extract_facts("Looking for hotel rooms, party of 5")}
    assert facts.get("party_size") == 5

    # "3 adults"
    facts = {f.key: f.value for f in extract_facts("Need tickets for 3 adults")}
    assert facts.get("party_size") == 3

    # word numbers: "table for two"
    facts = {f.key: f.value for f in extract_facts("Booking a stay for two")}
    assert facts.get("party_size") == 2

    # "4 passengers"
    facts = {f.key: f.value for f in extract_facts("Flight for 4 passengers")}
    assert facts.get("party_size") == 4


def test_extract_party_size_just_me_and_solo():
    for phrase in ["just me", "by myself", "alone", "traveling solo", "single passenger"]:
        facts = {f.key: f.value for f in extract_facts(f"I am booking for {phrase}")}
        assert facts.get("party_size") == 1, f"Failed for phrase: {phrase}"


# ---------------------------------------------------------------------------
# Destination extraction — linguistic context patterns (no allowlist)
# ---------------------------------------------------------------------------


def test_extract_destination_travel_context_patterns():
    """Destinations extracted by syntactic position, NOT by city lookup."""
    # "Fly to Goa" — Pattern 5 (bare "to <ProperNoun>")
    facts = {f.key: f.value for f in extract_facts("Fly to Goa")}
    assert facts.get("destination") == "Goa"

    # "Planning to visit Mumbai" — Pattern 3 (visit)
    facts = {f.key: f.value for f in extract_facts("Planning to visit Mumbai")}
    assert facts.get("destination") == "Mumbai"

    # "Show me hotels in Delhi" — Pattern 4 (accommodation + in)
    facts = {f.key: f.value for f in extract_facts("Show me hotels in Delhi")}
    assert facts.get("destination") == "Delhi"

    # "Flights from Mumbai to Bengaluru" — Pattern 1 (route)
    facts = {f.key: f.value for f in extract_facts("Flights from Mumbai to Bengaluru")}
    assert facts.get("destination") == "Bengaluru"

    # "heading to jaipur" — Pattern 2 (travel verb, lowercase city)
    facts = {f.key: f.value for f in extract_facts("heading to jaipur")}
    assert facts.get("destination") == "Jaipur"


def test_destination_arbitrary_without_predefined_vocabulary():
    """Destinations NOT in any predefined vocabulary are extracted correctly
    by linguistic context pattern alone.  Uses several unrelated cities that
    were never listed in the codebase.
    """
    cases = [
        # Strong travel-verb + "to" (Pattern 2 multi-word proper noun)
        ("Flights to Reykjavik please", "Reykjavik"),
        ("Planning a trip to Nairobi this summer", "Nairobi"),
        ("Can I fly to Tbilisi?", "Tbilisi"),
        ("Book a flight to Ulaanbaatar", "Ulaanbaatar"),
        ("traveling to Vilnius next week", "Vilnius"),
        # Accommodation + "in" (Pattern 4)
        ("Looking for hotels in Zanzibar", "Zanzibar"),
        # Visit without "to" (Pattern 3)
        ("I am visiting Kathmandu", "Kathmandu"),
        # Bare "to ProperNoun" (Pattern 5)
        ("I need seats to Reykjavik", "Reykjavik"),
    ]
    for utterance, expected in cases:
        facts = {f.key: f.value for f in extract_facts(utterance)}
        assert facts.get("destination") == expected, (
            f"Expected destination={expected!r} from {utterance!r}, "
            f"got {facts.get('destination')!r}"
        )


def test_no_allowlist_attribute_exists():
    """Verify that KNOWN_DESTINATIONS has been removed from the module."""
    import app.extraction as m

    assert not hasattr(m, "KNOWN_DESTINATIONS"), (
        "KNOWN_DESTINATIONS must not exist — destination extraction must be allowlist-free"
    )


# ---------------------------------------------------------------------------
# Budget extraction
# ---------------------------------------------------------------------------


def test_extract_budget_comparative_and_explicit():
    # "under 9000"
    facts = {f.key: f.value for f in extract_facts("Hotels under 9000")}
    assert facts.get("budget") == 9000

    # "budget 20000"
    facts = {f.key: f.value for f in extract_facts("We have a budget 20000 for the trip")}
    assert facts.get("budget") == 20000

    # "less than 5000"
    facts = {f.key: f.value for f in extract_facts("Looking for fares less than 5000")}
    assert facts.get("budget") == 5000

    # "below 8,500"
    facts = {f.key: f.value for f in extract_facts("Rooms below 8,500 INR")}
    assert facts.get("budget") == 8500

    # "up to 12000"
    facts = {f.key: f.value for f in extract_facts("Can spend up to 12000")}
    assert facts.get("budget") == 12000


def test_budget_does_not_match_durations_or_weights():
    # "under 2 hours" is duration, not budget
    facts = {f.key: f.value for f in extract_facts("Flight block time under 2 hours")}
    assert "budget" not in facts

    # "15 kg" is weight, not budget
    facts = {f.key: f.value for f in extract_facts("Baggage allowance under 15 kg")}
    assert "budget" not in facts

    # "inside 72 hours" is time window
    facts = {f.key: f.value for f in extract_facts("Can I cancel within 72 hours?")}
    assert "budget" not in facts


# ---------------------------------------------------------------------------
# Cabin-class extraction
# ---------------------------------------------------------------------------


def test_extract_cabin_class():
    # business class
    facts = {f.key: f.value for f in extract_facts("Book in business class")}
    assert facts.get("cabin_class") == "business"

    # economy
    facts = {f.key: f.value for f in extract_facts("What are the rules for economy?")}
    assert facts.get("cabin_class") == "economy"

    # premium economy
    facts = {f.key: f.value for f in extract_facts("Check seats in premium economy")}
    assert facts.get("cabin_class") == "premium_economy"

    # economy saver
    facts = {f.key: f.value for f in extract_facts("Is economy saver refundable?")}
    assert facts.get("cabin_class") == "economy_saver"

    # first class
    facts = {f.key: f.value for f in extract_facts("Looking for first class seats")}
    assert facts.get("cabin_class") == "first"


# ---------------------------------------------------------------------------
# Multi-fact utterance
# ---------------------------------------------------------------------------


def test_multiple_facts_in_one_utterance():
    utterance = "Find flights to Goa for 2 people in business class under 9000"
    extracted = extract_facts(utterance)
    facts = {f.key: f.value for f in extracted}

    assert facts.get("destination") == "Goa"
    assert facts.get("party_size") == 2
    assert facts.get("cabin_class") == "business"
    assert facts.get("budget") == 9000

    # Verify types
    assert isinstance(facts["destination"], str)
    assert isinstance(facts["party_size"], int)
    assert isinstance(facts["cabin_class"], str)
    assert isinstance(facts["budget"], (int, float))


# ---------------------------------------------------------------------------
# Update-phrasing variations
# ---------------------------------------------------------------------------


def test_wording_variations_for_updates():
    # "actually make it 5 people"
    facts = {f.key: f.value for f in extract_facts("Actually make it 5 people")}
    assert facts.get("party_size") == 5

    # "change to 4 adults"
    facts = {f.key: f.value for f in extract_facts("Change to 4 adults")}
    assert facts.get("party_size") == 4


# ---------------------------------------------------------------------------
# Empty / unrelated utterances
# ---------------------------------------------------------------------------


def test_no_facts_in_unrelated_or_empty_text():
    assert extract_facts("") == []
    assert extract_facts("   ") == []
    assert extract_facts("Hello there, how are you?") == []
    assert extract_facts("What is the weather like?") == []
    assert extract_facts("Can you explain how this demo works?") == []


def test_no_false_extraction_from_arbitrary_nouns():
    # Common verbs after "to" must not be extracted as destinations
    facts = {f.key: f.value for f in extract_facts("I want to confirm the booking")}
    assert "destination" not in facts

    facts = {f.key: f.value for f in extract_facts("Can I talk to an agent?")}
    assert "destination" not in facts

    facts = {f.key: f.value for f in extract_facts("What is the refund policy?")}
    assert "destination" not in facts
