from __future__ import annotations

import pytest

from app.facts import extract_facts
from app.runtime import AgentRuntime
from app.session import Session
from tests.collector import Collector


def test_extract_facts_full_utterance() -> None:
    facts = extract_facts("Book Goa for two under 20k.")
    assert facts == {
        "destination": "Goa",
        "people": 2,
        "budget": 20000,
    }


def test_extract_facts_people_word() -> None:
    facts = extract_facts("Actually, we're five.")
    assert facts == {"people": 5}


def test_extract_facts_budget_shorthand() -> None:
    facts = extract_facts("The budget is now 15k.")
    assert facts == {"budget": 15000}


def test_extract_facts_destination() -> None:
    facts = extract_facts("We are going to Bangalore.")
    assert facts == {"destination": "Bangalore"}


def test_extract_facts_empty_on_unrelated_text() -> None:
    facts = extract_facts("hello there")
    assert facts == {}


@pytest.mark.parametrize(
    ("utterance", "expected_budget"),
    [
        ("under 20k", 20000),
        ("under 15000", 15000),
        ("budget of ₹15,000", 15000),
        ("up to 20,000", 20000),
        ("below 12000", 12000),
        ("within 10k", 10000),
        ("cheaper than 8000", 8000),
        ("max 25k", 25000),
    ],
)
def test_budget_variations(utterance: str, expected_budget: int) -> None:
    facts = extract_facts(utterance)
    assert facts.get("budget") == expected_budget


@pytest.mark.parametrize(
    ("utterance", "expected_people"),
    [
        ("for two", 2),
        ("we are 5", 5),
        ("5 people", 5),
        ("5 guests", 5),
        ("5 passengers", 5),
        ("table for four", 4),
        ("party of 6", 6),
    ],
)
def test_people_variations(utterance: str, expected_people: int) -> None:
    facts = extract_facts(utterance)
    assert facts.get("people") == expected_people


@pytest.mark.parametrize(
    ("utterance", "expected_destination"),
    [
        ("Book Goa", "Goa"),
        ("trip to Mumbai", "Mumbai"),
        ("flights to Delhi", "Delhi"),
        ("visiting Bengaluru", "Bengaluru"),
        ("flight from Bengaluru to Mumbai", "Mumbai"),
    ],
)
def test_destination_variations(utterance: str, expected_destination: str) -> None:
    facts = extract_facts(utterance)
    assert facts.get("destination") == expected_destination


def test_irrelevant_numbers_not_treated_as_facts() -> None:
    # Flight numbers, gate/terminal numbers, durations, options
    assert extract_facts("I have flight 240 at 5pm") == {}
    assert extract_facts("gate 4 terminal 3") == {}
    assert extract_facts("delayed by 3 hours") == {}
    assert extract_facts("under 2 hours before departure") == {}
    assert extract_facts("call the 24-hour priority desk") == {}
    assert extract_facts("checking option 2 on the screen") == {}


async def test_facts_update_across_turns_in_runtime_session() -> None:
    collector = Collector()
    session = Session(session_id="runtime-facts-test")
    runtime = AgentRuntime(session, collector)

    # Turn 1: Initial booking parameters
    await runtime.on_final("Book Goa for two under 20k.")
    await runtime._task

    assert session.get_all_facts() == {
        "destination": "Goa",
        "people": 2,
        "budget": 20000,
    }

    # Turn 2: Swerve / party size update
    await runtime.on_final("Actually, we're five.")
    await runtime._task

    assert session.get_all_facts() == {
        "destination": "Goa",
        "people": 5,
        "budget": 20000,
    }

    # Turn 3: Budget change
    await runtime.on_final("The budget is now 15k.")
    await runtime._task

    assert session.get_all_facts() == {
        "destination": "Goa",
        "people": 5,
        "budget": 15000,
    }
