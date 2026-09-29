from __future__ import annotations

from app.goals import GoalTracker
from app.schemas import GoalAction


def run(utterances: list[str]) -> tuple[GoalTracker, list[GoalAction]]:
    tracker = GoalTracker()
    actions = []
    for text in utterances:
        classification = tracker.classify(text)
        tracker.apply(text, classification)
        actions.append(classification.action)
    return tracker, actions


def test_first_utterance_pushes_a_goal():
    tracker, actions = run(["Find me a flight to Mumbai on Friday"])
    assert actions == [GoalAction.PUSH]
    assert tracker.active is not None
    assert tracker.active.status.value == "active"


def test_terse_amendment_refines_instead_of_switching():
    """The regression that matters: 'make it under 6000' shares no nouns with
    the goal it amends, so a naive overlap rule reads it as a new goal."""
    tracker, actions = run(
        ["Find me a flight from Bengaluru to Mumbai on Friday", "make it under 6000 rupees"]
    )
    assert actions[1] is GoalAction.REFINE
    assert len(tracker.stack) == 1
    assert tracker.active is not None
    assert "under 6000" in tracker.active.constraints


def test_explicit_switch_parks_rather_than_drops_the_goal():
    tracker, actions = run(
        ["Find me a flight to Mumbai", "actually, what is the refund policy if I cancel?"]
    )
    assert actions[1] is GoalAction.SWITCH
    statuses = [g.status.value for g in tracker.stack]
    assert "parked" in statuses, "the original goal must survive the swerve"
    assert len(tracker.stack) == 2


def test_revert_resumes_the_parked_goal_with_its_constraints():
    tracker, actions = run(
        [
            "Find me a flight from Bengaluru to Mumbai on Friday",
            "make it under 6000 rupees",
            "actually, what is the refund policy if I cancel?",
            "anyway, back to the flight",
        ]
    )
    assert actions[3] is GoalAction.REVERT
    active = tracker.active
    assert active is not None
    assert "flight" in active.text.lower()
    assert "under 6000" in active.constraints, "constraints survive a detour"


def test_agreeing_to_the_agents_offer_continues_the_goal():
    """From a live session: "yes please tell the rules and details" answered the
    agent's question about the flight, but became a goal of its own, so the
    'Still open' chip later offered to resume that sentence instead of the trip."""
    tracker, actions = run([
        "I want to book a flight from Hyderabad to Mumbai via airline Air India",
        "yes please tell the rules and details",
        "I want to search for an apartment in Mathikere Bangalore",
    ])
    assert actions[1] is GoalAction.CONTINUE
    assert actions[2] is GoalAction.SWITCH
    parked = [g for g in tracker.stack if g.status.value == "parked"]
    assert [g.text for g in parked] == ["I want to book a flight from Hyderabad to Mumbai via airline Air India"]


def test_an_affirmation_does_not_override_an_explicit_switch_or_return():
    _, actions = run(["Find me a flight to Mumbai on Friday", "okay actually what about hotels in Delhi"])
    assert actions[1] is GoalAction.SWITCH
    _, actions = run(["Find me a flight to Mumbai on Friday", "what about hotels in Delhi instead",
                      "okay back to the flight"])
    assert actions[2] is GoalAction.REVERT


def test_a_long_new_request_starting_with_yes_is_not_forced_into_the_old_goal():
    _, actions = run(["Find me a flight to Mumbai on Friday",
                      "yes and separately compare apartment rents deposits brokerage and commute times across Pune suburbs"])
    assert actions[1] is not GoalAction.CONTINUE
