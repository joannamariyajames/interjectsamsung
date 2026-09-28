from __future__ import annotations

from app.session import Session


def test_new_session_starts_with_empty_fact_ledger() -> None:
    session = Session(session_id="test-session")
    assert session.facts == {}
    assert session.get_all_facts() == {}
    assert session.has_fact("budget") is False
    assert session.get_fact("budget") is None
    assert session.get_fact("budget", 0) == 0


def test_setting_and_getting_facts() -> None:
    session = Session(session_id="test-session")
    session.set_fact("budget", 15000)
    session.set_fact("people", 5)
    session.set_fact("destination", "Goa")

    assert session.has_fact("budget") is True
    assert session.has_fact("people") is True
    assert session.has_fact("destination") is True
    assert session.has_fact("hotel") is False

    assert session.get_fact("budget") == 15000
    assert session.get_fact("people") == 5
    assert session.get_fact("destination") == "Goa"

    all_facts = session.get_all_facts()
    assert all_facts == {"budget": 15000, "people": 5, "destination": "Goa"}


def test_updating_fact_replaces_old_value() -> None:
    session = Session(session_id="test-session")
    session.set_fact("budget", 10000)
    assert session.get_fact("budget") == 10000

    session.set_fact("budget", 15000)
    assert session.get_fact("budget") == 15000


def test_get_all_facts_returns_safe_copy() -> None:
    session = Session(session_id="test-session")
    session.set_fact("budget", 15000)

    facts_copy = session.get_all_facts()
    facts_copy["budget"] = 20000
    facts_copy["new_key"] = "test"

    # Internal session facts must remain untouched
    assert session.get_fact("budget") == 15000
    assert session.has_fact("new_key") is False


def test_facts_survive_goal_switching_and_parking() -> None:
    session = Session(session_id="test-session")
    session.set_fact("budget", 15000)

    # Goal A: Hotel comparison
    goal_a_text = "We are comparing hotels."
    cls_a = session.goals.classify(goal_a_text)
    goal_a = session.goals.apply(goal_a_text, cls_a)
    assert goal_a.status.value == "active"
    assert session.get_fact("budget") == 15000

    # Goal B: Detour to refund policy (parks Goal A)
    goal_b_text = "Wait, what is the refund policy?"
    cls_b = session.goals.classify(goal_b_text)
    goal_b = session.goals.apply(goal_b_text, cls_b)
    assert goal_b.status.value == "active"
    assert goal_a.status.value == "parked"

    # Update fact during detour
    session.set_fact("budget", 12000)
    session.set_fact("refund_window_days", 7)

    # Return to Goal A (resumes Goal A, marks Goal B done)
    return_text = "anyway, back to the hotels"
    cls_return = session.goals.classify(return_text)
    goal_resumed = session.goals.apply(return_text, cls_return)
    assert goal_resumed.goal_id == goal_a.goal_id
    assert goal_resumed.status.value == "active"
    assert goal_b.status.value == "done"

    # Facts must still be intact on Session
    assert session.get_fact("budget") == 12000
    assert session.get_fact("refund_window_days") == 7
    assert session.has_fact("budget") is True


def test_session_reset_clears_facts() -> None:
    session = Session(session_id="test-session")
    session.set_fact("budget", 15000)
    session.set_fact("destination", "Goa")
    assert session.has_fact("budget") is True

    session.reset()

    assert session.facts == {}
    assert session.get_all_facts() == {}
    assert session.has_fact("budget") is False
    assert session.get_fact("budget") is None
