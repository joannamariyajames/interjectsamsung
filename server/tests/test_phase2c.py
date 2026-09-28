"""Phase 2C integration tests: Extraction -> Session -> BACKSPACE Core.

Verifies:
A. Session creates an independent BackspaceCore instance.
B. Session.reset() resets Backspace state.
C. Realistic utterance flows through runtime extraction into BackspaceCore.
D. Changed facts flow through BackspaceCore versioning without manual invalidation.
E. Non-travel/non-fact utterances do not create extraneous facts.
F. Open-ended values (destinations not in demo corpus) work without allowlists.
G. Cancellation safety during a turn.
"""

from __future__ import annotations

import asyncio
import pytest

from app.backspace import ChangeKind, FactStatus
from app.runtime import AgentRuntime
from app.session import Session
from tests.collector import Collector


def _make_runtime(session_id: str = "phase2c-test") -> tuple[AgentRuntime, Collector, Session]:
    collector = Collector()
    session = Session(session_id=session_id)
    runtime = AgentRuntime(session, collector)
    return runtime, collector, session


# ---------------------------------------------------------------------------
# Requirement A: Session creates an independent BackspaceCore
# ---------------------------------------------------------------------------


def test_session_creates_independent_backspace_core():
    session1 = Session(session_id="s1")
    session2 = Session(session_id="s2")

    assert session1.backspace is not None
    assert session2.backspace is not None
    assert session1.backspace is not session2.backspace

    # Assert fact in session1; session2 must remain empty
    session1.backspace.assert_fact(
        key="destination",
        value="Kyoto",
        source="test",
        turn_id="t1",
    )

    assert session1.backspace.get_fact("destination") is not None
    assert session1.backspace.get_fact("destination").value == "Kyoto"
    assert session2.backspace.get_fact("destination") is None


# ---------------------------------------------------------------------------
# Requirement B: Session.reset() resets Backspace state
# ---------------------------------------------------------------------------


def test_session_reset_resets_backspace_state():
    session = Session(session_id="reset-test")
    session.backspace.assert_fact(
        key="party_size",
        value=4,
        source="test",
        turn_id="t1",
    )
    assert session.backspace.get_fact("party_size") is not None

    session.reset()

    assert session.backspace.get_fact("party_size") is None
    assert session.backspace.snapshot()["facts"] == {}


# ---------------------------------------------------------------------------
# Requirement C: Realistic utterance flows through runtime into BackspaceCore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_realistic_utterance_flows_through_runtime_into_backspace():
    runtime, collector, session = _make_runtime()

    utterance = "Find hotels in Kyoto for 5 people"
    await runtime.on_final(utterance)
    assert runtime._task is not None
    await runtime._task

    # Verify facts asserted in session.backspace
    dest_fact = session.backspace.get_fact("destination")
    assert dest_fact is not None
    assert dest_fact.value == "Kyoto"
    assert dest_fact.source == "extraction"
    assert dest_fact.turn_id == runtime._turn_id
    assert dest_fact.status == FactStatus.CURRENT
    assert dest_fact.goal_id is not None

    size_fact = session.backspace.get_fact("party_size")
    assert size_fact is not None
    assert size_fact.value == 5
    assert size_fact.source == "extraction"
    assert size_fact.turn_id == runtime._turn_id
    assert size_fact.goal_id == dest_fact.goal_id


# ---------------------------------------------------------------------------
# Requirement D: Changed fact flows through Core versioning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_changed_fact_flows_through_core_versioning():
    runtime, collector, session = _make_runtime()

    # Turn 1: 2 people
    await runtime.on_final("Find hotels in Kyoto for 2 people")
    assert runtime._task is not None
    await runtime._task

    fact_v1 = session.backspace.get_fact("party_size")
    assert fact_v1 is not None
    assert fact_v1.value == 2
    assert fact_v1.version == 1
    assert fact_v1.status == FactStatus.CURRENT
    v1_id = fact_v1.fact_id

    # Turn 2: change to 5 people
    await runtime.on_final("Actually make it 5 people")
    assert runtime._task is not None
    await runtime._task

    fact_v2 = session.backspace.get_fact("party_size")
    assert fact_v2 is not None
    assert fact_v2.value == 5
    assert fact_v2.version == 2
    assert fact_v2.status == FactStatus.CURRENT
    assert fact_v2.supersedes == v1_id

    # Verify history in notebook
    history = session.backspace.get_fact_history("party_size")
    assert len(history) == 2
    assert history[0].value == 2
    assert history[0].status == FactStatus.SUPERSEDED
    assert history[0].superseded_by == fact_v2.fact_id
    assert history[1].value == 5
    assert history[1].status == FactStatus.CURRENT

    # Destination should still be preserved
    dest_fact = session.backspace.get_fact("destination")
    assert dest_fact is not None
    assert dest_fact.value == "Kyoto"


# ---------------------------------------------------------------------------
# Requirement E: Non-travel / non-fact utterance does not create facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_travel_utterance_does_not_create_facts():
    runtime, collector, session = _make_runtime()

    await runtime.on_final("What are the baggage limits?")
    assert runtime._task is not None
    await runtime._task

    snapshot = session.backspace.snapshot()
    assert snapshot["facts"] == {}, "Non-travel question should not assert facts into BackspaceCore"


# ---------------------------------------------------------------------------
# Requirement F: Open-ended values (destinations not in demo corpus)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_ended_unseen_destination_and_constraints():
    runtime, collector, session = _make_runtime()

    # "Reykjavik" is not in flights.md or hotels.md demo corpus
    utterance = "Flights to Reykjavik for 3 people under 15000 in business class"
    await runtime.on_final(utterance)
    assert runtime._task is not None
    await runtime._task

    dest = session.backspace.get_fact("destination")
    assert dest is not None
    assert dest.value == "Reykjavik"

    size = session.backspace.get_fact("party_size")
    assert size is not None
    assert size.value == 3

    budget = session.backspace.get_fact("budget")
    assert budget is not None
    assert budget.value == 15000

    cabin = session.backspace.get_fact("cabin_class")
    assert cabin is not None
    assert cabin.value == "business"


# ---------------------------------------------------------------------------
# Requirement G: Cancellation safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_cancellation_preserves_safety_and_checkpoints():
    runtime, collector, session = _make_runtime()

    await runtime.on_final("Find flights to Goa for 2 people")
    # Wait until streaming starts
    while not collector.of("token"):
        await asyncio.sleep(0.01)

    # Interrupt
    await runtime.interrupt("barge_in")

    assert not runtime.busy
    assert session.checkpoint is not None
    # Facts asserted before streaming remain valid in core
    dest = session.backspace.get_fact("destination")
    assert dest is not None
    assert dest.value == "Goa"
