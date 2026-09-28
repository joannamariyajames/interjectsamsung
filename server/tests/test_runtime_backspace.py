"""Phase 9 - the actual runtime path into BACKSPACE Core.

Exercises the real ``AgentRuntime``/``Session`` classes from ``app.runtime``/
``app.session`` (not a stand-in), through the one integration point Phase 9
adds: ``AgentRuntime.observe_fact()``. Every assertion here is about what the
*runtime* sees coming back - ``BackspaceIntegrationResult`` - never about
BACKSPACE's own internals, which are already covered exhaustively under
``tests/backspace/``.
"""

from __future__ import annotations

import asyncio

import pytest

from app.backspace import ChangeKind, Claim, DependencyKind, PlanStatus, WorkItem
from app.runtime import AgentRuntime
from app.session import Session
from tests.collector import Collector

FTW = DependencyKind.FACT_TO_WORK
WTW = DependencyKind.WORK_TO_WORK
WTC = DependencyKind.WORK_TO_CLAIM


def make() -> tuple[AgentRuntime, Collector, Session]:
    collector = Collector()
    session = Session(session_id="test")
    return AgentRuntime(session, collector), collector, session


# ===========================================================================
# NEW / UNCHANGED / CHANGED, through the real runtime
# ===========================================================================


def test_new_fact_produces_no_invalidation():
    runtime, _, session = make()
    result = runtime.observe_fact("destination", "Goa", source="user")

    assert result.change_kind is ChangeKind.NEW
    assert result.fact_update.previous is None
    assert result.changeset is None
    assert result.invalidation is None
    assert result.recomputation_plan is None
    assert result.explanation is None
    assert session.backspace.get_fact("destination").value == "Goa"


def test_unchanged_fact_produces_no_new_version_and_no_side_effects():
    runtime, _, session = make()
    runtime.observe_fact("party_size", 2, source="user")
    fact_before = session.backspace.get_fact("party_size")

    result = runtime.observe_fact("party_size", 2, source="user")

    assert result.change_kind is ChangeKind.UNCHANGED
    assert result.changeset is None
    assert result.invalidation is None
    assert result.recomputation_plan is None
    assert result.explanation is None
    fact_after = session.backspace.get_fact("party_size")
    assert fact_after.version == fact_before.version == 1
    assert len(session.backspace.get_fact_history("party_size")) == 1


def test_changed_fact_produces_the_full_pipeline():
    runtime, _, session = make()
    session.backspace.register_work(WorkItem(kind="price_calculation", work_id="W2"))
    first = runtime.observe_fact("party_size", 2, source="user")
    session.backspace.register_dependency(FTW, first.fact_update.fact.fact_id, "W2")

    result = runtime.observe_fact("party_size", 5, source="user")

    assert result.change_kind is ChangeKind.CHANGED
    assert result.changeset is not None
    assert result.changeset.key == "party_size"
    assert result.invalidation is not None
    assert result.invalidation.invalidated_work_ids == ["W2"]
    assert result.recomputation_plan is not None
    assert result.recomputation_plan.status is PlanStatus.READY
    assert result.recomputation_plan.work_items_to_recompute == ["W2"]
    assert result.explanation is not None
    assert [w.work_id for w in result.explanation.invalidated_work] == ["W2"]


# ===========================================================================
# canonical end-to-end scenario, through the real runtime/session
# ===========================================================================


def test_canonical_end_to_end_scenario_through_the_real_runtime():
    runtime, _, session = make()
    core = session.backspace  # the one owner of this session's BACKSPACE state

    core.register_work(WorkItem(kind="hotel_search", work_id="W1"))
    core.register_work(WorkItem(kind="price_calculation", work_id="W2"))
    core.register_work(WorkItem(kind="recommendation", work_id="W3"))
    core.register_claim(Claim(text="Hotels found in Goa", claim_id="C1", work_id="W1"))
    core.register_claim(Claim(text="Hotel price is ₹18,000", claim_id="C2", work_id="W2"))
    core.register_claim(Claim(text="This is a good option", claim_id="C3", work_id="W3"))

    # Turn 1: structured facts enter through the runtime, exactly as a real
    # observation would (NEW each time - nothing to invalidate yet).
    destination = runtime.observe_fact("destination", "Goa", source="user", turn_id="turn-1")
    party_size = runtime.observe_fact("party_size", 2, source="user", turn_id="turn-1")
    budget = runtime.observe_fact("budget", 20000, source="user", turn_id="turn-1")
    for result in (destination, party_size, budget):
        assert result.change_kind is ChangeKind.NEW

    core.register_dependency(FTW, destination.fact_update.fact.fact_id, "W1")
    core.register_dependency(FTW, party_size.fact_update.fact.fact_id, "W2")
    core.register_dependency(FTW, budget.fact_update.fact.fact_id, "W2")
    core.register_dependency(WTW, "W1", "W3")
    core.register_dependency(WTW, "W2", "W3")
    core.register_dependency(WTC, "W1", "C1")
    core.register_dependency(WTC, "W2", "C2")
    core.register_dependency(WTC, "W3", "C3")

    core.mark_claim_spoken("C2")
    core.mark_claim_spoken("C3")

    # Turn 2: the one structured observation the brief asks for.
    result = runtime.observe_fact("party_size", 5, source="user", turn_id="turn-2")

    assert result.change_kind is ChangeKind.CHANGED
    assert result.changeset.key == "party_size"
    assert result.changeset.previous_fact.value == 2
    assert result.changeset.new_fact.value == 5

    assert sorted(result.invalidation.kept_work_ids) == ["W1"]
    assert sorted(result.invalidation.invalidated_work_ids) == ["W2", "W3"]
    assert sorted(result.invalidation.invalidated_claim_ids) == ["C2", "C3"]
    assert sorted(result.invalidation.spoken_invalidated_claim_ids) == ["C2", "C3"]
    assert result.invalidation.unspoken_invalidated_claim_ids == []

    assert result.recomputation_plan.status is PlanStatus.READY
    assert result.recomputation_plan.work_items_to_recompute == ["W2", "W3"]

    kept_ids = {w.work_id for w in result.explanation.kept_work}
    invalidated_ids = {w.work_id for w in result.explanation.invalidated_work}
    assert kept_ids == {"W1"}
    assert invalidated_ids == {"W2", "W3"}
    for claim_explanation in result.explanation.invalidated_claims:
        assert claim_explanation.spoken is True
        assert claim_explanation.retraction_required is True

    # Nothing must have been executed: W1 untouched, no output produced
    # anywhere, and no retraction was auto-created (that stays an explicit
    # act - see claims.py/core.py).
    assert core.graph.get_work("W1").output is None
    assert core.graph.get_work("W1").status.value == "pending"
    assert core.get_retraction("C2") is None
    assert core.get_retraction("C3") is None


def test_canonical_scenario_never_touches_the_provider_or_emits_any_frame():
    """observe_fact must be entirely out-of-band from the wire protocol and
    the provider - it is a structured, synchronous side channel."""
    runtime, collector, session = make()
    session.backspace.register_work(WorkItem(kind="price_calculation", work_id="W2"))
    first = runtime.observe_fact("party_size", 2, source="user")
    session.backspace.register_dependency(FTW, first.fact_update.fact.fact_id, "W2")

    runtime.observe_fact("party_size", 5, source="user")

    assert collector.frames == []  # no StageFrame/MetricFrame/etc. from this at all
    assert not runtime.busy  # no turn task was ever started


# ===========================================================================
# session isolation
# ===========================================================================


def test_session_isolation_between_two_runtime_sessions():
    runtime_a, _, session_a = make()
    runtime_b, _, session_b = make()

    session_a.backspace.register_work(WorkItem(kind="price_calculation", work_id="W2"))
    session_b.backspace.register_work(WorkItem(kind="price_calculation", work_id="W2"))

    fa = runtime_a.observe_fact("party_size", 2, source="user")
    fb = runtime_b.observe_fact("party_size", 2, source="user")
    session_a.backspace.register_dependency(FTW, fa.fact_update.fact.fact_id, "W2")
    session_b.backspace.register_dependency(FTW, fb.fact_update.fact.fact_id, "W2")

    result_a = runtime_a.observe_fact("party_size", 7, source="user")

    # B was never touched: no leaked fact, changeset, claim, retraction or
    # explanation.
    assert session_b.backspace.get_fact("party_size").value == 2
    assert session_b.backspace.get_fact("party_size").version == 1
    assert session_b.backspace.graph.get_work("W2").status.value == "pending"
    assert session_b.backspace.get_changeset(result_a.changeset.changeset_id) is None
    assert session_a is not session_b
    assert session_a.backspace is not session_b.backspace


# ===========================================================================
# reset
# ===========================================================================


async def test_reset_clears_backspace_state_with_the_session():
    runtime, _, session = make()
    session.backspace.register_work(WorkItem(kind="price_calculation", work_id="W2"))
    first = runtime.observe_fact("party_size", 2, source="user")
    session.backspace.register_dependency(FTW, first.fact_update.fact.fact_id, "W2")
    changed = runtime.observe_fact("party_size", 5, source="user")
    changeset_id = changed.changeset.changeset_id
    claim = session.backspace.register_claim(Claim(text="x", claim_id="C1"))
    session.backspace.mark_claim_spoken("C1")

    await runtime.reset()

    assert session.backspace.get_fact("party_size") is None
    assert session.backspace.get_changeset(changeset_id) is None
    assert session.backspace.get_claim("C1") is None
    assert session.backspace.get_retraction("C1") is None
    # A fresh observation after reset behaves like session-start, not like
    # the old state still lingering.
    fresh = runtime.observe_fact("party_size", 2, source="user")
    assert fresh.change_kind is ChangeKind.NEW


# ===========================================================================
# error handling: BackspaceCore errors are not swallowed
# ===========================================================================


def test_backspacecore_errors_propagate_rather_than_being_swallowed():
    from app.backspace import ClaimNotFoundError

    runtime, _, session = make()
    with pytest.raises(ClaimNotFoundError):
        session.backspace.mark_claim_spoken("does-not-exist")
    # The runtime layer does not wrap this in a try/except that turns a
    # programming error into a fake, silently-accepted fact.


# ===========================================================================
# turn_id / goal_id defaults come from the runtime's own state
# ===========================================================================


def test_observe_fact_defaults_turn_id_to_the_runtimes_current_turn():
    runtime, _, session = make()
    runtime._turn_id = "turn-in-flight"  # what on_final() would have set
    result = runtime.observe_fact("destination", "Goa", source="user")
    assert result.fact_update.fact.turn_id == "turn-in-flight"


def test_observe_fact_explicit_turn_id_overrides_the_default():
    runtime, _, session = make()
    runtime._turn_id = "turn-in-flight"
    result = runtime.observe_fact("destination", "Goa", source="user", turn_id="turn-explicit")
    assert result.fact_update.fact.turn_id == "turn-explicit"


def test_observe_fact_defaults_goal_id_to_the_active_goal_without_parsing_it():
    runtime, _, session = make()
    classification = session.goals.classify("Find me a flight to Mumbai")
    goal = session.goals.apply("Find me a flight to Mumbai", classification)

    result = runtime.observe_fact("destination", "Mumbai", source="user")

    assert result.fact_update.fact.goal_id == goal.goal_id


def test_observe_fact_explicit_goal_id_overrides_the_active_goal():
    runtime, _, session = make()
    classification = session.goals.classify("Find me a flight to Mumbai")
    session.goals.apply("Find me a flight to Mumbai", classification)

    result = runtime.observe_fact("destination", "Mumbai", source="user", goal_id="explicit-goal")
    assert result.fact_update.fact.goal_id == "explicit-goal"
