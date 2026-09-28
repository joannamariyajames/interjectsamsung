"""Integration tests for BACKSPACE Tool WorkItem execution and Harness stale-work gate.

Verifies:
1. unchanged fact → no invalidation.
2. changed fact → previous fact's dependent WorkItem becomes STALE.
3. stale WorkItem → underlying tool function is NOT called.
4. invalidated WorkItem → underlying tool function is NOT called.
5. unaffected WorkItem remains executable.
6. successful valid WorkItem records its output and transitions to VALID.
7. arbitrary/non-travel fact changes work identically.
8. runtime turn execution wires WorkItem lifecycle and invalidation end-to-end.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.backspace import (
    BackspaceCore,
    ChangeKind,
    DependencyKind,
    FactStatus,
    WorkItem,
    WorkStatus,
)
from app.harness import Harness, ToolOutcome
from app.runtime import AgentRuntime
from app.session import Session
from tests.collector import Collector


# ---------------------------------------------------------------------------
# 1. Unchanged fact → no invalidation
# ---------------------------------------------------------------------------
def test_unchanged_fact_no_invalidation():
    core = BackspaceCore()
    u1 = core.assert_fact("cluster_id", "prod-east", source="config", turn_id="t1")
    assert u1.status == ChangeKind.NEW

    work = WorkItem(
        kind="metrics_query",
        turn_id="t1",
        depends_on_facts=[u1.fact.fact_id],
    )
    core.register_work(work)
    core.register_dependency(DependencyKind.FACT_TO_WORK, u1.fact.fact_id, work.work_id)

    # Re-asserting identical value is UNCHANGED
    u2 = core.assert_fact("cluster_id", "prod-east", source="config", turn_id="t2")
    assert u2.status == ChangeKind.UNCHANGED
    assert u2.changeset is None

    # WorkItem status remains unaffected
    assert work.status == WorkStatus.PENDING


# ---------------------------------------------------------------------------
# 2. Changed fact → previous fact's dependent WorkItem becomes STALE
# ---------------------------------------------------------------------------
def test_changed_fact_invalidates_dependent_work_item():
    core = BackspaceCore()
    u1 = core.assert_fact("database_host", "db-v1.internal", source="env", turn_id="t1")
    fact_v1_id = u1.fact.fact_id

    work = WorkItem(
        kind="db_query",
        turn_id="t1",
        depends_on_facts=[fact_v1_id],
    )
    core.register_work(work)
    core.register_dependency(DependencyKind.FACT_TO_WORK, fact_v1_id, work.work_id)

    # Change fact
    u2 = core.assert_fact("database_host", "db-v2.internal", source="env", turn_id="t2")
    assert u2.status == ChangeKind.CHANGED
    assert u2.changeset is not None

    # Invalidation bridge
    core.invalidate(u2.changeset)

    # Check that work item became STALE
    registered_work = core.graph.get_work(work.work_id)
    assert registered_work is not None
    assert registered_work.status == WorkStatus.STALE


# ---------------------------------------------------------------------------
# 3. Stale WorkItem → underlying tool function is NOT called
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stale_work_item_blocks_execution():
    harness = Harness(strict=True)
    budget = harness.new_budget()

    work = WorkItem(
        kind="price_quote",
        status=WorkStatus.STALE,
        turn_id="t1",
    )

    with patch("app.tools._price_quote", new_callable=AsyncMock) as mock_tool:
        outcome = await harness.call(
            "price_quote",
            budget,
            work=work,
            route="DEL-BOM",
            cabin="economy",
        )

        assert outcome.status == "blocked"
        assert "stale" in outcome.verdict
        mock_tool.assert_not_called()
        # Budget used count should not increase for blocked stale call before admission
        assert budget.used == 0


# ---------------------------------------------------------------------------
# 4. Invalidated WorkItem → underlying tool function is NOT called
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalidated_work_item_blocks_execution():
    harness = Harness(strict=True)
    budget = harness.new_budget()

    work = WorkItem(
        kind="search_corpus",
        status=WorkStatus.INVALIDATED,
        turn_id="t1",
    )

    with patch("app.tools._search_corpus", new_callable=AsyncMock) as mock_tool:
        outcome = await harness.call(
            "search_corpus",
            budget,
            work=work,
            query="test query",
        )

        assert outcome.status == "blocked"
        assert "invalidated" in outcome.verdict
        mock_tool.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Unaffected WorkItem remains executable
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unaffected_work_item_remains_executable():
    core = BackspaceCore()
    harness = Harness(strict=True)
    budget = harness.new_budget()

    u_lang = core.assert_fact("language", "python", source="user", turn_id="t1")
    u_framework = core.assert_fact("framework", "fastapi", source="user", turn_id="t1")

    # Work 1 depends on language
    w1 = WorkItem(kind="price_quote", turn_id="t1", depends_on_facts=[u_lang.fact.fact_id])
    core.register_work(w1)
    core.register_dependency(DependencyKind.FACT_TO_WORK, u_lang.fact.fact_id, w1.work_id)

    # Work 2 depends on framework
    w2 = WorkItem(kind="price_quote", turn_id="t1", depends_on_facts=[u_framework.fact.fact_id])
    core.register_work(w2)
    core.register_dependency(DependencyKind.FACT_TO_WORK, u_framework.fact.fact_id, w2.work_id)

    # Change only language
    u_lang_2 = core.assert_fact("language", "rust", source="user", turn_id="t2")
    assert u_lang_2.status == ChangeKind.CHANGED
    core.invalidate(u_lang_2.changeset)

    assert w1.status == WorkStatus.STALE
    assert w2.status == WorkStatus.PENDING

    # w1 is blocked
    outcome_1 = await harness.call("price_quote", budget, work=w1, route="SFO-JFK", cabin="economy")
    assert outcome_1.status == "blocked"

    # w2 executes successfully
    outcome_2 = await harness.call("price_quote", budget, work=w2, route="SFO-JFK", cabin="economy")
    assert outcome_2.status == "ok"
    assert w2.status == WorkStatus.VALID
    assert w2.output is not None


# ---------------------------------------------------------------------------
# 6. Successful valid WorkItem records output and transitions to VALID
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_successful_work_item_records_output():
    harness = Harness(strict=True)
    budget = harness.new_budget()

    work = WorkItem(kind="price_quote", turn_id="t1", status=WorkStatus.PENDING)

    outcome = await harness.call(
        "price_quote",
        budget,
        work=work,
        route="BLR-BOM",
        cabin="economy",
    )

    assert outcome.status == "ok"
    assert work.status == WorkStatus.VALID
    assert work.output == outcome.result
    assert isinstance(work.output, dict)
    assert "quote_inr" in work.output


# ---------------------------------------------------------------------------
# 7. Arbitrary non-travel domain facts work identically
# ---------------------------------------------------------------------------
def test_arbitrary_domain_fact_invalidation():
    core = BackspaceCore()

    u_device = core.assert_fact("device_type", "smart_tv", source="telemetry", turn_id="t1")
    u_resolution = core.assert_fact("resolution", "4k", source="telemetry", turn_id="t1")

    work = WorkItem(
        kind="render_stream",
        turn_id="t1",
        depends_on_facts=[u_device.fact.fact_id, u_resolution.fact.fact_id],
    )
    core.register_work(work)
    core.register_dependency(DependencyKind.FACT_TO_WORK, u_device.fact.fact_id, work.work_id)
    core.register_dependency(DependencyKind.FACT_TO_WORK, u_resolution.fact.fact_id, work.work_id)

    # Change resolution
    u_resolution_2 = core.assert_fact("resolution", "1080p", source="telemetry", turn_id="t2")
    assert u_resolution_2.status == ChangeKind.CHANGED
    invalidation = core.invalidate(u_resolution_2.changeset)

    assert work.work_id in invalidation.invalidated_work_ids
    assert work.status == WorkStatus.STALE


# ---------------------------------------------------------------------------
# 8. Runtime end-to-end integration: tool creates WorkItem & invalidates on turn 2
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runtime_creates_work_and_invalidates_on_fact_change():
    collector = Collector()
    session = Session(session_id="runtime-tool-test")
    runtime = AgentRuntime(session, collector)

    # Turn 1: user asks query with facts
    await runtime.on_final("Find hotels in Kyoto for 2 people")
    assert runtime._task is not None
    await runtime._task

    # Verify work items were registered
    work_ids = session.backspace.graph.all_work_ids()
    assert len(work_ids) >= 1

    search_work_items = [
        session.backspace.graph.get_work(wid)
        for wid in work_ids
        if session.backspace.graph.get_work(wid).kind == "search_corpus"
    ]
    assert len(search_work_items) >= 1
    t1_work = search_work_items[0]
    assert t1_work.status == WorkStatus.VALID
    assert len(t1_work.depends_on_facts) >= 1

    # Turn 2: user updates party size
    await runtime.on_final("Actually make it 5 people")
    assert runtime._task is not None
    await runtime._task

    # Previous work item depending on party_size v1 should now be STALE
    assert t1_work.status == WorkStatus.STALE


# ---------------------------------------------------------------------------
# 9. Selective invalidation: unrelated fact change does NOT stale WorkItem
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runtime_unrelated_fact_change_does_not_stale_work_item():
    collector = Collector()
    session = Session(session_id="runtime-selective-test")
    runtime = AgentRuntime(session, collector)

    # Pre-seed an unrelated fact into session (e.g. ui_theme="dark")
    session.backspace.assert_fact("ui_theme", "dark", source="settings", turn_id="t0")
    theme_fact = session.backspace.get_fact("ui_theme")
    assert theme_fact is not None

    # Turn 1: user asks query that establishes destination and party_size
    await runtime.on_final("Find hotels in Kyoto for 2 people")
    assert runtime._task is not None
    await runtime._task

    # Find the search_corpus work item
    work_ids = session.backspace.graph.all_work_ids()
    search_work_items = [
        session.backspace.graph.get_work(wid)
        for wid in work_ids
        if session.backspace.graph.get_work(wid).kind == "search_corpus"
    ]
    assert len(search_work_items) >= 1
    t1_work = search_work_items[0]
    assert t1_work.status == WorkStatus.VALID

    # Verify t1_work depends ONLY on facts that contributed to its query (not ui_theme)
    assert theme_fact.fact_id not in t1_work.depends_on_facts

    # Change the unrelated fact
    u_theme_2 = session.backspace.assert_fact("ui_theme", "light", source="settings", turn_id="t2")
    assert u_theme_2.status == ChangeKind.CHANGED
    session.backspace.invalidate(u_theme_2.changeset)

    # t1_work must remain VALID because ui_theme was unrelated to its query
    assert t1_work.status == WorkStatus.VALID


# ---------------------------------------------------------------------------
# 10. send_booking depends on canonical facts (destination, dates, party_size)
#     and stales on relevant change, while unrelated fact does NOT stale it
# ---------------------------------------------------------------------------
def test_send_booking_depends_on_canonical_facts_and_stales_selectively():
    session = Session(session_id="booking-deps-test")
    runtime = AgentRuntime(session, Collector())

    # Pre-seed canonical facts
    u_dest = session.backspace.assert_fact("destination", "Kyoto", source="extraction", turn_id="t1")
    u_party = session.backspace.assert_fact("party_size", 2, source="extraction", turn_id="t1")
    u_date = session.backspace.assert_fact("dates", "2025-10-10", source="extraction", turn_id="t1")
    u_tier = session.backspace.assert_fact("account_tier", "gold", source="system", turn_id="t1")

    # Construct booking args using the canonical facts
    booking_args = {
        "route": "Kyoto",
        "date": "2025-10-10",
        "passenger": "2",
    }
    work = runtime._create_and_register_work("send_booking", tool_args=booking_args, turn_id="t1")

    # Verify work depends on destination, party_size, dates, but NOT account_tier
    assert u_dest.fact.fact_id in work.depends_on_facts
    assert u_party.fact.fact_id in work.depends_on_facts
    assert u_date.fact.fact_id in work.depends_on_facts
    assert u_tier.fact.fact_id not in work.depends_on_facts
    assert work.status == WorkStatus.PENDING

    # Changing unrelated fact (account_tier) does NOT stale the booking work item
    u_tier_2 = session.backspace.assert_fact("account_tier", "platinum", source="system", turn_id="t2")
    assert u_tier_2.status == ChangeKind.CHANGED
    session.backspace.invalidate(u_tier_2.changeset)
    assert work.status == WorkStatus.PENDING

    # Changing relevant fact (dates) stales the booking work item
    u_date_2 = session.backspace.assert_fact("dates", "2025-10-15", source="extraction", turn_id="t3")
    assert u_date_2.status == ChangeKind.CHANGED
    session.backspace.invalidate(u_date_2.changeset)
    assert work.status == WorkStatus.STALE


# ---------------------------------------------------------------------------
# 11. send_booking stales on party_size or destination change
# ---------------------------------------------------------------------------
def test_send_booking_stales_on_party_size_and_destination_changes():
    session = Session(session_id="booking-party-dest-test")
    runtime = AgentRuntime(session, Collector())

    u_dest = session.backspace.assert_fact("destination", "Kyoto", source="extraction", turn_id="t1")
    u_party = session.backspace.assert_fact("party_size", 3, source="extraction", turn_id="t1")

    booking_args = {"route": "Kyoto", "date": "", "passenger": "3"}
    work = runtime._create_and_register_work("send_booking", tool_args=booking_args, turn_id="t1")
    assert u_dest.fact.fact_id in work.depends_on_facts
    assert u_party.fact.fact_id in work.depends_on_facts
    assert work.status == WorkStatus.PENDING

    # Changing party_size stales the work item
    u_party_2 = session.backspace.assert_fact("party_size", 4, source="extraction", turn_id="t2")
    assert u_party_2.status == ChangeKind.CHANGED
    session.backspace.invalidate(u_party_2.changeset)
    assert work.status == WorkStatus.STALE

    # Register a new work item with updated facts
    work2 = runtime._create_and_register_work("send_booking", tool_args={"route": "Kyoto", "date": "", "passenger": "4"}, turn_id="t2")
    assert work2.status == WorkStatus.PENDING
    assert u_dest.fact.fact_id in work2.depends_on_facts
    assert u_party_2.fact.fact_id in work2.depends_on_facts

    # Changing destination stales the new work item
    u_dest_2 = session.backspace.assert_fact("destination", "Osaka", source="extraction", turn_id="t3")
    assert u_dest_2.status == ChangeKind.CHANGED
    session.backspace.invalidate(u_dest_2.changeset)
    assert work2.status == WorkStatus.STALE


# ---------------------------------------------------------------------------
# 12. End-to-end runtime turn: send_booking populates args and tracks selective dependencies
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runtime_turn_send_booking_arg_construction_and_dependencies():
    collector = Collector()
    session = Session(session_id="runtime-booking-turn-test")
    runtime = AgentRuntime(session, collector)

    # Turn 1: user expresses trip facts
    await runtime.on_final("I want to visit Kyoto with 3 people")
    assert runtime._task is not None
    await runtime._task

    # Pre-seed unrelated fact
    session.backspace.assert_fact("ui_theme", "dark", source="settings", turn_id="t1")
    theme_fact = session.backspace.get_fact("ui_theme")
    assert theme_fact is not None

    # Turn 2: user triggers booking action intent
    await runtime.on_final("Please book the ticket for me now")
    assert runtime._task is not None
    await runtime._task

    # Find the send_booking work item
    work_ids = session.backspace.graph.all_work_ids()
    booking_work_items = [
        session.backspace.graph.get_work(wid)
        for wid in work_ids
        if session.backspace.graph.get_work(wid).kind == "send_booking"
    ]
    assert len(booking_work_items) == 1
    booking_work = booking_work_items[0]

    # Verify tool frames captured canonical facts in args
    tool_frames = collector.of("tool")
    booking_tool_frames = [f for f in tool_frames if f.get("name") == "send_booking"]
    assert len(booking_tool_frames) >= 1
    last_call = booking_tool_frames[-1]
    assert "Kyoto" in str(last_call.get("args", {}).get("route", ""))
    assert last_call.get("args", {}).get("passenger") == "[redacted]"

    # Verify booking_work dependencies: includes destination and party_size, NOT ui_theme
    dest_fact = session.backspace.get_fact("destination")
    party_fact = session.backspace.get_fact("party_size")
    assert dest_fact is not None
    assert party_fact is not None
    assert dest_fact.fact_id in booking_work.depends_on_facts
    assert party_fact.fact_id in booking_work.depends_on_facts
    assert theme_fact.fact_id not in booking_work.depends_on_facts

    # Unrelated fact change does NOT stale booking work
    u_theme_2 = session.backspace.assert_fact("ui_theme", "light", source="settings", turn_id="t3")
    session.backspace.invalidate(u_theme_2.changeset)
    assert booking_work.status == WorkStatus.PENDING

    # Relevant fact change (destination) DOES stale booking work
    u_dest_2 = session.backspace.assert_fact("destination", "Osaka", source="extraction", turn_id="t4")
    session.backspace.invalidate(u_dest_2.changeset)
    assert booking_work.status == WorkStatus.STALE

