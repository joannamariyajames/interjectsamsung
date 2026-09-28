"""Phase 4 End-to-End Backspace Reconciliation Tests.

Verifies:
1. Complete lifecycle flow:
   Original request -> tool call -> user correction -> old work becomes stale ->
   stale call blocked by Phase 3 gate -> new tool call executes with updated facts.
2. Unaffected facts and work items are preserved during fact correction.
3. Telemetry buffering and flushing only emits valid work items, pruning stale ones.
4. Gemini's 12-tool surface remains strictly static throughout the lifecycle.
5. End-to-end integration between Session.facts, BackspaceCore, and FDBBackspaceAdapter.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from app.backspace import (
    BackspaceCore,
    ChangeKind,
    DependencyKind,
    FactStatus,
    WorkItem,
    WorkStatus,
)
from app.fdb.adapter import FDBBackspaceAdapter
from app.fdb.runner import wrap_assistant_tools
from app.runtime import AgentRuntime
from app.session import Session
from livekit.agents import llm
from livekit.agents.llm import ToolContext
from livekit.plugins.google.realtime.realtime_api import create_tools_config
from lk_agent_tool import AssistantFnc, LatencyTracker


# ===========================================================================
# 1. Complete end-to-end reconciliation flow test
# ===========================================================================


@pytest.mark.asyncio
async def test_phase4_complete_reconciliation_flow(tmp_path: Path):
    """Verify complete flow: original request -> tool call -> user correction ->
    old work becomes stale -> stale call blocked -> new tool call executes."""
    telemetry_file = tmp_path / "agent_tool_calls.log"

    # Setup session, core, adapter, and wrapped tools
    session = Session(session_id="phase4_e2e_room")
    core = session.backspace
    adapter = FDBBackspaceAdapter(
        core=core,
        room_name="phase4_e2e_room",
        session=session,
        telemetry_path=str(telemetry_file),
    )
    tracker = LatencyTracker()
    fnc_ctx = AssistantFnc(tracker, "phase4_e2e_room")
    tools = wrap_assistant_tools(adapter, fnc_ctx)
    tool_map = {t.info.name: t for t in tools}

    # Verify initial static 12-tool surface
    assert len(tools) == 12
    initial_cfg = create_tools_config(ToolContext(tools))

    # -----------------------------------------------------------------------
    # Step 1: Original Request & First Tool Call
    # -----------------------------------------------------------------------
    # User originally asks for running shoes with max price 100
    res1_raw = await tool_map["search_products"](query="running shoes", max_price=100.0)
    res1 = json.loads(res1_raw)
    assert res1["status"] == "success"
    assert "running shoes" in res1["products"][0]["name"].lower()

    # Verify session facts recorded
    assert session.has_fact("query")
    assert session.has_fact("max_price")
    assert session.get_fact("query") == "running shoes"
    assert session.get_fact("max_price") == 100.0

    # Verify WorkItem w1 created and registered in BackspaceCore
    w1_list = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "search_products"
    ]
    assert len(w1_list) == 1
    w1 = w1_list[0]
    assert w1.status == WorkStatus.VALID
    assert w1.output == res1

    # Verify dependencies on v1 facts
    q_fact_v1 = core.get_fact("query")
    mp_fact_v1 = core.get_fact("max_price")
    assert q_fact_v1 is not None and q_fact_v1.version == 1
    assert mp_fact_v1 is not None and mp_fact_v1.version == 1
    assert q_fact_v1.fact_id in w1.depends_on_facts
    assert mp_fact_v1.fact_id in w1.depends_on_facts

    # -----------------------------------------------------------------------
    # Step 2: User Correction
    # -----------------------------------------------------------------------
    # User corrects fact: "Actually, search for hiking boots instead"
    update = session.set_fact("query", "hiking boots", source="user", turn_id="t2")
    assert update.status == ChangeKind.CHANGED
    assert update.changeset is not None

    # Verify Session.facts has the updated query and preserved max_price
    assert session.get_fact("query") == "hiking boots"
    assert session.get_fact("max_price") == 100.0
    q_fact_v2 = core.get_fact("query")
    assert q_fact_v2 is not None and q_fact_v2.version == 2
    assert q_fact_v2.status == FactStatus.CURRENT

    # Verify old WorkItem w1 was marked STALE by reconciliation
    assert core.graph.get_work(w1.work_id).status == WorkStatus.STALE

    # -----------------------------------------------------------------------
    # Step 3: Stale Call Blocked by Phase 3 Execution Gate
    # -----------------------------------------------------------------------
    # Attempting to re-invoke or complete work with the stale WorkItem w1
    stale_attempt_raw = await tool_map["search_products"](
        query="running shoes", work=w1
    )
    stale_attempt = json.loads(stale_attempt_raw)
    assert stale_attempt == {"status": "cancelled", "reason": "superseded"}

    # Direct adapter gate invocation with stale work is also blocked
    called = False

    def dummy_backend(**kw):
        nonlocal called
        called = True
        return {"status": "should_not_run"}

    gate_result = adapter.execute_tool(
        "search_products",
        {"query": "running shoes"},
        call_fn=dummy_backend,
        work=w1,
    )
    assert gate_result == {"status": "cancelled", "reason": "superseded"}
    assert not called, "Backend call_fn must not be executed for stale work"

    # -----------------------------------------------------------------------
    # Step 4: Next Tool Call Executes with Updated Facts & Fresh WorkItem
    # -----------------------------------------------------------------------
    # The next tool call runs with query="hiking boots" (inheriting max_price=100.0 from session)
    res2_raw = await tool_map["search_products"](query="hiking boots")
    res2 = json.loads(res2_raw)
    assert res2["status"] == "success"
    assert "hiking boots" in res2["products"][0]["name"].lower()

    # Verify fresh WorkItem w2 created
    all_search_works = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "search_products"
    ]
    assert len(all_search_works) == 2
    w2 = [w for w in all_search_works if w.work_id != w1.work_id][0]

    assert w2.work_id != w1.work_id
    assert w2.status == WorkStatus.VALID
    assert q_fact_v2.fact_id in w2.depends_on_facts
    assert q_fact_v1.fact_id not in w2.depends_on_facts

    # -----------------------------------------------------------------------
    # Step 5: Telemetry Pruning & Flush Verification
    # -----------------------------------------------------------------------
    # Only valid work w2 is written; stale work w1 is pruned from telemetry log
    flushed_count = adapter.flush()
    assert flushed_count == 1
    assert telemetry_file.exists()

    records = [
        json.loads(line)
        for line in telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(records) == 1
    rec = records[0]
    assert rec["call"]["function"] == "search_products"
    assert rec["call"]["args"]["query"] == "hiking boots"
    assert rec["call"]["args"]["max_price"] == 100.0

    # -----------------------------------------------------------------------
    # Step 6: Verify Static 12-Tool Surface Preserved
    # -----------------------------------------------------------------------
    assert len(tools) == 12
    final_cfg = create_tools_config(ToolContext(tools))
    assert final_cfg == initial_cfg


# ===========================================================================
# 2. Preservation of unaffected facts and independent work
# ===========================================================================


@pytest.mark.asyncio
async def test_phase4_unaffected_work_and_facts_preserved(tmp_path: Path):
    """Verify that correcting one fact invalidates dependent work while preserving
    independent work items and unrelated facts."""
    telemetry_file = tmp_path / "telemetry_unaffected.log"

    session = Session(session_id="phase4_unaffected_room")
    core = session.backspace
    adapter = FDBBackspaceAdapter(
        core=core,
        room_name="phase4_unaffected_room",
        session=session,
        telemetry_path=str(telemetry_file),
    )
    tracker = LatencyTracker()
    fnc_ctx = AssistantFnc(tracker, "phase4_unaffected_room")
    tools = wrap_assistant_tools(adapter, fnc_ctx)
    tool_map = {t.info.name: t for t in tools}

    # 1. Establish fact and work item A: search_products(query="laptop")
    res_search_raw = await tool_map["search_products"](query="laptop")
    assert json.loads(res_search_raw)["status"] == "success"

    # 2. Establish independent fact and work item B: track_order(order_id="ORD-456")
    res_track_raw = await tool_map["track_order"](order_id="ORD-456")
    assert json.loads(res_track_raw)["status"] == "success"

    # Verify both work items are initially VALID
    works = {w.kind: w for w in [core.graph.get_work(wid) for wid in core.graph.all_work_ids()]}
    assert works["search_products"].status == WorkStatus.VALID
    assert works["track_order"].status == WorkStatus.VALID

    # 3. User corrects ONLY the search query: "Actually, look for desktop"
    update = session.set_fact("query", "desktop", source="user")
    assert update.status == ChangeKind.CHANGED

    # 4. Verify search work becomes STALE, but order tracking work remains VALID
    assert core.graph.get_work(works["search_products"].work_id).status == WorkStatus.STALE
    assert core.graph.get_work(works["track_order"].work_id).status == WorkStatus.VALID

    # Unaffected fact (order_id) remains CURRENT in Session and Backspace
    assert session.get_fact("order_id") == "ORD-456"
    assert core.get_fact("order_id").status == FactStatus.CURRENT

    # 5. Execute new search for desktop
    res_search2_raw = await tool_map["search_products"](query="desktop")
    assert json.loads(res_search2_raw)["status"] == "success"

    # 6. Flush telemetry: contains track_order and new search, but NOT old search
    flushed = adapter.flush()
    assert flushed == 2

    records = [
        json.loads(line)
        for line in telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    funcs = [r["call"]["function"] for r in records]
    assert "track_order" in funcs
    assert "search_products" in funcs

    search_rec = [r for r in records if r["call"]["function"] == "search_products"][0]
    assert search_rec["call"]["args"]["query"] == "desktop"


# ===========================================================================
# 3. Multi-turn runtime observation with adapter execution gate
# ===========================================================================


@pytest.mark.asyncio
async def test_phase4_runtime_observation_end_to_end(tmp_path: Path):
    """Verify integration between AgentRuntime.observe_fact, Session.facts,
    and FDBBackspaceAdapter execution gate."""
    telemetry_file = tmp_path / "runtime_e2e.log"

    session = Session(session_id="runtime_e2e_session")
    runtime = AgentRuntime(session, emit=lambda frame: asyncio.sleep(0))
    adapter = FDBBackspaceAdapter(
        core=session.backspace,
        room_name="runtime_e2e_session",
        session=session,
        telemetry_path=str(telemetry_file),
    )
    tracker = LatencyTracker()
    fnc_ctx = AssistantFnc(tracker, "runtime_e2e_session")
    tools = wrap_assistant_tools(adapter, fnc_ctx)
    tool_map = {t.info.name: t for t in tools}

    # Turn 1: Runtime observes fact query="sneakers"
    r1 = runtime.observe_fact("query", "sneakers", source="user", turn_id="turn_1")
    assert r1.change_kind == ChangeKind.NEW
    assert session.facts["query"] == "sneakers"

    # Tool executes with query from session facts
    res1_raw = await tool_map["search_products"](query="sneakers")
    assert json.loads(res1_raw)["status"] == "success"

    search_work_1 = [
        session.backspace.graph.get_work(wid)
        for wid in session.backspace.graph.all_work_ids()
        if session.backspace.graph.get_work(wid).kind == "search_products"
    ][0]
    assert search_work_1.status == WorkStatus.VALID

    # Turn 2: User correction observed via runtime
    r2 = runtime.observe_fact("query", "boots", source="user", turn_id="turn_2")
    assert r2.change_kind == ChangeKind.CHANGED
    assert session.facts["query"] == "boots"

    # Verify search_work_1 was marked STALE
    assert session.backspace.graph.get_work(search_work_1.work_id).status == WorkStatus.STALE

    # Phase 3 adapter gate blocks any attempt to reuse search_work_1
    blocked_raw = await tool_map["search_products"](query="sneakers", work=search_work_1)
    assert json.loads(blocked_raw) == {"status": "cancelled", "reason": "superseded"}

    # Fresh tool call executes with updated fact
    res2_raw = await tool_map["search_products"](query="boots")
    assert json.loads(res2_raw)["status"] == "success"

    # Telemetry records only the fresh call
    adapter.flush()
    records = [
        json.loads(line)
        for line in telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["call"]["args"]["query"] == "boots"


# ===========================================================================
# 4. Chained parent-child work reconciliation
# ===========================================================================


@pytest.mark.asyncio
async def test_phase4_chained_parent_work_reconciliation(tmp_path: Path):
    """Verify that when a fact is corrected, the parent work becomes STALE, and
    any subsequent child tool referencing parent_work_id is cancelled and blocked."""
    session = Session(session_id="chained_room")
    adapter = FDBBackspaceAdapter(
        core=session.backspace,
        room_name="chained_room",
        session=session,
        telemetry_path=str(tmp_path / "chained.log"),
    )
    tracker = LatencyTracker()
    fnc_ctx = AssistantFnc(tracker, "chained_room")
    tools = wrap_assistant_tools(adapter, fnc_ctx)
    tool_map = {t.info.name: t for t in tools}

    # 1. Parent search call executes
    res_parent_raw = await tool_map["search_products"](query="camera")
    assert json.loads(res_parent_raw)["status"] == "success"

    parent_work = [
        session.backspace.graph.get_work(wid)
        for wid in session.backspace.graph.all_work_ids()
        if session.backspace.graph.get_work(wid).kind == "search_products"
    ][0]
    assert parent_work.status == WorkStatus.VALID

    # 2. Fact correction invalidates parent work
    update = session.set_fact("query", "camcorder", source="user")
    assert update.status == ChangeKind.CHANGED
    assert session.backspace.graph.get_work(parent_work.work_id).status == WorkStatus.STALE

    # 3. Chained child tool (add_to_cart) referencing parent work is blocked
    child_res_raw = await tool_map["add_to_cart"](
        product_id="PROD_CAM", quantity=1, parent_work_id=parent_work.work_id
    )
    child_res = json.loads(child_res_raw)
    assert child_res == {"status": "cancelled", "reason": "superseded"}

    # 4. Independent new parent call with updated fact executes cleanly
    new_parent_raw = await tool_map["search_products"](query="camcorder")
    assert json.loads(new_parent_raw)["status"] == "success"

    new_parent_work = [
        w
        for w in [
            session.backspace.graph.get_work(wid)
            for wid in session.backspace.graph.all_work_ids()
            if session.backspace.graph.get_work(wid).kind == "search_products"
        ]
        if w.work_id != parent_work.work_id
    ][0]
    assert new_parent_work.status == WorkStatus.VALID

    # 5. New child referencing valid new parent executes cleanly
    valid_child_raw = await tool_map["add_to_cart"](
        product_id="PROD_CAM", quantity=1, parent_work_id=new_parent_work.work_id
    )
    assert json.loads(valid_child_raw)["status"] == "success"
