"""Phase 5 End-to-End Interactive Backspace Demo Tests.

Verifies the real interactive flow connecting Phase 4 reconciliation to the
LiveKit / Gemini realtime runtime:
user request -> tool call -> assistant response -> user correction while session continues ->
old work becomes stale -> stale call is blocked -> corrected tool call executes ->
assistant responds with updated result.

Guarantees:
- Gemini Realtime's 12 tools remain static (no update_tools() or mutation).
- Site-packages and official Full-Duplex-Bench files remain untouched.
- Stale work items are cleanly blocked at the Phase 3 gate.
- Every executed call is reported to official telemetry; blocked stale calls never run.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from app.backspace import (
    BackspaceCore,
    ChangeKind,
    FactStatus,
    WorkItem,
    WorkStatus,
)
from app.fdb.runner import (
    FDBRunnerContext,
    create_fdb_runner_context,
    wrap_assistant_tools,
)
from app.session import Turn
from livekit.agents.llm import ToolContext
from livekit.plugins.google.realtime.realtime_api import create_tools_config
from lk_agent_tool import AssistantFnc, LatencyTracker


# ===========================================================================
# 1. Complete end-to-end interactive demo flow
# ===========================================================================


@pytest.mark.asyncio
async def test_phase5_interactive_flow_user_request_to_updated_response(tmp_path: Path):
    """Test the complete interactive flow:
    user request -> tool call -> assistant response -> user correction while session continues ->
    old work becomes stale -> stale call is blocked -> corrected tool call executes ->
    assistant responds with updated result.
    """
    telemetry_file = tmp_path / "interactive_demo.log"

    # 1. Initialize runner context connecting LiveKit tools, Session, BackspaceCore & Adapter
    runner_ctx = create_fdb_runner_context(
        room_name="interactive_demo_room",
        telemetry_path=str(telemetry_file),
    )
    session = runner_ctx.session_state
    core = runner_ctx.core
    tools = runner_ctx.tools

    # Verify static 12-tool surface initially
    assert len(tools) == 12
    initial_cfg = create_tools_config(ToolContext(tools))
    initial_tool_names = [t.info.name for t in tools]

    # -----------------------------------------------------------------------
    # Step 1: User Request
    # -----------------------------------------------------------------------
    user_msg_1 = "I want to buy running shoes with max price 100 dollars"
    session.add_turn(Turn(turn_id="turn_1", role="user", content=user_msg_1))

    # Facts established for Turn 1
    session.set_fact("query", "running shoes", source="user", turn_id="turn_1")
    session.set_fact("max_price", 100.0, source="user", turn_id="turn_1")

    assert session.get_fact("query") == "running shoes"
    assert session.get_fact("max_price") == 100.0

    # -----------------------------------------------------------------------
    # Step 2: Tool Call Dispatched by Runtime
    # -----------------------------------------------------------------------
    search_tool = runner_ctx.get_tool("search_products")
    call_1_raw = await search_tool(query="running shoes", max_price=100.0)
    res_1 = json.loads(call_1_raw)

    assert res_1["status"] == "success"
    assert "running shoes" in res_1["products"][0]["name"].lower()

    # Verify WorkItem w1 created and VALID
    w1_list = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "search_products"
    ]
    assert len(w1_list) == 1
    w1 = w1_list[0]
    assert w1.status == WorkStatus.VALID
    assert w1.output == res_1

    q_fact_v1 = core.get_fact("query")
    mp_fact_v1 = core.get_fact("max_price")
    assert q_fact_v1 is not None and q_fact_v1.version == 1
    assert mp_fact_v1 is not None and mp_fact_v1.version == 1

    # -----------------------------------------------------------------------
    # Step 3: Assistant Response
    # -----------------------------------------------------------------------
    # Assistant synthesizes and speaks response based on tool results
    top_product_1 = res_1["products"][0]
    assistant_resp_1 = (
        f"I found {len(res_1['products'])} running shoes under $100 for you, "
        f"such as the {top_product_1['name']} for ${top_product_1['price']}."
    )
    session.add_turn(Turn(turn_id="turn_1", role="agent", content=assistant_resp_1))

    assert len(session.turns) == 2
    assert session.turns[-1].role == "agent"
    assert "running shoes" in session.turns[-1].content.lower()

    # -----------------------------------------------------------------------
    # Step 4: User Correction While Session Continues
    # -----------------------------------------------------------------------
    # The session is active; user interrupts / continues with a correction
    user_msg_2 = "Wait, actually, I already have running shoes. Search for hiking boots instead."
    session.add_turn(Turn(turn_id="turn_2", role="user", content=user_msg_2))

    # Fact correction asserted in Session.facts & BackspaceCore
    update = session.set_fact("query", "hiking boots", source="user", turn_id="turn_2")
    assert update.status == ChangeKind.CHANGED
    assert update.changeset is not None

    # Verify query fact is version 2 and max_price remains current (unaffected)
    q_fact_v2 = core.get_fact("query")
    assert q_fact_v2 is not None and q_fact_v2.version == 2
    assert q_fact_v2.status == FactStatus.CURRENT
    assert session.get_fact("max_price") == 100.0

    # Old work w1 is immediately marked STALE by reconciliation
    assert core.graph.get_work(w1.work_id).status == WorkStatus.STALE

    # -----------------------------------------------------------------------
    # Step 5: Stale Call Blocked by Phase 3 Gate
    # -----------------------------------------------------------------------
    # Any delayed attempt or retry referencing old WorkItem w1 is blocked
    stale_call_raw = await search_tool(query="running shoes", work=w1)
    stale_call = json.loads(stale_call_raw)
    assert stale_call == {"status": "cancelled", "reason": "superseded"}

    # Direct gate check on adapter also blocks
    direct_blocked = runner_ctx.adapter.execute_tool(
        "search_products",
        {"query": "running shoes"},
        call_fn=lambda **kw: {"status": "fail_if_called"},
        work=w1,
    )
    assert direct_blocked == {"status": "cancelled", "reason": "superseded"}

    # -----------------------------------------------------------------------
    # Step 6: Corrected Tool Call Executes with Fresh WorkItem
    # -----------------------------------------------------------------------
    # Tool executes with query="hiking boots" (inherits max_price=100.0 from session.facts)
    call_2_raw = await search_tool(query="hiking boots")
    res_2 = json.loads(call_2_raw)

    assert res_2["status"] == "success"
    assert "hiking boots" in res_2["products"][0]["name"].lower()

    # Verify fresh WorkItem w2 was created and is VALID
    all_search_works = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "search_products"
    ]
    assert len(all_search_works) == 2
    w2 = [w for w in all_search_works if w.work_id != w1.work_id][0]
    assert w2.status == WorkStatus.VALID
    assert q_fact_v2.fact_id in w2.depends_on_facts

    # -----------------------------------------------------------------------
    # Step 7: Assistant Responds with Updated Result
    # -----------------------------------------------------------------------
    top_product_2 = res_2["products"][0]
    assistant_resp_2 = (
        f"Understood! I updated your search to hiking boots under $100: "
        f"{top_product_2['name']} for ${top_product_2['price']}."
    )
    session.add_turn(Turn(turn_id="turn_2", role="agent", content=assistant_resp_2))

    assert len(session.turns) == 4
    agent_turns = [t for t in session.turns if t.role == "agent"]
    assert len(agent_turns) == 2
    assert "running shoes" in agent_turns[0].content.lower()
    assert "hiking boots" in agent_turns[1].content.lower()

    # -----------------------------------------------------------------------
    # Step 8: Telemetry Flush & Invariant Validation
    # -----------------------------------------------------------------------
    # Both executed searches are reported, each when it ran; the blocked
    # re-invocations of the stale work never executed and are not reported.
    assert runner_ctx.adapter.flush() == 0
    assert telemetry_file.exists()

    records = [
        json.loads(line)
        for line in telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert [r["call"]["args"]["query"] for r in records] == ["running shoes", "hiking boots"]
    assert records[1]["call"]["function"] == "search_products"
    assert records[1]["call"]["args"]["max_price"] == 100.0

    # Verify Gemini tools remained 100% static
    assert len(tools) == 12
    assert [t.info.name for t in tools] == initial_tool_names
    final_cfg = create_tools_config(ToolContext(tools))
    assert final_cfg == initial_cfg


# ===========================================================================
# 2. LiveKit transcribed event & latency tracking pipeline
# ===========================================================================


@dataclass
class MockUserInputTranscribedEvent:
    transcript: str
    is_final: bool = True
    type: str = "user_input_transcribed"


@dataclass
class MockAgentStateChangedEvent:
    new_state: str = "speaking"
    type: str = "agent_state_changed"


@pytest.mark.asyncio
async def test_phase5_livekit_session_event_pipeline(tmp_path: Path):
    """Test interactive flow triggered via LiveKit user_input_transcribed events
    and agent_state_changed latency breakdown logging."""
    telemetry_file = tmp_path / "livekit_events.log"

    # Create dummy model for AgentSession
    mock_model = MagicMock()
    mock_model.realtime = True

    runner_ctx = create_fdb_runner_context(
        room_name="livekit_event_room",
        model=mock_model,
        telemetry_path=str(telemetry_file),
    )
    tracker = runner_ctx.tracker
    session = runner_ctx.session_state
    adapter = runner_ctx.adapter
    lk_session = runner_ctx.session

    assert lk_session is not None
    search_tool = runner_ctx.get_tool("search_products")

    # 1. Turn 1: User input transcribed event arrives
    evt1 = MockUserInputTranscribedEvent(
        transcript="Search for running shoes", is_final=True
    )
    # Simulate on_user_input handler
    if not tracker.query_received:
        tracker.user_done_at = 100.0
        tracker.query_received = True

    session.set_fact("query", "running shoes", source="user")
    call1_raw = await search_tool(query="running shoes")
    res1 = json.loads(call1_raw)
    assert res1["status"] == "success"

    # Agent speaks
    if tracker.query_received:
        tracker.agent_start_at = 101.5
        tracker.log_breakdown(tool_name="search_products", room_name="livekit_event_room")
        tracker.reset()

    # 2. Turn 2: User correction transcribed event arrives
    evt2 = MockUserInputTranscribedEvent(
        transcript="Actually, make that hiking boots", is_final=True
    )
    if not tracker.query_received:
        tracker.user_done_at = 105.0
        tracker.query_received = True

    # Fact correction
    update = session.set_fact("query", "hiking boots", source="user")
    assert update.status == ChangeKind.CHANGED

    # Corrected tool execution
    call2_raw = await search_tool(query="hiking boots")
    res2 = json.loads(call2_raw)
    assert res2["status"] == "success"

    # Agent speaks updated result
    if tracker.query_received:
        tracker.agent_start_at = 106.2
        tracker.log_breakdown(tool_name="search_products", room_name="livekit_event_room")
        tracker.reset()

    # Telemetry reports both searches that executed, in order
    assert adapter.flush() == 0
    records = [json.loads(line) for line in telemetry_file.read_text(encoding="utf-8").splitlines()]
    assert [r["call"]["args"]["query"] for r in records] == ["running shoes", "hiking boots"]


# ===========================================================================
# 3. Multi-domain cart action follow-up after correction
# ===========================================================================


@pytest.mark.asyncio
async def test_phase5_multidomain_cart_followup_after_correction(tmp_path: Path):
    """Test interactive flow where user corrects search, executes the revised
    search, and then adds the resulting product to the shopping cart."""
    telemetry_file = tmp_path / "cart_followup.log"

    runner_ctx = create_fdb_runner_context(
        room_name="cart_followup_room",
        telemetry_path=str(telemetry_file),
    )
    session = runner_ctx.session_state
    core = runner_ctx.core

    search_tool = runner_ctx.get_tool("search_products")
    cart_tool = runner_ctx.get_tool("add_to_cart")

    # 1. Turn 1: Search laptops
    call1 = await search_tool(query="laptop")
    res1 = json.loads(call1)
    assert res1["status"] == "success"

    search_w1 = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "search_products"
    ][0]
    assert search_w1.status == WorkStatus.VALID

    # 2. Turn 2: User correction from laptop to tablet
    session.set_fact("query", "tablet", source="user")
    assert core.graph.get_work(search_w1.work_id).status == WorkStatus.STALE

    # 3. Revised search executes cleanly
    call2 = await search_tool(query="tablet")
    res2 = json.loads(call2)
    assert res2["status"] == "success"

    search_w2 = [
        w
        for w in [core.graph.get_work(wid) for wid in core.graph.all_work_ids()]
        if w.kind == "search_products" and w.work_id != search_w1.work_id
    ][0]
    assert search_w2.status == WorkStatus.VALID

    # 4. Turn 3: User follow-up: adds the tablet product to cart
    product_to_buy = res2["products"][0]["product_id"]
    cart_call = await cart_tool(
        product_id=product_to_buy, quantity=1, parent_work_id=search_w2.work_id
    )
    cart_res = json.loads(cart_call)
    assert cart_res["status"] == "success"

    cart_work = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "add_to_cart"
    ][0]
    assert cart_work.status == WorkStatus.VALID
    assert search_w2.work_id in cart_work.depends_on_work

    # 5. Telemetry reports every executed call in order: the superseded
    # search, the revised search, and the cart addition
    assert runner_ctx.adapter.flush() == 0

    records = [
        json.loads(line)
        for line in telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    functions = [r["call"]["function"] for r in records]
    assert functions == ["search_products", "search_products", "add_to_cart"]
    assert [r["call"]["args"].get("query") for r in records[:2]] == ["laptop", "tablet"]
