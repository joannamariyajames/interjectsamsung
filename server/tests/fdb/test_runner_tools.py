"""Unit tests for wrapped LiveKit tools execution with prepare_function_arguments.

Tests that tools wrapped via wrap_assistant_tools execute properly with the
positional and keyword arguments produced by prepare_function_arguments(),
while strictly preserving the tool signature and schema.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from app.backspace import BackspaceCore, FactStatus, WorkItem, WorkStatus
from app.fdb.adapter import FDBBackspaceAdapter
from app.fdb.runner import wrap_assistant_tools
from livekit.agents import llm
from livekit.agents.llm import ToolContext
from livekit.agents.llm.utils import prepare_function_arguments
from livekit.plugins.google.realtime.realtime_api import create_tools_config
from lk_agent_tool import AssistantFnc, LatencyTracker


@pytest.fixture
def fdb_environment(tmp_path: Path):
    tracker = LatencyTracker()
    fnc_ctx = AssistantFnc(tracker, "test_runner_room")
    core = BackspaceCore()
    # Never the default /tmp/agent_tool_calls.log: that is the live file the
    # benchmark reads, and executed calls are written to it immediately.
    telemetry_file = tmp_path / "agent_tool_calls.log"
    adapter = FDBBackspaceAdapter(core=core, room_name="test_runner_room", telemetry_path=str(telemetry_file))
    tools = wrap_assistant_tools(adapter, fnc_ctx)
    tool_map = {t.info.name: t for t in tools}
    return {
        "tracker": tracker,
        "fnc_ctx": fnc_ctx,
        "core": core,
        "adapter": adapter,
        "tools": tools,
        "tool_map": tool_map,
        "telemetry_file": telemetry_file,
    }


def test_tool_signatures_and_schemas_preserved(tmp_path: Path):
    """Verify tool signatures and Gemini declarations match official unwrapped tools."""
    tracker_raw = LatencyTracker()
    fnc_raw = AssistantFnc(tracker_raw, "raw_room")
    raw_tools = llm.find_function_tools(fnc_raw)
    raw_cfg = create_tools_config(ToolContext(raw_tools))

    tracker_wrap = LatencyTracker()
    fnc_wrap = AssistantFnc(tracker_wrap, "wrap_room")
    core = BackspaceCore()
    adapter = FDBBackspaceAdapter(core=core, room_name="wrap_room", telemetry_path=str(tmp_path / "wrap.log"))
    wrapped_tools = wrap_assistant_tools(adapter, fnc_wrap)
    wrapped_cfg = create_tools_config(ToolContext(wrapped_tools))

    assert raw_cfg == wrapped_cfg, "Gemini Tool declarations changed after wrapping"

    raw_map = {t.info.name: t for t in raw_tools}
    wrapped_map = {t.info.name: t for t in wrapped_tools}

    for name, raw_tool in raw_map.items():
        assert name in wrapped_map
        wrap_tool = wrapped_map[name]
        assert inspect.signature(raw_tool) == inspect.signature(wrap_tool)
        assert raw_tool.info.name == wrap_tool.info.name
        assert raw_tool.info.description == wrap_tool.info.description


@pytest.mark.asyncio
async def test_search_products_with_prepare_function_arguments(fdb_environment):
    """Test search_products(query='shoes') executes correctly with prepare_function_arguments."""
    env = fdb_environment
    tool = env["tool_map"]["search_products"]
    core: BackspaceCore = env["core"]

    # 1. Prepare arguments using LiveKit's official argument preparer
    raw_args = {"query": "shoes"}
    fnc_args, fnc_kwargs = prepare_function_arguments(fnc=tool, json_arguments=raw_args)

    # Note: prepare_function_arguments binds query as positional arg
    assert len(fnc_args) >= 1
    assert fnc_args[0] == "shoes"

    # 2. Execute through wrapped tool
    result_raw = await tool(*fnc_args, **fnc_kwargs)
    result = json.loads(result_raw)

    assert result["status"] == "success"
    assert "shoes" in result["products"][0]["name"]

    # 3. Verify BACKSPACE work item and fact registration
    query_fact = core.get_fact("query")
    assert query_fact is not None
    assert query_fact.value == "shoes"
    assert query_fact.status == FactStatus.CURRENT

    works = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "search_products"
    ]
    assert len(works) == 1
    assert works[0].status == WorkStatus.VALID
    assert query_fact.fact_id in works[0].depends_on_facts


@pytest.mark.asyncio
async def test_add_to_cart_with_prepare_function_arguments(fdb_environment):
    """Test add_to_cart(product_id='PROD1', quantity=2) with prepare_function_arguments."""
    env = fdb_environment
    tool = env["tool_map"]["add_to_cart"]
    core: BackspaceCore = env["core"]

    raw_args = {"product_id": "PROD1", "quantity": 2}
    fnc_args, fnc_kwargs = prepare_function_arguments(fnc=tool, json_arguments=raw_args)

    result_raw = await tool(*fnc_args, **fnc_kwargs)
    result = json.loads(result_raw)

    assert result["status"] == "success"
    assert result["product_id"] == "PROD1"
    assert result["quantity"] == 2
    assert result["cart_total"] == pytest.approx(199.98)

    prod_fact = core.get_fact("product_id")
    qty_fact = core.get_fact("quantity")
    assert prod_fact is not None and prod_fact.value == "PROD1"
    assert qty_fact is not None and qty_fact.value == 2


@pytest.mark.asyncio
async def test_track_order_with_prepare_function_arguments(fdb_environment):
    """Test track_order(order_id='BOB12') with prepare_function_arguments."""
    env = fdb_environment
    tool = env["tool_map"]["track_order"]
    core: BackspaceCore = env["core"]

    raw_args = {"order_id": "BOB12"}
    fnc_args, fnc_kwargs = prepare_function_arguments(fnc=tool, json_arguments=raw_args)

    result_raw = await tool(*fnc_args, **fnc_kwargs)
    result = json.loads(result_raw)

    assert result["status"] == "success"
    assert result["order_id"] == "BOB12"
    assert "Out for delivery" in result["shipping_status"]

    order_fact = core.get_fact("order_id")
    assert order_fact is not None and order_fact.value == "BOB12"


@pytest.mark.asyncio
async def test_calculate_commute_default_arg_with_prepare_function_arguments(fdb_environment):
    """Test calculate_commute with omitted default parameter through prepare_function_arguments."""
    env = fdb_environment
    tool = env["tool_map"]["calculate_commute"]

    raw_args = {"origin_address": "Downtown", "destination_address": "Airport"}
    fnc_args, fnc_kwargs = prepare_function_arguments(fnc=tool, json_arguments=raw_args)

    result_raw = await tool(*fnc_args, **fnc_kwargs)
    result = json.loads(result_raw)

    assert result["status"] == "success"
    assert result["mode"] == "driving"
    assert result["duration_mins"] == 25


@pytest.mark.asyncio
async def test_direct_invocation_styles(fdb_environment):
    """Verify tool execution succeeds across all invocation styles."""
    env = fdb_environment
    st = env["tool_map"]["search_products"]
    fnc_ctx = env["fnc_ctx"]

    # 1. tool(*fnc_args, **fnc_kwargs)
    args, kwargs = prepare_function_arguments(fnc=st, json_arguments={"query": "shoes"})
    r1 = json.loads(await st(*args, **kwargs))
    assert r1["status"] == "success"

    # 2. tool(query="shoes")
    r2 = json.loads(await st(query="shoes"))
    assert r2["status"] == "success"

    # 3. tool("shoes")
    r3 = json.loads(await st("shoes"))
    assert r3["status"] == "success"

    # 4. tool._func(fnc_ctx, *args, **kwargs)
    r4 = json.loads(await st._func(fnc_ctx, *args, **kwargs))
    assert r4["status"] == "success"

    # 5. tool._func(*args, **kwargs) without fnc_ctx
    r5 = json.loads(await st._func(*args, **kwargs))
    assert r5["status"] == "success"

    # 6. fnc_ctx.search_products(query="shoes")
    r6 = json.loads(await fnc_ctx.search_products(query="shoes"))
    assert r6["status"] == "success"


@pytest.mark.asyncio
async def test_phase3_stale_tool_call_blocked(fdb_environment):
    """Test that a STALE / INVALIDATED / SUPERSEDED WorkItem blocks tool execution."""
    env = fdb_environment
    tool = env["tool_map"]["search_products"]
    core: BackspaceCore = env["core"]
    adapter: FDBBackspaceAdapter = env["adapter"]

    # 1. Create a WorkItem and explicitly mark it STALE
    work = WorkItem(
        kind="search_products",
        work_id="work_test_stale_123",
        status=WorkStatus.STALE,
    )
    core.register_work(work)

    # 2. Attempt to execute tool with this stale work item
    raw_res = await tool(query="shoes", work=work)
    res = json.loads(raw_res)

    # 3. Assert execution was blocked and returned cancelled/superseded
    assert res == {"status": "cancelled", "reason": "superseded"}

    # 4. Assert telemetry was NOT buffered for this blocked call
    assert not any(b["work_id"] == "work_test_stale_123" for b in adapter._buffer)


@pytest.mark.asyncio
async def test_phase3_valid_tool_call_executes_and_buffers_telemetry(tmp_path: Path):
    """Test that valid tool calls execute normally and preserve telemetry buffering."""
    telemetry_file = tmp_path / "telemetry.log"
    tracker = LatencyTracker()
    fnc_ctx = AssistantFnc(tracker, "phase3_room")
    core = BackspaceCore()
    adapter = FDBBackspaceAdapter(
        core=core,
        room_name="phase3_room",
        telemetry_path=str(telemetry_file),
    )
    tools = wrap_assistant_tools(adapter, fnc_ctx)
    tool_map = {t.info.name: t for t in tools}
    tool = tool_map["search_products"]

    raw_res = await tool(query="running shoes")
    res = json.loads(raw_res)

    assert res["status"] == "success"
    assert "running shoes" in res["products"][0]["name"]

    assert telemetry_file.exists()  # written when the call executed
    assert adapter.flush() == 0

    record = json.loads(telemetry_file.read_text(encoding="utf-8").strip())
    assert record["room"] == "phase3_room"
    assert record["call"]["function"] == "search_products"
    assert record["call"]["args"] == {"query": "running shoes"}
    assert record["call"]["timestamp_start"] > 0
    assert record["call"]["timestamp_end"] >= record["call"]["timestamp_start"]


@pytest.mark.asyncio
async def test_phase3_no_runtime_tool_mutation_across_tool_lifecycle(fdb_environment):
    """Prove that all 12 tools remain static and unmutated across fact changes and corrections."""
    env = fdb_environment
    tools = env["tools"]
    tool_map = env["tool_map"]

    # 1. Capture tool context, count, names, and Gemini declarations initially
    assert len(tools) == 12
    initial_names = [t.info.name for t in tools]
    initial_sigs = {t.info.name: inspect.signature(t) for t in tools}
    initial_cfg = create_tools_config(ToolContext(tools))

    # 2. Execute multiple calls (initial search, self-correction, follow-up tool)
    await tool_map["search_products"](query="running shoes")

    # Invalidate query fact in BackspaceCore
    update = env["core"].assert_fact("query", "hiking boots", source="user", turn_id="t2")
    env["core"].invalidate(update.changeset)

    # Execute revised search call
    await tool_map["search_products"](query="hiking boots")

    # Execute follow-up cart tool
    await tool_map["add_to_cart"](product_id="PROD1", quantity=1)

    # 3. Assert tools list, count, order, and signatures are completely UNMUTATED
    assert len(tools) == 12
    assert [t.info.name for t in tools] == initial_names
    current_sigs = {t.info.name: inspect.signature(t) for t in tools}
    assert current_sigs == initial_sigs

    # Gemini config remains 100% identical without any runtime mutation
    current_cfg = create_tools_config(ToolContext(tools))
    assert current_cfg == initial_cfg


@pytest.mark.asyncio
async def test_phase3_chained_parent_work_staleness_blocks_child_tool(fdb_environment):
    """Test that child tool referencing a stale parent WorkItem is gated without runtime tool mutation."""
    env = fdb_environment
    tool_map = env["tool_map"]
    core: BackspaceCore = env["core"]
    adapter: FDBBackspaceAdapter = env["adapter"]

    # 1. Execute parent tool (search_products)
    res1_raw = await tool_map["search_products"](query="running shoes")
    assert json.loads(res1_raw)["status"] == "success"

    parent_works = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "search_products"
    ]
    assert len(parent_works) == 1
    parent_work = parent_works[0]

    # 2. Fact change marks parent work STALE
    update = core.assert_fact("query", "hiking boots", source="user", turn_id="t2")
    core.invalidate(update.changeset)
    assert core.graph.get_work(parent_work.work_id).status == WorkStatus.STALE

    # 3. Chained child tool referencing parent work is blocked at execution gate
    child_res_raw = await tool_map["add_to_cart"](
        product_id="PROD1", quantity=1, parent_work_id=parent_work.work_id
    )
    child_res = json.loads(child_res_raw)

    assert child_res == {"status": "cancelled", "reason": "superseded"}

    # 4. The blocked child never executed, so it is never reported; the parent
    # did execute, so it stays reported even though it is now stale
    assert adapter.flush() == 0
    records = [json.loads(line) for line in env["telemetry_file"].read_text(encoding="utf-8").splitlines()]
    assert [r["call"]["function"] for r in records] == ["search_products"]


def test_runner_session_start_explicitly_disables_recording():
    """Verify that runner.py explicitly passes record=False to session.start."""
    import ast
    from pathlib import Path

    runner_py = Path(__file__).resolve().parents[2] / "app" / "fdb" / "runner.py"
    tree = ast.parse(runner_py.read_text(encoding="utf-8"))

    start_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "start":
                kw_names = {kw.arg: kw.value for kw in node.keywords}
                start_calls.append(kw_names)

    assert len(start_calls) >= 1
    for kw in start_calls:
        assert "record" in kw
        assert isinstance(kw["record"], ast.Constant) and kw["record"].value is False
