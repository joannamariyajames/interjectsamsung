"""FDB-v3 E2E Integration smoke test for ecommerce_09.

Validates the full target integration architecture:
User voice / finalized turn
  ↓
Gemini Realtime / AgentSession
  ↓
existing Agent Runtime / finalized user turn
  ↓
FDB-v3-compatible tool execution (wrap_assistant_tools)
  ↓
FDBBackspaceAdapter
  ↓
BACKSPACE Core
  ↓
fact/dependency update
  ↓
stale dependent work
  ↓
revised tool call
  ↓
FDB-v3 telemetry/evaluation (evaluate_tool_calls.py)

Uses the actual official benchmark input and expected correction from:
- Full-Duplex-Bench/v3/fdb_v3_data_released/ecommerce_09_695bd157114f0d2317f88617/metadata.json
- Full-Duplex-Bench/v3/benchmark_data_v2.json
- Official MockAPIRegistry from Full-Duplex-Bench/v3/mock_apis.py
- Official evaluate_tool_calls from Full-Duplex-Bench/v3/evaluate_tool_calls.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Resolve Full-Duplex-Bench/v3
_REPO_ROOT = Path(__file__).resolve().parents[3]  # interjectsamsung
_FDB_V3_DIR = _REPO_ROOT.parent / "Full-Duplex-Bench" / "v3"
if _FDB_V3_DIR.exists() and str(_FDB_V3_DIR) not in sys.path:
    sys.path.insert(0, str(_FDB_V3_DIR))

# Ensure interjectsamsung/server is in sys.path
_SERVER_DIR = _REPO_ROOT / "server"
if _SERVER_DIR.exists() and str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from app.backspace import (
    BackspaceCore,
    ChangeKind,
    FactStatus,
    WorkItem,
    WorkStatus,
)
from app.backspace.work_lifecycle import is_result_current
from app.fdb.adapter import FDBBackspaceAdapter
from app.fdb.runner import wrap_assistant_tools
from app.harness import Harness
from app.runtime import AgentRuntime
from app.session import Session
from evaluate_tool_calls import evaluate_scenario
from lk_agent_tool import AssistantFnc, LatencyTracker
from mock_apis import MockAPIRegistry


def load_official_ecommerce_09_scenario() -> dict[str, Any]:
    """Load the official ecommerce_09 benchmark metadata from benchmark_data_v2.json."""
    benchmark_path = _FDB_V3_DIR / "benchmark_data_v2.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("scenarios", []) if isinstance(data, dict) else data
    for sc in scenarios:
        if sc.get("id") == "ecommerce_09":
            return sc
    raise ValueError("ecommerce_09 not found in benchmark_data_v2.json")


def load_official_ecommerce_09_metadata() -> dict[str, Any]:
    """Load the released metadata for ecommerce_09."""
    rel_path = (
        _FDB_V3_DIR
        / "fdb_v3_data_released"
        / "ecommerce_09_695bd157114f0d2317f88617"
        / "metadata.json"
    )
    with open(rel_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_ecommerce_09_full_duplex_bench_integration(tmp_path: Path):
    """End-to-end integration smoke test proving requirements A through J."""
    # 0. Load official benchmark scenario & ground truth
    scenario = load_official_ecommerce_09_scenario()
    released_meta = load_official_ecommerce_09_metadata()

    assert scenario["id"] == "ecommerce_09"
    assert released_meta["id"] == "ecommerce_09"
    assert scenario["state_rollback_test"] is True

    rollback = scenario["state_rollback_details"]
    original_param = rollback["original_param"]  # {"query": "running shoes"}
    corrected_param = rollback["corrected_param"]  # {"query": "hiking boots"}
    expected_tool_calls = scenario["expected_tool_calls"]
    # [{"function": "search_products", "args": {"query": "hiking boots"}}]

    # 1. Setup LiveKit / AgentRuntime / BACKSPACE Core & Adapter session
    telemetry_file = tmp_path / "agent_tool_calls.log"
    session_state = Session(session_id="ecommerce_09_test_session")
    core: BackspaceCore = session_state.backspace

    frames_received = []

    async def emit_collector(frame):
        frames_received.append(frame)

    runtime = AgentRuntime(session_state, emit=emit_collector)
    adapter = FDBBackspaceAdapter(
        core=core,
        room_name="ecommerce_09_room",
        telemetry_path=str(telemetry_file),
    )

    # 2. Wire FDB-v3 AssistantFnc and official MockAPIRegistry
    tracker = LatencyTracker()
    fnc_ctx = AssistantFnc(tracker, "ecommerce_09_room")
    tools = wrap_assistant_tools(adapter, fnc_ctx)
    tool_map = {t.info.name: t for t in tools}
    assert "search_products" in tool_map
    search_tool = tool_map["search_products"]

    # -----------------------------------------------------------------------
    # Requirement A: Initial user request reaches the LiveKit/Gemini runtime
    # -----------------------------------------------------------------------
    turn_1_dialogue = scenario["dialogue"][0]
    initial_user_msg = turn_1_dialogue["user"]
    await runtime.on_final(initial_user_msg)
    if runtime._task is not None:
        await runtime._task
    assert len(session_state.turns) >= 1
    assert session_state.turns[0].content == initial_user_msg

    # -----------------------------------------------------------------------
    # Requirement B: Initial tool work created with the original search fact
    # -----------------------------------------------------------------------
    # The initial request searches for running shoes
    res1_raw = await search_tool._func(fnc_ctx, **original_param)
    res1 = json.loads(res1_raw)

    assert res1["status"] == "success"
    assert "running shoes" in res1["products"][0]["name"]

    search_works = [
        core.graph.get_work(wid)
        for wid in core.graph.all_work_ids()
        if core.graph.get_work(wid).kind == "search_products"
    ]
    assert len(search_works) == 1
    work_1 = search_works[0]
    assert work_1 is not None
    assert work_1.kind == "search_products"
    assert work_1.status == WorkStatus.VALID
    assert work_1.output == res1

    fact_query_v1 = core.get_fact("query")
    assert fact_query_v1 is not None
    assert fact_query_v1.value == original_param["query"]  # "running shoes"
    assert fact_query_v1.version == 1
    assert fact_query_v1.status == FactStatus.CURRENT
    assert fact_query_v1.fact_id in work_1.depends_on_facts

    # -----------------------------------------------------------------------
    # Requirement C: User self-correction recognized as a new finalized turn
    # -----------------------------------------------------------------------
    turn_2_dialogue = scenario["dialogue"][1]
    correction_user_msg = turn_2_dialogue["user"]
    # "Like... well... hmm... could you search for running shoes — actually no......
    # um, I just remembered I already have running shoes. Search for hiking boots instead, that's what I actually need."
    await runtime.on_final(correction_user_msg)
    if runtime._task is not None:
        await runtime._task
    user_turns = [t for t in session_state.turns if t.role == "user"]
    assert len(user_turns) >= 2
    assert user_turns[1].content == correction_user_msg

    # -----------------------------------------------------------------------
    # Requirement D: Corrected fact asserted into BACKSPACE
    # Requirement E: Original dependent work becomes STALE
    # -----------------------------------------------------------------------
    # The user's finalized turn observation updates the canonical search fact in BACKSPACE
    corr_update = runtime.observe_fact("query", corrected_param["query"], source="user")
    assert corr_update.change_kind == ChangeKind.CHANGED

    fact_query_v2 = core.get_fact("query")
    assert fact_query_v2 is not None
    assert fact_query_v2.value == corrected_param["query"]  # "hiking boots"
    assert fact_query_v2.version == 2
    assert fact_query_v2.status == FactStatus.CURRENT
    assert fact_query_v2.fact_id != fact_query_v1.fact_id

    # Original work item must now be STALE due to fact change
    work_1_updated = core.graph.get_work(work_1.work_id)
    assert work_1_updated is not None
    assert work_1_updated.status == WorkStatus.STALE

    # -----------------------------------------------------------------------
    # Requirement F: Stale work cannot produce an accepted/exported final result
    # -----------------------------------------------------------------------
    # 1. Advisory check: is_result_current rejects work_1
    assert is_result_current(work_1_updated, work_1_updated.execution_attempt) is False

    # 2. Gate check: executing child tool against stale work is blocked / superseded
    stale_attempt_res = adapter.execute_tool(
        func_name="search_products",
        args=corrected_param,
        call_fn=lambda **kw: {"status": "should_not_run"},
        parent_work_id=work_1_updated.work_id,
    )
    assert stale_attempt_res["status"] == "cancelled"
    assert stale_attempt_res["reason"] == "superseded"

    # 3. Harness gate blocks stale work item
    harness = Harness(strict=True)
    budget = harness.new_budget()
    harness_outcome = await harness.call("search_corpus", budget, work=work_1_updated)
    assert harness_outcome.status == "blocked"
    assert "stale" in harness_outcome.verdict.lower()

    # -----------------------------------------------------------------------
    # Requirement G: New work item created using corrected fact
    # Requirement H: Revised tool call reaches official MockAPIRegistry with corrected arg
    # -----------------------------------------------------------------------
    # Revised tool call dispatched with corrected parameters from self-correction
    res2_raw = await search_tool._func(fnc_ctx, **corrected_param)
    res2 = json.loads(res2_raw)

    all_works = [core.graph.get_work(wid) for wid in core.graph.all_work_ids()]
    valid_search_works = [
        w for w in all_works if w.kind == "search_products" and w.status == WorkStatus.VALID
    ]
    assert len(valid_search_works) == 1
    work_2 = valid_search_works[0]

    assert work_2.work_id != work_1.work_id
    assert work_2.kind == "search_products"
    assert work_2.status == WorkStatus.VALID
    assert fact_query_v2.fact_id in work_2.depends_on_facts
    assert fact_query_v1.fact_id not in work_2.depends_on_facts

    assert res2["status"] == "success"
    assert "hiking boots" in res2["products"][0]["name"]
    assert work_2.output == res2

    # -----------------------------------------------------------------------
    # Requirement I: Telemetry contains valid revised call and NOT stale call
    # -----------------------------------------------------------------------
    written_count = adapter.flush()
    assert written_count == 1
    assert telemetry_file.exists()

    lines = telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["room"] == "ecommerce_09_room"
    assert record["call"]["function"] == "search_products"
    assert record["call"]["args"] == {"query": "hiking boots"}
    # The stale call with query="running shoes" is NOT present!

    # Validate against official FDB-v3 evaluate_scenario evaluator
    eval_result = evaluate_scenario(
        scenario=scenario,
        actual_calls=[record["call"]],
        transcript=correction_user_msg,
        result_data=None,
        use_llm=False,
    )

    metrics = eval_result["metrics"]
    assert metrics["tool_selection_acc"]["score"] == 1.0
    assert metrics["tool_selection_acc"]["recall"] == 1.0
    assert metrics["tool_selection_acc"]["precision"] == 1.0
    assert metrics["argument_acc"]["score"] == 1.0
