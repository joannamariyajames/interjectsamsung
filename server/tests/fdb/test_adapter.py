"""Deterministic tests for Member 4 FDBBackspaceAdapter.

Ensures the adapter correctly interacts with Member 1's BackspaceCore without
touching external networks, LiveKit Cloud, audio, or real LLMs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.backspace import (
    BackspaceCore,
    DependencyKind,
    WorkStatus,
)
from app.fdb.adapter import FDBBackspaceAdapter


@pytest.fixture
def core() -> BackspaceCore:
    return BackspaceCore()


@pytest.fixture
def telemetry_file(tmp_path: Path) -> Path:
    return tmp_path / "agent_tool_calls.log"


@pytest.fixture
def adapter(core: BackspaceCore, telemetry_file: Path) -> FDBBackspaceAdapter:
    return FDBBackspaceAdapter(
        core=core,
        room_name="test-room-123",
        telemetry_path=str(telemetry_file),
    )


def test_valid_tool_delegates_once_to_call_fn(adapter: FDBBackspaceAdapter):
    """1. Valid tool invocation delegates exactly once to call_fn."""
    mock_fn = MagicMock(return_value={"status": "success", "flights": []})
    result = adapter.execute_tool(
        func_name="search_flights",
        args={"destination": "London", "date": "2026-08-20"},
        call_fn=mock_fn,
        turn_id="t1",
    )

    mock_fn.assert_called_once_with(destination="London", date="2026-08-20")
    assert result == {"status": "success", "flights": []}


def test_one_invocation_creates_one_work_item(adapter: FDBBackspaceAdapter, core: BackspaceCore):
    """2. One invocation creates exactly one WorkItem."""
    mock_fn = MagicMock(return_value={"status": "success"})
    adapter.execute_tool(
        func_name="search_flights",
        args={"destination": "London"},
        call_fn=mock_fn,
        turn_id="t1",
    )

    registered_work = list(core.graph._work.values())
    assert len(registered_work) == 1
    work = registered_work[0]
    assert work.kind == "search_flights"
    assert work.turn_id == "t1"
    assert work.status == WorkStatus.VALID
    assert work.provenance["args"] == {"destination": "London"}


def test_tool_arguments_create_expected_facts_and_dependencies(
    adapter: FDBBackspaceAdapter, core: BackspaceCore
):
    """3. Tool arguments create the expected fact/dependency relationships."""
    mock_fn = MagicMock(return_value={"status": "success"})
    adapter.execute_tool(
        func_name="search_flights",
        args={"destination": "London", "date": "2026-08-20"},
        call_fn=mock_fn,
        turn_id="t1",
    )

    dest_fact = core.get_fact("destination")
    date_fact = core.get_fact("date")

    assert dest_fact is not None
    assert dest_fact.value == "London"
    assert date_fact is not None
    assert date_fact.value == "2026-08-20"

    work = list(core.graph._work.values())[0]
    dependencies = core.get_dependencies(work.work_id)
    assert dest_fact.fact_id in dependencies
    assert date_fact.fact_id in dependencies


def test_changed_fact_invalidates_dependent_work_item(
    adapter: FDBBackspaceAdapter, core: BackspaceCore
):
    """4. A changed fact invalidates its dependent WorkItem through the existing Core."""
    mock_fn = MagicMock(return_value={"status": "success"})
    adapter.execute_tool(
        func_name="search_flights",
        args={"destination": "London"},
        call_fn=mock_fn,
        turn_id="t1",
    )
    work = list(core.graph._work.values())[0]
    assert work.status == WorkStatus.VALID

    # Now user updates destination
    update = core.assert_fact("destination", "Paris", source="user", turn_id="t2")
    invalidation = core.invalidate(update.changeset)

    assert work.work_id in invalidation.invalidated_work_ids
    assert core.graph.get_work(work.work_id).status == WorkStatus.STALE


def test_stale_invalidated_work_item_not_passed_to_call_fn(
    adapter: FDBBackspaceAdapter, core: BackspaceCore
):
    """5. A stale/invalidated WorkItem is not passed to call_fn."""
    # Setup a parent work item that gets invalidated
    parent_mock = MagicMock(return_value={"status": "success"})
    adapter.execute_tool(
        func_name="search_flights",
        args={"destination": "London"},
        call_fn=parent_mock,
        turn_id="t1",
    )
    parent_work = list(core.graph._work.values())[0]

    # Invalidate parent
    update = core.assert_fact("destination", "Paris", source="user", turn_id="t2")
    core.invalidate(update.changeset)
    assert parent_work.status == WorkStatus.STALE

    # Chained child tool should be detected as stale before execution
    child_mock = MagicMock()
    result = adapter.execute_tool(
        func_name="book_flight",
        args={"passenger_name": "Alice"},
        call_fn=child_mock,
        turn_id="t3",
        parent_work_id=parent_work.work_id,
    )

    child_mock.assert_not_called()
    assert result["status"] == "cancelled"


def test_suppressed_execution_returns_cancelled_superseded(
    adapter: FDBBackspaceAdapter, core: BackspaceCore
):
    """6. Suppressed execution returns status=cancelled, reason=superseded."""
    adapter.execute_tool(
        func_name="search_apartments",
        args={"city": "Tokyo"},
        call_fn=lambda **kw: {"status": "success"},
        turn_id="t1",
    )
    parent_work = list(core.graph._work.values())[0]
    parent_work.status = WorkStatus.INVALIDATED

    child_mock = MagicMock()
    result = adapter.execute_tool(
        func_name="calculate_commute",
        args={"origin_address": "Shinjuku", "destination_address": "Shibuya"},
        call_fn=child_mock,
        turn_id="t2",
        parent_work_id=parent_work.work_id,
    )

    child_mock.assert_not_called()
    assert result == {
        "status": "cancelled",
        "reason": "superseded",
    }


def test_valid_result_attached_to_work_item(adapter: FDBBackspaceAdapter, core: BackspaceCore):
    """7. Valid result is attached to the WorkItem."""
    expected_result = {"status": "success", "flights": [{"flight_id": "FL999"}]}
    adapter.execute_tool(
        func_name="search_flights",
        args={"destination": "Berlin"},
        call_fn=lambda **kw: expected_result,
        turn_id="t1",
    )

    work = list(core.graph._work.values())[0]
    assert work.output == expected_result
    assert work.status == WorkStatus.VALID


def test_valid_call_buffered_with_official_telemetry_schema(adapter: FDBBackspaceAdapter):
    """8. Valid call is buffered with the official telemetry schema."""
    adapter.execute_tool(
        func_name="search_products",
        args={"query": "headphones", "max_price": 100.0},
        call_fn=lambda **kw: {"status": "success"},
        turn_id="t1",
    )

    assert len(adapter._buffer) == 1
    buffered = adapter._buffer[0]
    record = buffered["record"]

    assert record["room"] == "test-room-123"
    assert "call" in record
    call = record["call"]
    assert call["function"] == "search_products"
    assert call["args"] == {"query": "headphones", "max_price": 100.0}
    assert isinstance(call["timestamp_start"], float)
    assert isinstance(call["timestamp_end"], float)
    assert call["timestamp_end"] >= call["timestamp_start"]


def test_stale_call_omitted_during_flush(
    adapter: FDBBackspaceAdapter, core: BackspaceCore, telemetry_file: Path
):
    """9. Stale call is omitted during flush."""
    adapter.execute_tool(
        func_name="search_flights",
        args={"destination": "Rome"},
        call_fn=lambda **kw: {"status": "success"},
        turn_id="t1",
    )
    work = list(core.graph._work.values())[0]

    # Invalidate the fact
    update = core.assert_fact("destination", "Milan", source="user", turn_id="t2")
    core.invalidate(update.changeset)
    assert work.status == WorkStatus.STALE

    written = adapter.flush()
    assert written == 0
    assert not telemetry_file.exists()


def test_valid_call_emitted_during_flush(
    adapter: FDBBackspaceAdapter, telemetry_file: Path
):
    """10. Valid call is emitted during flush."""
    adapter.execute_tool(
        func_name="get_card_benefits",
        args={"card_type": "platinum"},
        call_fn=lambda **kw: {"status": "success"},
        turn_id="t1",
    )

    written = adapter.flush()
    assert written == 1
    assert telemetry_file.exists()

    lines = telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["room"] == "test-room-123"
    assert data["call"]["function"] == "get_card_benefits"
    assert data["call"]["args"] == {"card_type": "platinum"}


def test_original_timestamps_preserved(adapter: FDBBackspaceAdapter, telemetry_file: Path):
    """11. Original timestamps are preserved."""
    adapter.execute_tool(
        func_name="track_order",
        args={"order_id": "ORD123"},
        call_fn=lambda **kw: {"status": "success"},
        turn_id="t1",
    )
    buffer_entry = adapter._buffer[0]
    expected_start = buffer_entry["record"]["call"]["timestamp_start"]
    expected_end = buffer_entry["record"]["call"]["timestamp_end"]

    adapter.flush()

    line = telemetry_file.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["call"]["timestamp_start"] == expected_start
    assert data["call"]["timestamp_end"] == expected_end


def test_parent_work_id_creates_exactly_one_work_to_work_dependency(
    adapter: FDBBackspaceAdapter, core: BackspaceCore
):
    """12. parent_work_id creates exactly one WORK_TO_WORK dependency."""
    adapter.execute_tool(
        func_name="search_flights",
        args={"destination": "Madrid"},
        call_fn=lambda **kw: {"status": "success", "flight_id": "FL100"},
        turn_id="t1",
    )
    parent_work = list(core.graph._work.values())[0]

    adapter.execute_tool(
        func_name="book_flight",
        args={"passenger_name": "Bob"},
        call_fn=lambda **kw: {"status": "success"},
        turn_id="t2",
        parent_work_id=parent_work.work_id,
    )
    child_work = list(core.graph._work.values())[1]

    assert parent_work.work_id in child_work.depends_on_work

    # Verify dependency in core graph
    dependents = core.get_direct_dependents(parent_work.work_id)
    assert child_work.work_id in dependents


def test_room_session_adapters_isolated(tmp_path: Path):
    """13. Room/session adapter instances do not share Core state or telemetry buffers."""
    file1 = tmp_path / "room1.log"
    file2 = tmp_path / "room2.log"

    core1 = BackspaceCore()
    core2 = BackspaceCore()

    adapter1 = FDBBackspaceAdapter(core1, "room-1", str(file1))
    adapter2 = FDBBackspaceAdapter(core2, "room-2", str(file2))

    adapter1.execute_tool(
        func_name="search_flights",
        args={"destination": "Cairo"},
        call_fn=lambda **kw: {"status": "success"},
        turn_id="t1",
    )

    assert len(adapter1._buffer) == 1
    assert len(adapter2._buffer) == 0
    assert core1.get_fact("destination") is not None
    assert core2.get_fact("destination") is None

    adapter1.flush()
    adapter2.flush()

    assert file1.exists()
    assert not file2.exists()


def test_calling_flush_twice_does_not_duplicate_records(
    adapter: FDBBackspaceAdapter, telemetry_file: Path
):
    """14. Calling flush twice does not duplicate records."""
    adapter.execute_tool(
        func_name="search_products",
        args={"query": "laptop"},
        call_fn=lambda **kw: {"status": "success"},
        turn_id="t1",
    )

    count1 = adapter.flush()
    count2 = adapter.flush()

    assert count1 == 1
    assert count2 == 0

    lines = telemetry_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_resolve_realtime_model_gemini3_8(monkeypatch: pytest.MonkeyPatch):
    """15. resolve_realtime_model handles gemini3_8, maps GEMINI_API_KEY to GOOGLE_API_KEY, and uses gemini-3.8-live."""
    from app.fdb.runner import resolve_realtime_model

    monkeypatch.setenv("LK_PROVIDER", "gemini3_8")
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key_value")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    model = resolve_realtime_model()

    assert model is not None
    assert getattr(model, "_opts", None) is not None
    assert model._opts.model == "gemini-3.8-live"
    import os
    assert os.environ.get("GOOGLE_API_KEY") == "mock_key_value"

