"""Argument clean-up enforces the value-form rules by parameter kind, never by item."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from app.backspace import BackspaceCore
from app.fdb import runner as fdb_runner
from app.fdb.adapter import FDBBackspaceAdapter
from app.fdb.arguments import normalize_arguments
from app.session import Session, Turn


@pytest.mark.parametrize("spoken, joined", [
    ("X-Y-Z-9-8", "XYZ98"), ("Q 7 X 2", "Q7X2"), ("M.N.4", "MN4"), ("R_2", "R2"), (" T-1 ", "T1"),
])
def test_codes_spelled_character_by_character_are_joined(spoken: str, joined: str) -> None:
    assert normalize_arguments({"order_id": spoken})["order_id"] == joined
    assert normalize_arguments({"doc_number": spoken})["doc_number"] == joined
    assert normalize_arguments({"confirmation_code": spoken})["confirmation_code"] == joined


@pytest.mark.parametrize("value", ["ORD-55821", "XYZ98", "SKU 1204", "PROD1", "12-345"])
def test_codes_that_are_not_spelled_out_are_left_alone(value: str) -> None:
    assert normalize_arguments({"order_id": value})["order_id"] == value


def test_only_identifier_parameters_are_joined() -> None:
    args = {"query": "a b c", "destination": "S-F", "bill_type": "c-c", "quantity": 2}
    assert normalize_arguments(args) == args


@pytest.mark.parametrize("value, heard, expected", [
    ("2031-04-09", "fly me out on April ninth", "April 9"),
    ("2031-04-09", "April 9th 2031 please", "2031-04-09"),  # the user said the year: kept
    ("October 3rd", "", "October 3"),
    ("the 22nd of May", "", "the 22 of May"),
    ("next Friday", "", "next Friday"),
    ("2031-13-40", "", "2031-13-40"),  # not a real date: untouched
])
def test_dates_keep_the_form_the_user_gave(value: str, heard: str, expected: str) -> None:
    assert normalize_arguments({"date": value}, heard)["date"] == expected
    assert normalize_arguments({"departure_date": value}, heard)["departure_date"] == expected


async def test_every_tool_call_goes_out_normalized_and_is_logged_that_way(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FDB_NORMALIZE_ARGS", raising=False)
    session = Session(session_id="room")
    session.add_turn(Turn("t1", "user", "where is order X Y Z 9 8, and book a flight on the 5th of June"))
    adapter = FDBBackspaceAdapter(BackspaceCore(), "room", session=session, telemetry_path=str(tmp_path / "t.log"))
    tools = {t.info.name: t for t in fdb_runner.wrap_assistant_tools(
        adapter, fdb_runner.AssistantFnc(fdb_runner.LatencyTracker(), "room"))}
    await tools["track_order"](order_id="X-Y-Z-9-8")
    await tools["search_flights"](destination="Oslo", date="2031-06-05")
    logged = [json.loads(line)["call"]["args"] for line in (tmp_path / "t.log").read_text(encoding="utf-8").splitlines()]
    assert logged == [{"order_id": "XYZ98"}, {"destination": "Oslo", "date": "June 5"}]


async def test_normalization_can_be_switched_off(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FDB_NORMALIZE_ARGS", "0")
    adapter = FDBBackspaceAdapter(BackspaceCore(), "room", telemetry_path=str(tmp_path / "t.log"))
    tools = {t.info.name: t for t in fdb_runner.wrap_assistant_tools(
        adapter, fdb_runner.AssistantFnc(fdb_runner.LatencyTracker(), "room"))}
    await tools["track_order"](order_id="X-Y-Z-9-8")
    assert json.loads((tmp_path / "t.log").read_text(encoding="utf-8"))["call"]["args"] == {"order_id": "X-Y-Z-9-8"}


def test_no_test_item_values_are_written_into_the_normalizer() -> None:
    source = Path(__file__).resolve().parents[2] / "app" / "fdb" / "arguments.py"
    data = fdb_runner._FDB_V3_DIR / "benchmark_data_v2.json"
    if not data.exists():
        pytest.skip("FDB-v3 checkout not present")
    code = source.read_text(encoding="utf-8")
    expected = json.loads(data.read_text(encoding="utf-8"))
    values = {str(v) for s in expected["scenarios"] for c in s.get("expected_tool_calls", s.get("tool_calls", []))
              for v in (c.get("arguments") or c.get("args") or {}).values() if len(str(v)) >= 3}
    assert values, "expected values not found in the benchmark file"
    values -= {"True", "False", "None"}  # Python's own keywords, not test data
    leaked = sorted(v for v in values if re.search(rf"(?<![\w.]){re.escape(v)}(?![\w.])", code))
    assert not leaked, leaked


@pytest.mark.parametrize("sent, passed", [("true", True), ("False", False), ("1800", 1800), ("2.5", 2.5),
                                          ("-3", -3), ("downtown", "downtown"), ("1,800", "1,800")])
def test_untyped_backend_parameters_get_typed_literals(sent: str, passed) -> None:
    assert normalize_arguments({"value": sent}, untyped={"value"})["value"] == passed
    assert normalize_arguments({"value": sent})["value"] == sent  # typed parameters: untouched


def test_only_the_backends_any_parameters_are_untyped() -> None:
    untyped = {n: fdb_runner._untyped_backend_params(n) for n in fdb_runner.registry.FUNCTIONS}
    assert untyped["update_search_filter"] == {"value"}
    assert all(not v for n, v in untyped.items() if n != "update_search_filter")


async def test_a_filter_value_reaches_the_api_as_a_boolean(tmp_path: Path) -> None:
    adapter = FDBBackspaceAdapter(BackspaceCore(), "room", telemetry_path=str(tmp_path / "t.log"))
    tools = {t.info.name: t for t in fdb_runner.wrap_assistant_tools(
        adapter, fdb_runner.AssistantFnc(fdb_runner.LatencyTracker(), "room"))}
    await tools["update_search_filter"](filter_name="parking", value="true")
    logged = json.loads((tmp_path / "t.log").read_text(encoding="utf-8"))["call"]["args"]
    assert logged == {"filter_name": "parking", "value": True}
