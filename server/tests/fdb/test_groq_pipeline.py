"""The Groq cascaded benchmark agent and its supporting tools - offline, no network."""

from __future__ import annotations

import json
import typing
from pathlib import Path

import pytest
from app.backspace import BackspaceCore
from app.fdb import bench_utils
from app.fdb import runner as fdb_runner
from app.fdb.adapter import FDBBackspaceAdapter
from app.providers.mock import MockProvider
from livekit.agents.llm import utils as lk_utils


def _search_products(tmp_path: Path, nullable: bool):
    adapter = FDBBackspaceAdapter(BackspaceCore(), "room", telemetry_path=str(tmp_path / "t.log"))
    fnc = fdb_runner.AssistantFnc(fdb_runner.LatencyTracker(), "room")
    tools = fdb_runner.wrap_assistant_tools(adapter, fnc, nullable_optionals=nullable)
    return {t.info.name: t for t in tools}["search_products"]


def test_nullable_optionals_accept_null_on_the_groq_path(tmp_path: Path) -> None:
    tool = _search_products(tmp_path, nullable=True)
    schema = lk_utils.function_arguments_to_pydantic_model(tool).model_json_schema()
    assert {"type": "null"} in schema["properties"]["max_price"]["anyOf"]
    assert schema["required"] == ["query"]


def test_official_schema_is_untouched_by_default(tmp_path: Path) -> None:
    tool = _search_products(tmp_path, nullable=False)
    assert typing.get_type_hints(tool)["max_price"] is float
    # ...and the official class itself is never mutated by the Groq path
    _search_products(tmp_path, nullable=True)
    assert typing.get_type_hints(fdb_runner.AssistantFnc.search_products)["max_price"] is float


async def test_null_optional_argument_is_dropped_before_the_call(tmp_path: Path) -> None:
    tool = _search_products(tmp_path, nullable=True)
    result = json.loads(await tool(query="hiking boots", max_price=None))
    assert result["status"] == "success"
    record = json.loads((tmp_path / "t.log").read_text(encoding="utf-8"))
    assert record["call"]["args"] == {"query": "hiking boots"}  # no invented max_price


def test_interject_agent_keeps_the_official_instructions_and_adds_rules() -> None:
    official = fdb_runner.VoiceAgent().instructions
    ours = fdb_runner.InterjectVoiceAgent().instructions
    assert ours.startswith(official)
    assert ours[len(official):] == fdb_runner.ARGUMENT_RULES
    # the rules are general: no benchmark scenario ids or domains in them
    for token in ("ecommerce_", "finance_", "housing_", "travel_"):
        assert token not in fdb_runner.ARGUMENT_RULES


def test_groq_components_build_a_cascaded_pipeline_without_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LK_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    components = fdb_runner.resolve_session_components()
    assert set(components) == {"stt", "llm", "tts", "vad"}
    assert components["llm"].model == "openai/gpt-oss-120b"
    import os

    assert "GEMINI_API_KEY" not in os.environ  # no hidden Gemini calls from any component


async def test_runner_context_uses_more_tool_steps_and_an_offline_background_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FDB_MAX_TOOL_STEPS", "6")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    llm = fdb_runner.build_groq_llm()
    # AgentSession pre-warms the LLM's HTTP connection; a unit test stays offline
    monkeypatch.setattr(llm, "prewarm", lambda *a, **k: None)
    ctx = fdb_runner.create_fdb_runner_context(
        "room", components={"llm": llm}, telemetry_path=str(tmp_path / "t.log")
    )
    try:
        assert ctx.session is not None
        assert ctx.session.options.max_tool_steps == 6
        assert isinstance(ctx.runtime.provider, MockProvider)
    finally:
        await ctx.session.aclose()
        await ctx.runtime.shutdown()


def test_subset_takes_the_first_recording_of_each_scenario(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for folder, scenario in [("a_01_x", "a_01"), ("a_01_y", "a_01"), ("b_02_x", "b_02"), ("c_03_x", "c_03")]:
        d = data / folder
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"id": scenario}), encoding="utf-8")
        (d / "input.wav").write_bytes(b"RIFF")
    out = tmp_path / "subset"
    bench_utils.subset(str(data), str(out), "2")
    assert sorted(p.name for p in out.iterdir()) == ["a_01_x", "b_02_x"]


def test_groq_pipeline_can_be_built_inside_a_job_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiveKit runs each job's entrypoint off the main thread. Plugins imported
    there fail with "Plugins must be registered on the main thread" and the
    agent never answers, so they must already be imported with the runner."""
    import threading

    monkeypatch.setenv("LK_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    outcome: dict[str, object] = {}

    def job() -> None:
        try:
            outcome["components"] = sorted(fdb_runner.resolve_session_components())
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    t = threading.Thread(target=job)
    t.start()
    t.join(timeout=60)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["components"] == ["llm", "stt", "tts", "vad"]


async def test_the_agent_waits_out_a_mid_request_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    monkeypatch.delenv("FDB_MIN_ENDPOINTING_DELAY", raising=False)
    llm = fdb_runner.build_groq_llm()
    monkeypatch.setattr(llm, "prewarm", lambda *a, **k: None)
    ctx = fdb_runner.create_fdb_runner_context("room", components={"llm": llm}, telemetry_path=str(tmp_path / "t.log"))
    try:
        # 1.2 s of silence ends a turn, not LiveKit's 0.3 s default for this pipeline
        assert ctx.session.options.endpointing["min_delay"] == 1.2
    finally:
        await ctx.session.aclose()
        await ctx.runtime.shutdown()
    monkeypatch.setenv("FDB_MIN_ENDPOINTING_DELAY", "0.8")
    ctx = fdb_runner.create_fdb_runner_context("room", components={"llm": llm}, telemetry_path=str(tmp_path / "t.log"))
    try:
        assert ctx.session.options.endpointing["min_delay"] == 0.8
    finally:
        await ctx.session.aclose()
        await ctx.runtime.shutdown()
