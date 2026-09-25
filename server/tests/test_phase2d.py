"""Phase 2D integration tests: Facts → Retrieval & Generation pipeline.

Verifies:
A. GenerationRequest carries the `facts` dict populated from BackspaceCore.
B. Facts influence retrieval query composition (non-self-contained turns).
C. MockProvider surfaces facts in generated text.
D. Multi-turn fact accumulation: all facts from prior turns are present in later
   GenerationRequests.
E. Changed facts propagate: when a fact is updated, the new value appears in
   GenerationRequest, not the old one.
F. Non-fact utterances produce empty facts dicts.
G. Provider plan steps mention facts when present.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.backspace import ChangeKind, FactStatus
from app.providers.base import GenerationRequest
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider
from app.providers.openai_compat import OpenAICompatProvider
from app.runtime import AgentRuntime
from app.session import Session
from tests.collector import Collector


def _make_runtime(session_id: str = "phase2d-test") -> tuple[AgentRuntime, Collector, Session]:
    collector = Collector()
    session = Session(session_id=session_id)
    runtime = AgentRuntime(session, collector)
    return runtime, collector, session


# ---------------------------------------------------------------------------
# Requirement A: GenerationRequest carries facts from BackspaceCore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_request_carries_facts():
    """After a fact-bearing utterance, the GenerationRequest passed to the
    provider must include the extracted facts."""
    runtime, collector, session = _make_runtime()

    captured_requests: list[GenerationRequest] = []
    original_plan = runtime.provider.plan

    def capturing_plan(request: GenerationRequest) -> list[str]:
        captured_requests.append(request)
        return original_plan(request)

    runtime.provider.plan = capturing_plan

    await runtime.on_final("Find flights to Tokyo for 4 people")
    assert runtime._task is not None
    await runtime._task

    assert len(captured_requests) >= 1
    req = captured_requests[0]
    assert isinstance(req.facts, dict)

    # At minimum, destination and party_size should be present
    assert "destination" in req.facts
    assert req.facts["destination"] == "Tokyo"
    assert "party_size" in req.facts
    assert req.facts["party_size"] == 4


# ---------------------------------------------------------------------------
# Requirement B: Facts influence retrieval query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facts_influence_retrieval_query():
    """On a terse follow-up (non-self-contained), the retrieval query must
    include fact values so the corpus search is richer."""
    runtime, collector, session = _make_runtime()

    # Turn 1: establish facts
    await runtime.on_final("Find flights to Tokyo for 4 people")
    assert runtime._task is not None
    await runtime._task

    # Verify facts were stored
    assert session.backspace.get_fact("destination") is not None
    assert session.backspace.get_fact("destination").value == "Tokyo"

    # Turn 2: terse refinement — should fold facts into retrieval query.
    # We intercept _retrieve (which wraps both speculation and cold paths)
    # to capture the query string the runtime composes.
    retrieve_queries: list[str] = []
    original_retrieve = runtime._retrieve

    async def spy_retrieve(query, budget, use_speculation=True):
        retrieve_queries.append(query)
        return await original_retrieve(query, budget, use_speculation=use_speculation)

    runtime._retrieve = spy_retrieve

    # "make it business class" is a refinement of the existing goal, not a new one
    await runtime.on_final("make it business class")
    assert runtime._task is not None
    await runtime._task

    # The retrieval query should contain fact values from the session
    assert len(retrieve_queries) >= 1
    query = retrieve_queries[0].lower()
    assert "tokyo" in query or "4" in query, (
        f"Expected fact values in retrieval query, got: {query}"
    )


# ---------------------------------------------------------------------------
# Requirement C: MockProvider surfaces facts in generated text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_provider_surfaces_facts_in_text():
    """The MockProvider must include session facts in the composed output."""
    runtime, collector, session = _make_runtime()

    await runtime.on_final("Find hotels in Osaka for 3 people under 8000")
    assert runtime._task is not None
    await runtime._task

    output = collector.text()
    # The mock provider should mention facts from the conversation
    assert "facts from our conversation" in output.lower() or "destination" in output.lower(), (
        f"Expected facts in mock output, got: {output[:200]}"
    )


# ---------------------------------------------------------------------------
# Requirement D: Multi-turn fact accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_turn_fact_accumulation():
    """Facts from earlier turns must persist and appear in later
    GenerationRequests alongside new facts."""
    runtime, collector, session = _make_runtime()

    captured_requests: list[GenerationRequest] = []
    original_plan = runtime.provider.plan

    def capturing_plan(request: GenerationRequest) -> list[str]:
        captured_requests.append(request)
        return original_plan(request)

    runtime.provider.plan = capturing_plan

    # Turn 1: destination + party_size
    await runtime.on_final("Find flights to Berlin for 2 people")
    assert runtime._task is not None
    await runtime._task

    # Turn 2: add budget
    await runtime.on_final("Keep it under 5000")
    assert runtime._task is not None
    await runtime._task

    assert len(captured_requests) >= 2
    turn2_req = captured_requests[-1]

    # Turn 2 must carry destination from turn 1
    assert "destination" in turn2_req.facts
    assert turn2_req.facts["destination"] == "Berlin"

    # Turn 2 should also carry budget (if extracted)
    # party_size from turn 1 should persist
    assert "party_size" in turn2_req.facts
    assert turn2_req.facts["party_size"] == 2


# ---------------------------------------------------------------------------
# Requirement E: Changed facts propagate updated values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_changed_facts_propagate_updated_values():
    """When a fact changes (e.g. party_size 2 → 5), the GenerationRequest
    must carry the new value, not the old one."""
    runtime, collector, session = _make_runtime()

    captured_requests: list[GenerationRequest] = []
    original_plan = runtime.provider.plan

    def capturing_plan(request: GenerationRequest) -> list[str]:
        captured_requests.append(request)
        return original_plan(request)

    runtime.provider.plan = capturing_plan

    # Turn 1: 2 people
    await runtime.on_final("Find flights to Paris for 2 people")
    assert runtime._task is not None
    await runtime._task

    # Turn 2: change to 5
    await runtime.on_final("Actually make it 5 people")
    assert runtime._task is not None
    await runtime._task

    assert len(captured_requests) >= 2
    turn2_req = captured_requests[-1]

    assert "party_size" in turn2_req.facts
    assert turn2_req.facts["party_size"] == 5, (
        f"Expected updated party_size=5, got {turn2_req.facts['party_size']}"
    )


# ---------------------------------------------------------------------------
# Requirement F: Non-fact utterances produce empty facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_fact_utterance_produces_empty_facts():
    """A question that doesn't contain extractable facts should result in
    an empty facts dict in GenerationRequest."""
    runtime, collector, session = _make_runtime()

    captured_requests: list[GenerationRequest] = []
    original_plan = runtime.provider.plan

    def capturing_plan(request: GenerationRequest) -> list[str]:
        captured_requests.append(request)
        return original_plan(request)

    runtime.provider.plan = capturing_plan

    await runtime.on_final("What are the baggage limits?")
    assert runtime._task is not None
    await runtime._task

    assert len(captured_requests) >= 1
    req = captured_requests[0]
    assert req.facts == {}, (
        f"Expected empty facts for non-fact query, got: {req.facts}"
    )


# ---------------------------------------------------------------------------
# Requirement G: Provider plan steps mention facts
# ---------------------------------------------------------------------------


def test_mock_provider_plan_mentions_facts():
    """When facts are present, MockProvider.plan() must include a step about them."""
    provider = MockProvider()

    req_with_facts = GenerationRequest(
        goal="Find flights",
        utterance="flights to Tokyo",
        facts={"destination": "Tokyo", "party_size": 4},
    )
    steps = provider.plan(req_with_facts)
    assert any("fact" in s.lower() for s in steps), (
        f"Expected a plan step mentioning facts, got: {steps}"
    )

    req_no_facts = GenerationRequest(
        goal="What are baggage limits?",
        utterance="what are baggage limits",
    )
    steps_no_facts = provider.plan(req_no_facts)
    assert not any("fact" in s.lower() for s in steps_no_facts), (
        f"Expected no fact step for empty facts, got: {steps_no_facts}"
    )


# ---------------------------------------------------------------------------
# Requirement H: GenerationRequest.facts is always a plain dict
# ---------------------------------------------------------------------------


def test_generation_request_facts_default():
    """GenerationRequest.facts defaults to an empty dict."""
    req = GenerationRequest(goal="test", utterance="test")
    assert req.facts == {}
    assert isinstance(req.facts, dict)


# ---------------------------------------------------------------------------
# Requirement I: Domain-general arbitrary facts reach GenerationRequest & retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_domain_general_arbitrary_facts_reach_generation_and_retrieval():
    """Arbitrary non-travel facts (topic, language, format) in BackspaceCore
    must reach GenerationRequest.facts and contribute generically to follow-up retrieval."""
    runtime, collector, session = _make_runtime()

    # Pre-seed arbitrary domain facts into session.backspace
    session.backspace.assert_fact("topic", "compiler", source="session", turn_id="t0")
    session.backspace.assert_fact("language", "Python", source="session", turn_id="t0")
    session.backspace.assert_fact("format", "PDF", source="session", turn_id="t0")

    captured_requests: list[GenerationRequest] = []
    original_plan = runtime.provider.plan

    def capturing_plan(request: GenerationRequest) -> list[str]:
        captured_requests.append(request)
        return original_plan(request)

    runtime.provider.plan = capturing_plan

    # Initial turn to establish an active goal
    await runtime.on_final("Generate the technical documentation")
    assert runtime._task is not None
    await runtime._task

    # Turn 1 GenerationRequest should carry all arbitrary session facts
    assert len(captured_requests) >= 1
    req1 = captured_requests[0]
    assert req1.facts.get("topic") == "compiler"
    assert req1.facts.get("language") == "Python"
    assert req1.facts.get("format") == "PDF"

    # Turn 2: Follow-up refinement — should fold arbitrary facts into retrieval query
    retrieve_queries: list[str] = []
    original_retrieve = runtime._retrieve

    async def spy_retrieve(query, budget, use_speculation=True):
        retrieve_queries.append(query)
        return await original_retrieve(query, budget, use_speculation=use_speculation)

    runtime._retrieve = spy_retrieve

    await runtime.on_final("make it include the AST chapter")
    assert runtime._task is not None
    await runtime._task

    assert len(retrieve_queries) >= 1
    query = retrieve_queries[0].lower()
    assert "compiler" in query, f"Expected 'compiler' in retrieval query: {query}"
    assert "python" in query, f"Expected 'python' in retrieval query: {query}"
    assert "pdf" in query, f"Expected 'pdf' in retrieval query: {query}"


# ---------------------------------------------------------------------------
# Requirement J: Empty retrieval handling with facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_retrieval_with_facts():
    """When retrieval finds no evidence (empty list), GenerationRequest is still
    constructed with session facts, and the provider completes without error."""
    runtime, collector, session = _make_runtime()

    # Seed a fact
    session.backspace.assert_fact("department", "finance", source="session", turn_id="t0")

    # Force retrieval to return empty evidence
    async def empty_retrieve(query, budget, use_speculation=True):
        return []

    runtime._retrieve = empty_retrieve

    captured_requests: list[GenerationRequest] = []
    original_plan = runtime.provider.plan

    def capturing_plan(request: GenerationRequest) -> list[str]:
        captured_requests.append(request)
        return original_plan(request)

    runtime.provider.plan = capturing_plan

    await runtime.on_final("What are the quarterly figures?")
    assert runtime._task is not None
    await runtime._task

    assert len(captured_requests) >= 1
    req = captured_requests[0]
    assert req.evidence == []
    assert req.facts.get("department") == "finance"

    # Verify collector received tokens/completion without crashing
    text = collector.text()
    assert len(text) > 0


# ---------------------------------------------------------------------------
# Requirement K: Provider handling of facts (Gemini and OpenAICompat)
# ---------------------------------------------------------------------------


def test_gemini_provider_facts_handling():
    """GeminiProvider includes session facts in plan and _build_contents."""
    provider = GeminiProvider(api_key="fake-key", model="gemini-2.0-flash")
    request = GenerationRequest(
        goal="Compile project",
        utterance="Build it now",
        facts={"topic": "compiler", "language": "Python"},
    )

    steps = provider.plan(request)
    assert any("2 session fact(s)" in s for s in steps)

    contents = provider._build_contents(request)
    assert len(contents) >= 1
    prompt_text = " ".join(p.text for p in contents[-1].parts)
    assert "Session facts (canonical):" in prompt_text
    assert "topic: compiler" in prompt_text
    assert "language: Python" in prompt_text


def test_openai_compat_provider_facts_handling():
    """OpenAICompatProvider includes session facts in plan and _messages."""
    provider = OpenAICompatProvider()
    request = GenerationRequest(
        goal="Process data",
        utterance="Convert the file",
        facts={"format": "PDF", "department": "finance"},
    )

    steps = provider.plan(request)
    assert any("2 session fact(s)" in s for s in steps)

    messages = provider._messages(request)
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "Session facts (canonical):" in user_msg
    assert "format: PDF" in user_msg
    assert "department: finance" in user_msg

