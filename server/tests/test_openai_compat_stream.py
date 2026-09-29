"""OpenAICompatProvider.stream() against a fake /chat/completions endpoint.

This is the path Groq, OpenAI, Together, vLLM and Ollama all take. The HTTP
layer is an in-process httpx.MockTransport - no network, no API key - and
asyncio.sleep is replaced so retry waits are recorded, not waited.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from app.providers import openai_compat
from app.providers.base import SYSTEM, GenerationRequest
from app.providers.gemini import SYSTEM as GEMINI_SYSTEM
from app.providers.openai_compat import OpenAICompatProvider

_RealAsyncClient = httpx.AsyncClient


def _sse(*texts: str, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n" for t in texts]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class FakeEndpoint:
    """Replays one scripted httpx.Response per request and records the requests."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def _fake_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    return recorded


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch):
    def install(*responses: httpx.Response) -> FakeEndpoint:
        fake = FakeEndpoint(list(responses))
        transport = httpx.MockTransport(fake)
        monkeypatch.setattr(
            openai_compat.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw)
        )
        return fake

    monkeypatch.setattr(
        openai_compat,
        "settings",
        openai_compat.settings.__class__(
            llm_api_key="test-key",
            llm_base_url="https://llm.example/openai/v1/",
            llm_model="test-model",
        ),
    )
    return install


def _ok(*texts: str) -> httpx.Response:
    return httpx.Response(200, content=_sse(*texts), headers={"content-type": "text/event-stream"})


def _error(status: int, retry_after: str | None = None, message: str = "slow down") -> httpx.Response:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return httpx.Response(status, json={"error": {"message": message}}, headers=headers)


async def _collect(received: list[str]) -> None:
    request = GenerationRequest(goal="Delhi fares", utterance="how much is a flight from Delhi to Hyderabad")
    async for chunk in OpenAICompatProvider().stream(request):
        received.append(chunk)


async def test_streams_deltas_and_sends_an_openai_compatible_request(endpoint, sleeps) -> None:
    fake = endpoint(_ok("Fares ", "usually ", "vary."))
    received: list[str] = []

    await _collect(received)

    assert received == ["Fares ", "usually ", "vary."]
    (request,) = fake.requests
    assert str(request.url) == "https://llm.example/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "test-model" and body["stream"] is True
    assert body["messages"][0] == {"role": "system", "content": SYSTEM}
    assert "Delhi to Hyderabad" in body["messages"][-1]["content"]
    assert sleeps == []


async def test_rate_limit_is_retried_honouring_retry_after(endpoint, sleeps) -> None:
    fake = endpoint(_error(429, retry_after="2"), _ok("ok"))
    received: list[str] = []

    await _collect(received)

    assert received == ["ok"]
    assert len(fake.requests) == 2
    assert sleeps == [2.0]


async def test_transient_server_errors_back_off_then_give_up(endpoint, sleeps) -> None:
    fake = endpoint(_error(503), _error(502), _error(503, message="still down"))
    received: list[str] = []

    with pytest.raises(RuntimeError, match="provider returned 503: .*still down"):
        await _collect(received)

    assert received == []
    assert len(fake.requests) == 3
    assert sleeps == [1.0, 2.0]  # no wait after the final attempt


async def test_long_retry_after_fails_fast_instead_of_stalling_the_turn(endpoint, sleeps) -> None:
    fake = endpoint(_error(429, retry_after="3600", message="daily limit reached"))
    received: list[str] = []

    with pytest.raises(RuntimeError, match="provider returned 429: .*daily limit reached"):
        await _collect(received)

    assert len(fake.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [400, 401, 404])
async def test_client_errors_are_not_retried(endpoint, sleeps, status: int) -> None:
    fake = endpoint(_error(status, message="bad request"))
    received: list[str] = []

    with pytest.raises(RuntimeError, match=f"provider returned {status}"):
        await _collect(received)

    assert len(fake.requests) == 1
    assert sleeps == []


async def test_error_event_mid_stream_raises_without_resending_text(endpoint, sleeps) -> None:
    body = _sse("Partial ", done=False) + b'data: {"error": {"message": "overloaded"}}\n\n'
    fake = endpoint(httpx.Response(200, content=body), _ok("never sent"))
    received: list[str] = []

    with pytest.raises(RuntimeError, match="provider stream error: .*overloaded"):
        await _collect(received)

    assert received == ["Partial "]
    assert len(fake.requests) == 1
    assert sleeps == []


async def test_stream_ignores_keepalives_and_empty_deltas(endpoint, sleeps) -> None:
    body = (
        b": keep-alive\n\n"
        + b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
        + _sse("Hello", done=False)
        + b'data: {"choices": []}\n\n'
        + b"data: [DONE]\n\n"
    )
    endpoint(httpx.Response(200, content=body))
    received: list[str] = []

    await _collect(received)

    assert received == ["Hello"]


def test_real_model_providers_share_a_general_purpose_prompt() -> None:
    assert GEMINI_SYSTEM is SYSTEM
    assert openai_compat.SYSTEM is SYSTEM
    lowered = SYSTEM.lower()
    assert "general-purpose" in lowered and "general knowledge" in lowered
    assert "use a passage only when the user asks about" in lowered  # passages are optional, not a cage
    assert "never swap in a different route" in lowered
    assert "never invent facts that are not in the evidence" not in lowered


def test_the_prompt_frames_the_knowledge_base_as_a_fictional_demo_agency() -> None:
    lowered = SYSTEM.lower()
    assert "interject travel, a fictional demo travel agency" in lowered
    assert "not real-world facts" in lowered
    assert "never assume the user belongs to a company or corporate account" in lowered
    assert "never apply a client policy to the user unless" in lowered
    assert "general travel questions, ignore the passages" in lowered


def test_evidence_reaches_the_model_labelled_as_fictional_sample_data() -> None:
    request = GenerationRequest(
        goal="trip", utterance="I want to travel to Mumbai on Friday",
        evidence=[{"doc_id": "policy#7", "title": "Approval thresholds", "snippet": "For corporate clients..."}],
    )
    user = OpenAICompatProvider()._messages(request)[-1]["content"]
    assert "Evidence (Interject Travel demo knowledge base - fictional sample data):" in user
    assert "[policy#7] Approval thresholds" in user


def _use_model(monkeypatch: pytest.MonkeyPatch, model: str, **overrides: Any) -> None:
    monkeypatch.setattr(
        openai_compat,
        "settings",
        openai_compat.settings.__class__(
            llm_api_key="test-key", llm_base_url="https://llm.example/openai/v1/", llm_model=model, **overrides
        ),
    )


async def test_reasoning_models_are_asked_to_think_briefly_and_answers_are_capped(endpoint, sleeps, monkeypatch) -> None:
    fake = endpoint(_ok("ok"))
    _use_model(monkeypatch, "openai/gpt-oss-120b")
    await _collect([])
    body = json.loads(fake.requests[0].content)
    assert body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] == 900


async def test_other_models_get_the_cap_but_no_reasoning_field(endpoint, sleeps) -> None:
    fake = endpoint(_ok("ok"))
    await _collect([])
    body = json.loads(fake.requests[0].content)
    assert "reasoning_effort" not in body
    assert body["max_completion_tokens"] == 900


async def test_tuning_can_be_switched_off(endpoint, sleeps, monkeypatch) -> None:
    fake = endpoint(_ok("ok"))
    _use_model(monkeypatch, "openai/gpt-oss-120b", llm_max_tokens=0, llm_reasoning_effort="")
    await _collect([])
    body = json.loads(fake.requests[0].content)
    assert "reasoning_effort" not in body and "max_completion_tokens" not in body


async def test_an_endpoint_rejecting_a_tuning_field_is_asked_again_without_it(endpoint, sleeps, monkeypatch) -> None:
    fake = endpoint(_error(400, message="property 'reasoning_effort' is unsupported"), _ok("answer"))
    _use_model(monkeypatch, "openai/gpt-oss-120b")
    received: list[str] = []
    await _collect(received)
    assert received == ["answer"]
    assert len(fake.requests) == 2 and sleeps == []
    retry = json.loads(fake.requests[1].content)
    assert "reasoning_effort" not in retry and "max_completion_tokens" not in retry


def test_the_prompt_keeps_spoken_answers_short_and_honest() -> None:
    lowered = SYSTEM.lower()
    assert "two to four sentences" in lowered and "at most five short items" in lowered
    assert "never attach interject travel's fare names" in lowered
    assert "the airline's or hotel's own rules apply on the day" in lowered
    assert "no booking system, portal, account or live fares" in lowered
    assert "say briefly that you have no live data" in lowered
    assert "never name a specific business, building, office or project unless" in lowered
