"""Unit tests for GeminiProvider.

All tests mock Google GenAI API calls. No real Gemini API key is needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from app.config import Settings
from app.providers.base import GenerationRequest
from app.providers.gemini import SYSTEM, GeminiProvider
from app.runtime import AgentRuntime
from app.schemas import Stage
from app.session import Session
from tests.collector import Collector


class FakeChunk:
    def __init__(self, text: str | None) -> None:
        self.text = text


class FakeAsyncModel:
    def __init__(self, chunks: list[str], delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.called_with: dict[str, object] = {}

    async def generate_content_stream(self, *, model: str, contents: object, config: object) -> object:
        self.called_with = {
            "model": model,
            "contents": contents,
            "config": config,
        }

        async def _generator():
            for text in self.chunks:
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                yield FakeChunk(text)

        return _generator()


class FakeClient:
    def __init__(self, chunks: list[str], delay: float = 0.0) -> None:
        self.models = FakeAsyncModel(chunks, delay)

    @property
    def aio(self) -> object:
        mock = MagicMock()
        mock.models = self.models
        return mock


def test_gemini_provider_plan() -> None:
    provider = GeminiProvider(api_key="fake-key", model="gemini-2.0-flash")
    request = GenerationRequest(
        goal="Find flights to Seoul",
        utterance="Find me cheap tickets",
        constraints=["under 500 USD", "direct flight"],
        evidence=[
            {"doc_id": "doc1", "title": "Flight 101", "snippet": "Direct flight at 450 USD"},
            {"doc_id": "doc2", "title": "Flight 202", "snippet": "Direct flight at 480 USD"},
        ],
    )
    steps = provider.plan(request)
    assert len(steps) == 4
    assert "Find flights to Seoul" in steps[0]
    assert "under 500 USD, direct flight" in steps[1]
    assert "gemini-2.0-flash" in steps[2]
    assert "2 grounded passage(s)" in steps[2]
    assert "staying interruptible" in steps[3]


def test_gemini_provider_build_contents() -> None:
    provider = GeminiProvider(api_key="fake-key")
    request = GenerationRequest(
        goal="Book train",
        utterance="Which platform?",
        constraints=["standard class"],
        context=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! Where are you traveling?"},
        ],
        evidence=[{"doc_id": "train#1", "title": "Platform Info", "snippet": "Platform 4"}],
        notice="Cannot book tickets directly",
        resume_from="Platform is 4 and",
    )
    contents = provider._build_contents(request)
    assert len(contents) >= 3

    # Turn 0: user
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Hello"

    # Turn 1: model (assistant mapped to model)
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Hi! Where are you traveling?"

    # Turn 2: current prompt as user
    assert contents[2].role == "user"
    combined_user = " ".join(p.text for p in contents[2].parts)
    assert "Active goal: Book train" in combined_user
    assert "Constraints: standard class" in combined_user
    assert "[train#1] Platform Info: Platform 4" in combined_user
    assert "User just said: Which platform?" in combined_user
    assert "The safety harness refused part of this request" in combined_user
    assert "Platform is 4 and" in combined_user


def test_gemini_provider_consecutive_same_role_merged() -> None:
    provider = GeminiProvider(api_key="fake-key")
    request = GenerationRequest(
        goal="Ask questions",
        utterance="Second question",
        context=[
            {"role": "user", "content": "First message"},
            {"role": "user", "content": "Second message before answer"},
        ],
    )
    contents = provider._build_contents(request)
    # Consecutive user messages in context should be merged, plus the current utterance
    assert all(c.role == "user" for c in contents)
    assert len(contents) == 1
    full_text = " ".join(p.text for p in contents[0].parts)
    assert "First message" in full_text
    assert "Second message before answer" in full_text
    assert "Second question" in full_text


async def test_gemini_provider_stream_success() -> None:
    fake_client = FakeClient(chunks=["Hello", " from", " Gemini!"])
    provider = GeminiProvider(client=fake_client)
    request = GenerationRequest(goal="Greeting", utterance="Hi")

    stream = provider.stream(request)
    collected = []
    async for chunk in stream:
        collected.append(chunk)

    assert "".join(collected) == "Hello from Gemini!"
    assert fake_client.models.called_with["config"].system_instruction == SYSTEM
    assert fake_client.models.called_with["config"].temperature == 0.3


async def test_gemini_provider_stream_cancellation() -> None:
    fake_client = FakeClient(chunks=["Chunk 1", "Chunk 2", "Chunk 3"], delay=0.05)
    provider = GeminiProvider(client=fake_client)
    request = GenerationRequest(goal="Test cancellation", utterance="Go")

    async def consume(yielded: list[str]) -> None:
        async for chunk in provider.stream(request):
            yielded.append(chunk)

    yielded: list[str] = []
    task = asyncio.create_task(consume(yielded))
    await asyncio.sleep(0.06)  # allow first chunk to be emitted
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(yielded) >= 1
    assert len(yielded) < 3


async def test_gemini_provider_stream_error() -> None:
    class FailingModel:
        async def generate_content_stream(self, **kwargs: object) -> object:
            raise ValueError("Rate limit exceeded")

    class FailingClient:
        @property
        def aio(self) -> object:
            mock = MagicMock()
            mock.models = FailingModel()
            return mock

    provider = GeminiProvider(client=FailingClient())
    request = GenerationRequest(goal="Fail", utterance="Hello")

    with pytest.raises(RuntimeError, match="Gemini streaming error: Rate limit exceeded"):
        async for _ in provider.stream(request):
            pass


def test_gemini_provider_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("app.providers.gemini.settings", Settings(gemini_api_key=None))
    provider = GeminiProvider(api_key=None, client=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not configured"):
        provider._get_client()


async def test_gemini_provider_with_runtime_turn() -> None:
    fake_client = FakeClient(chunks=["Luggage ", "limit ", "is ", "23kg."])
    provider = GeminiProvider(client=fake_client)

    collector = Collector()
    session = Session(session_id="gemini-test-session")
    runtime = AgentRuntime(session, collector)
    runtime.provider = provider

    await runtime.on_final("What is the luggage limit?")
    await runtime._task

    assert "responding" in collector.stages()
    assert collector.text().strip() == "Luggage limit is 23kg."
    messages = collector.of("message")
    assert len(messages) >= 2
    agent_msg = messages[-1]
    assert agent_msg["role"] == "agent"
    assert agent_msg["status"] == "complete"
    assert agent_msg["content"] == "Luggage limit is 23kg."


async def test_gemini_provider_with_runtime_barge_in_interruption() -> None:
    # Slow streaming to allow interrupting mid-answer
    fake_client = FakeClient(
        chunks=["The ", "baggage ", "policy ", "for ", "international ", "flights ", "allows "],
        delay=0.04,
    )
    provider = GeminiProvider(client=fake_client)

    collector = Collector()
    session = Session(session_id="gemini-interrupt-test")
    runtime = AgentRuntime(session, collector)
    runtime.provider = provider

    await runtime.on_final("What is the international baggage policy?")

    # Wait for at least 2 tokens to appear
    while len(collector.of("token")) < 2:
        await asyncio.sleep(0.01)

    # Interrupt with barge-in
    await runtime.interrupt("barge_in")

    assert Stage.INTERRUPTED.value in collector.stages()
    assert session.checkpoint is not None
    assert len(session.checkpoint.partial_text) > 0
    assert not runtime.busy
