"""Unit tests for GeminiProvider.stream()'s 429 (RESOURCE_EXHAUSTED) retry.

Contract under test (app/providers/gemini.py):
- up to 3 attempts in total;
- only errors whose ``code`` is 429 are retried, sleeping 2s then 4s;
- a retry happens only while the failing attempt has yielded no text, so
  already-delivered text is never re-sent;
- once the budget is spent (or the error is not retryable) the error is
  raised as ``RuntimeError("Gemini streaming error: ...")``.

All tests use a fake client and the SDK's own error classes, constructed
locally. No real Gemini API key or network access is needed, and
``asyncio.sleep`` is replaced so the backoff delays are recorded, not waited.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from app.providers.base import GenerationRequest
from app.providers.gemini import GeminiProvider
from google.genai import errors


def _rate_limited() -> errors.ClientError:
    return errors.ClientError(
        429,
        {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}},
    )


class FakeChunk:
    def __init__(self, text: str | None) -> None:
        self.text = text


class ScriptedModel:
    """Each call to generate_content_stream consumes the next scripted outcome:
    an exception raised before streaming starts, or a list of chunk texts
    (optionally followed by an exception raised mid-stream)."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def generate_content_stream(self, **kwargs: object) -> object:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        texts, mid_stream_error = outcome

        async def _generator():
            for text in texts:
                yield FakeChunk(text)
            if mid_stream_error is not None:
                raise mid_stream_error

        return _generator()


class ScriptedClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.models = ScriptedModel(outcomes)

    @property
    def aio(self) -> object:
        mock = MagicMock()
        mock.models = self.models
        return mock


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def _fake_sleep(delay: float, *args: object, **kwargs: object) -> None:
        recorded.append(delay)

    monkeypatch.setattr("app.providers.gemini.asyncio.sleep", _fake_sleep)
    return recorded


async def _collect(provider: GeminiProvider, received: list[str]) -> None:
    async for text in provider.stream(GenerationRequest(goal="Retry", utterance="Hello")):
        received.append(text)


async def test_retries_after_429_and_returns_output(sleeps: list[float]) -> None:
    client = ScriptedClient([_rate_limited(), (["Luggage ", "limit ", "is 23kg."], None)])
    received: list[str] = []

    await _collect(GeminiProvider(client=client), received)

    assert received == ["Luggage ", "limit ", "is 23kg."]
    assert client.models.calls == 2
    assert sleeps == [2]


async def test_succeeds_on_final_attempt_after_two_429s(sleeps: list[float]) -> None:
    client = ScriptedClient([_rate_limited(), _rate_limited(), (["ok"], None)])
    received: list[str] = []

    await _collect(GeminiProvider(client=client), received)

    assert received == ["ok"]
    assert client.models.calls == 3
    assert sleeps == [2, 4]


async def test_gives_up_after_three_429s(sleeps: list[float]) -> None:
    client = ScriptedClient([_rate_limited(), _rate_limited(), _rate_limited()])
    received: list[str] = []

    with pytest.raises(RuntimeError, match="Gemini streaming error") as excinfo:
        await _collect(GeminiProvider(client=client), received)

    assert isinstance(excinfo.value.__cause__, errors.ClientError)
    assert excinfo.value.__cause__.code == 429
    assert received == []
    assert client.models.calls == 3
    assert sleeps == [2, 4]  # no sleep after the final attempt


@pytest.mark.parametrize(
    "error",
    [
        errors.ClientError(400, {"error": {"code": 400, "message": "Bad request", "status": "INVALID_ARGUMENT"}}),
        errors.ServerError(503, {"error": {"code": 503, "message": "Unavailable", "status": "UNAVAILABLE"}}),
        ValueError("no status code at all"),
    ],
    ids=["client-400", "server-503", "no-code"],
)
async def test_non_429_errors_are_not_retried(sleeps: list[float], error: Exception) -> None:
    client = ScriptedClient([error, (["should never be reached"], None)])
    received: list[str] = []

    with pytest.raises(RuntimeError, match="Gemini streaming error") as excinfo:
        await _collect(GeminiProvider(client=client), received)

    assert excinfo.value.__cause__ is error
    assert received == []
    assert client.models.calls == 1
    assert sleeps == []


async def test_429_after_text_was_emitted_is_not_retried(sleeps: list[float]) -> None:
    client = ScriptedClient([(["Partial "], _rate_limited()), (["duplicate"], None)])
    received: list[str] = []

    with pytest.raises(RuntimeError, match="Gemini streaming error"):
        await _collect(GeminiProvider(client=client), received)

    assert received == ["Partial "]  # delivered once, never re-sent
    assert client.models.calls == 1
    assert sleeps == []


async def test_cancellation_during_backoff_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _cancelled_sleep(delay: float, *args: object, **kwargs: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr("app.providers.gemini.asyncio.sleep", _cancelled_sleep)
    client = ScriptedClient([_rate_limited(), (["should never be reached"], None)])
    received: list[str] = []

    with pytest.raises(asyncio.CancelledError):
        await _collect(GeminiProvider(client=client), received)

    assert received == []
    assert client.models.calls == 1
