"""Resuming after a barge-in must continue the answer, never restart it.

Seen with a real hosted model (Groq): asked to continue, it re-answered from
the top. Two layers stop that reaching the user: the OpenAI-compatible request
puts the interrupted words in as the assistant's own turn, and the runtime
drops any restart of already-said words, whatever the provider does.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.providers.base import GenerationRequest
from app.providers.openai_compat import OpenAICompatProvider
from app.runtime import _drop_repeated_prefix

SAID = "For international flights, the typical **baggage** allowance is"


async def _chunks(*parts: str) -> AsyncIterator[str]:
    for part in parts:
        yield part


async def _run(already: str, *parts: str) -> str:
    out = []
    async for text in _drop_repeated_prefix(already, _chunks(*parts)):
        out.append(text)
    return "".join(out)


async def test_genuine_continuation_passes_through_unchanged() -> None:
    parts = (" 23 kg in ", "economy and ", "32 kg in business.")
    assert await _run(SAID, *parts) == "".join(parts)


async def test_restart_is_trimmed_even_when_chunks_split_words() -> None:
    restarted = ("For inter", "national flights, the typ", "ical baggage allowance is 23 kg", " in economy.")
    assert await _run(SAID, *restarted) == "23 kg in economy."


async def test_formatting_differences_do_not_hide_a_restart() -> None:
    restarted = ("**For international flights** - the typical baggage allowance is ", "23 kg.")
    assert await _run(SAID, *restarted) == "23 kg."


async def test_a_stream_that_only_repeats_adds_nothing() -> None:
    assert await _run(SAID, "For international flights, the typical") == ""


async def test_a_continuation_that_starts_like_the_answer_but_diverges_is_kept() -> None:
    parts = ("For international flights, ", "check the fare rules first.")
    assert await _run(SAID, *parts) == "".join(parts)


async def test_nothing_said_yet_means_no_filtering() -> None:
    assert await _run("", "Hello ", "there.") == "Hello there."


async def test_the_model_stream_is_closed_when_the_consumer_stops() -> None:
    closed = []

    async def endless() -> AsyncIterator[str]:
        try:
            yield " 23 kg"
            while True:
                yield " more"
        finally:
            closed.append(True)

    gen = _drop_repeated_prefix(SAID, endless())
    assert await gen.__anext__() == " 23 kg"
    await gen.aclose()
    assert closed == [True]


def test_openai_compat_resume_sends_the_partial_as_the_assistant_turn() -> None:
    request = GenerationRequest(goal="Baggage rules", utterance="go on", resume_from=SAID)
    messages = OpenAICompatProvider()._messages(request)

    assert [m["role"] for m in messages[-3:]] == ["user", "assistant", "user"]
    assert messages[-2]["content"] == SAID
    assert "Output only the remaining text" in messages[-1]["content"]
    assert SAID not in messages[-3]["content"]  # not quoted inside a user turn any more


def test_openai_compat_without_resume_ends_on_the_user_turn() -> None:
    messages = OpenAICompatProvider()._messages(GenerationRequest(goal="g", utterance="hi"))
    assert messages[-1]["role"] == "user" and "User just said: hi" in messages[-1]["content"]
    assert not any(m["role"] == "assistant" for m in messages)
