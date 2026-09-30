"""Offline demo mode: with no model key, the local engine answers only what the
demo agency's knowledge base covers, and says so plainly otherwise."""

from __future__ import annotations

import pytest
from app.providers.base import GenerationRequest
from app.providers.mock import OFFLINE_NOTE, MockProvider
from app.retrieval import corpus
from app.runtime import _hits_to_evidence


def answer(question: str, **extra) -> str:
    evidence = _hits_to_evidence(corpus.search(question))
    return MockProvider()._compose(GenerationRequest(goal=question, utterance=question, evidence=evidence, **extra))


@pytest.mark.parametrize("question", [
    "what is 2 plus 2",               # shares one word with a cabin-class passage
    "what time is it",                # ...with hotel check-in times
    "what is the weather in delhi today",  # ...with a route list
    "tell me a joke",                 # matches nothing at all
])
def test_off_topic_questions_get_the_offline_notice_not_a_passage(question) -> None:
    text = answer(question)
    assert text.startswith(OFFLINE_NOTE)
    assert f'"{question}"' in text and "Groq key" in text
    assert "[" not in text  # no passage cited as if it answered the question
    assert "general knowledge" not in text  # this engine has none to offer


@pytest.mark.parametrize("question, doc", [
    ("What are the baggage limits on each cabin?", "flights#2"),
    ("What happens to my refund if I cancel?", "policy#10"),
    ("what time is check-in", "hotels#6"),
    ("what's the Harbour House like", "hotels#5"),  # no travel word, but the passage clearly matches
])
def test_questions_the_knowledge_base_covers_are_still_answered_from_it(question, doc) -> None:
    text = answer(question)
    assert OFFLINE_NOTE not in text and f"[{doc}]" in text


def test_a_resumed_answer_is_never_replaced_by_the_notice() -> None:
    text = answer("what time is it", resume_from="Standard check-in is 14:00 and")
    assert OFFLINE_NOTE not in text and text.startswith("Picking up where I stopped")
