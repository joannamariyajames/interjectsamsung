"""Spoken control phrases: pausing the agent is not a request."""

import pytest
from app.speech_control import is_hold, is_resume


@pytest.mark.parametrize("text", ["hold on", "Hold on.", "stop", "Stop!", "wait", "wait a second", "hang on",
                                  "one sec", "just a moment", "okay stop", "stop talking", "shh", "be quiet",
                                  "hold on please", "that's enough"])
def test_hold_phrases(text: str) -> None:
    assert is_hold(text)


@pytest.mark.parametrize("text", ["stop for fuel", "wait, make it Ajmer", "hold on, take me to Agra",
                                  "stop the navigation", "what is the waiting time", "going", "go on"])
def test_requests_are_not_holds(text: str) -> None:
    assert not is_hold(text)


@pytest.mark.parametrize("text", ["go on", "Go on.", "okay continue", "carry on", "keep going", "yes go ahead"])
def test_resume_phrases(text: str) -> None:
    assert is_resume(text)


@pytest.mark.parametrize("text", ["hold on", "where are we going", "what should I continue with"])
def test_not_resume(text: str) -> None:
    assert not is_resume(text)
