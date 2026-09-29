"""The benchmark agent's spoken acknowledgement: queued at the end of the user's
turn, after a silent lead-in, outside the chat context. Offline: a stand-in
session and voice."""

from __future__ import annotations

import pytest
from app.fdb import local_tts
from app.fdb import runner as fdb_runner


class Voice:
    class config:
        sample_rate = 22050

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str):
        self.calls += 1
        yield type("Chunk", (), {"audio_int16_bytes": b"\x05\x00" * 4410})()  # 0.2 s of sound


class Session:
    def __init__(self, tts) -> None:
        self.tts = tts
        self.said: list[tuple[str, object, dict]] = []

    def say(self, text, *, audio=None, **kwargs):
        self.said.append((text, audio, kwargs))


class Message:
    def __init__(self, text: str) -> None:
        self.text_content = text


def _leading_silence_s(frames) -> float:
    total = 0
    for f in frames:
        if any(bytes(f.data)):
            break
        total += f.samples_per_channel
    return total / 22050


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch):
    for key in ("FDB_ACK_TEXT", "FDB_ACK_DELAY_S"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FDB_ACK", "1")  # opt-in
    voice = Voice()
    session = Session(local_tts.PiperTTS(voice=voice))
    monkeypatch.setattr(fdb_runner.InterjectVoiceAgent, "session", property(lambda self: session))
    a = fdb_runner.InterjectVoiceAgent()
    a.test_session, a.test_voice = session, voice
    return a


async def test_an_acknowledgement_is_queued_when_the_turn_ends(agent) -> None:
    await agent.on_user_turn_completed(None, Message("track my order please"))
    ((text, audio, kwargs),) = agent.test_session.said
    assert text == "Sure, one moment."
    assert kwargs == {"add_to_chat_ctx": False, "allow_interruptions": True}  # never mistaken for an answer
    frames = [f async for f in audio]
    lead = _leading_silence_s(frames)
    assert 0.99 <= lead <= 1.01  # the default 1 s lead-in: a resumed user interrupts it unheard
    spoken = sum(f.samples_per_channel for f in frames if any(bytes(f.data)))
    assert spoken >= 4410  # then the words


async def test_the_voice_renders_it_once_per_conversation(agent) -> None:
    await agent.on_user_turn_completed(None, Message("first"))
    await agent.on_user_turn_completed(None, Message("second"))
    assert len(agent.test_session.said) == 2 and agent.test_voice.calls == 1


async def test_the_lead_in_and_text_are_settings(agent, monkeypatch) -> None:
    monkeypatch.setenv("FDB_ACK_DELAY_S", "0.4")
    monkeypatch.setenv("FDB_ACK_TEXT", "Okay, checking.")
    await agent.on_user_turn_completed(None, Message("hi"))
    text, audio, _ = agent.test_session.said[0]
    lead = _leading_silence_s([f async for f in audio])
    assert text == "Okay, checking." and 0.39 <= lead <= 0.41


@pytest.mark.parametrize("case", ["disabled", "no_tts", "empty"])
async def test_no_acknowledgement_when_off_voiceless_or_empty(agent, monkeypatch, case) -> None:
    if case == "disabled":
        monkeypatch.setenv("FDB_ACK", "0")
    if case == "no_tts":
        agent.test_session.tts = None  # the text replay has no voice
    await agent.on_user_turn_completed(None, Message("" if case == "empty" else "hello"))
    assert agent.test_session.said == []


async def test_the_acknowledgement_is_off_by_default(agent, monkeypatch) -> None:
    monkeypatch.delenv("FDB_ACK")
    await agent.on_user_turn_completed(None, Message("track my order please"))
    assert agent.test_session.said == []
