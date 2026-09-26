"""Unit tests for standalone LiveKit worker and RealtimeModel configuration."""

from __future__ import annotations

import pytest
from app.config import Settings
from app.livekit_worker import build_realtime_model, create_agent_session
from livekit.agents import Agent, AgentSession
from livekit.plugins.google.realtime import RealtimeModel


def test_livekit_worker_config_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://livekit.example.com")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-lk-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-lk-secret")
    monkeypatch.setenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")

    s = Settings()
    assert s.livekit_url == "wss://livekit.example.com"
    assert s.livekit_api_key == "test-lk-key"
    assert s.livekit_api_secret == "test-lk-secret"
    assert s.gemini_live_model == "gemini-3.8-live"


def test_build_realtime_model_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.livekit_worker.settings", Settings(gemini_api_key="fake-gemini-key"))
    model = build_realtime_model(instructions="Test instruction")
    assert isinstance(model, RealtimeModel)


def test_build_realtime_model_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.livekit_worker.settings", Settings(gemini_api_key=None))
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not configured"):
        build_realtime_model(api_key=None)


@pytest.mark.asyncio
async def test_create_agent_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.livekit_worker.settings", Settings(gemini_api_key="fake-gemini-key"))
    agent, session = create_agent_session(instructions="Test instructions")
    assert isinstance(agent, Agent)
    assert isinstance(session, AgentSession)
