"""Tests for provider factory build_provider()."""

from __future__ import annotations

import pytest
from app.config import Settings
from app.providers import (
    GeminiProvider,
    MockProvider,
    OpenAICompatProvider,
    build_provider,
)


def test_build_provider_default_is_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.providers.settings", Settings(gemini_api_key=None, llm_api_key=None))
    provider = build_provider()
    assert isinstance(provider, MockProvider)
    assert provider.name == "local-deterministic"


def test_build_provider_gemini_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.providers.settings", Settings(gemini_api_key="fake-gemini-key", llm_api_key=None))
    provider = build_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"


def test_build_provider_openai_when_llm_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.providers.settings", Settings(gemini_api_key=None, llm_api_key="fake-openai-key"))
    provider = build_provider()
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.name == "openai-compatible"


def test_build_provider_gemini_takes_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.providers.settings",
        Settings(gemini_api_key="fake-gemini-key", llm_api_key="fake-openai-key"),
    )
    provider = build_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"
