"""End-to-end turns over the real /ws endpoint, the way the browser drives it.

Regression coverage for a failure no lower-level test could see: with a
GEMINI_API_KEY in the repository-root .env (the LiveKit worker's credential
file), app/config.py loaded it into every process, the browser agent silently
switched from the offline engine to Gemini, and once that key's quota was spent
every turn ended as "Recovered from an error." with zero tokens streamed.

No network access and no API key: the default provider is the deterministic
engine, and the failure case uses a provider that raises locally.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import app.runtime as runtime_module
from app.main import app
from app.providers.base import GenerationRequest
from app.providers.mock import MockProvider
from fastapi.testclient import TestClient

_SERVER_DIR = Path(__file__).resolve().parents[1]
_MAX_FRAMES = 5000


def _run_turn(ws: Any, text: str) -> list[dict[str, Any]]:
    """Send a finished utterance and collect frames until the turn settles."""
    ws.send_json({"t": "final", "text": text, "modality": "text"})
    frames: list[dict[str, Any]] = []
    seen_user_message = False
    for _ in range(_MAX_FRAMES):
        frame = ws.receive_json()
        frames.append(frame)
        if frame.get("t") == "message" and frame.get("role") == "user":
            seen_user_message = True
        if seen_user_message and frame.get("t") == "stage" and frame["stage"] == "idle":
            return frames
    raise AssertionError(f"turn for {text!r} never returned to idle")


def _connect(client: TestClient) -> tuple[Any, dict[str, Any]]:
    ws = client.websocket_connect("/ws")
    ws.__enter__()
    ready = ws.receive_json()
    assert ready["t"] == "ready"
    connected = ws.receive_json()
    assert (connected["t"], connected["stage"]) == ("stage", "idle")
    ws.send_json({"t": "config", "token_delay_ms": 1})
    assert ws.receive_json()["detail"] == "Settings applied."
    return ws, ready


def test_browser_flow_streams_real_answers_with_the_offline_engine() -> None:
    with TestClient(app) as client:
        ws, ready = _connect(client)
        try:
            assert ready["provider"] == "local-deterministic"
            assert client.get("/api/health").json()["provider"] == "local-deterministic"

            for text in ["I want to travel to Mumbai on friday", "what are the baggage limits on each cabin"]:
                frames = _run_turn(ws, text)
                tokens = [f["text"] for f in frames if f.get("t") == "token"]
                errors = [f for f in frames if f.get("t") == "error"]
                agent = [f for f in frames if f.get("t") == "message" and f.get("role") == "agent"]
                first_token = [f for f in frames if f.get("t") == "metric" and f["name"] == "time_to_first_token"]

                assert errors == [], errors
                assert tokens, f"no tokens streamed for {text!r}"
                assert first_token and first_token[0]["value"] >= 0
                assert len(agent) == 1 and agent[0]["status"] == "complete"
                assert agent[0]["content"] == "".join(tokens)
                assert frames[-1]["detail"] == "Your move."
                assert not any("Recovered from an error" in f.get("detail", "") for f in frames)

            baggage_answer = agent[0]["content"].lower()
            assert "cabin" in baggage_answer or "kg" in baggage_answer
        finally:
            ws.__exit__(None, None, None)


class _QuotaExhaustedProvider(MockProvider):
    """Plans like the offline engine, then fails the way a spent Gemini quota does."""

    name = "gemini"

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        raise RuntimeError("Gemini streaming error: 429 RESOURCE_EXHAUSTED. You exceeded your current quota.")
        yield ""  # pragma: no cover - makes this an async generator


def test_provider_failure_is_visible_in_stage_detail_and_server_log(monkeypatch, caplog) -> None:
    monkeypatch.setattr(runtime_module, "build_provider", _QuotaExhaustedProvider)
    caplog.set_level(logging.ERROR, logger="app.runtime")

    with TestClient(app) as client:
        ws, ready = _connect(client)
        try:
            assert ready["provider"] == "gemini"
            frames = _run_turn(ws, "what are the baggage limits on each cabin")
        finally:
            ws.__exit__(None, None, None)

    errors = [f["message"] for f in frames if f.get("t") == "error"]
    assert errors == ["RuntimeError: Gemini streaming error: 429 RESOURCE_EXHAUSTED. You exceeded your current quota."]
    assert not [f for f in frames if f.get("t") == "token"]
    assert frames[-1]["detail"].startswith("Recovered from an error - RuntimeError: Gemini streaming error: 429")

    logged = [r for r in caplog.records if r.name == "app.runtime" and r.exc_info]
    assert logged, "the failed turn left no traceback in the server log"
    assert "provider=gemini" in logged[0].getMessage()
    assert "RESOURCE_EXHAUSTED" in str(logged[0].exc_info[1])


def test_web_backend_does_not_load_dotenv_but_livekit_worker_does() -> None:
    """Importing the web app must not read any .env; the LiveKit worker still must.

    Runs outside pytest (the loaders skip themselves under pytest), with
    load_dotenv replaced by a counter so no real file is read.
    """
    code = (
        "import dotenv\n"
        "calls = []\n"
        "dotenv.load_dotenv = lambda *a, **k: calls.append(1) or True\n"
        "import app.main\n"
        "web = len(calls)\n"
        "import app.livekit_worker\n"
        "print(web, len(calls) - web)\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTEST_CURRENT_TEST", "PYTEST_VERSION", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"}
    }
    env["PYTHONPATH"] = str(_SERVER_DIR)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=_SERVER_DIR, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    web_loads, worker_loads = map(int, result.stdout.split()[-2:])
    assert web_loads == 0, "the web backend loaded a .env file"
    assert worker_loads >= 1, "the LiveKit worker no longer loads its .env"


_BURST_ANSWER = (
    "A one-way economy flight from Delhi to Hyderabad usually costs between four and "
    "nine thousand rupees, depending on the airline, the day and how early you book it."
)


class _BurstProvider(MockProvider):
    """Delivers the whole answer in one chunk, the way a fast hosted model (Groq)
    does, then holds the stream open like an HTTP response still in flight."""

    name = "openai-compatible"
    streams: list[dict[str, bool]] = []

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        state = {"closed": False}
        _BurstProvider.streams.append(state)
        try:
            yield _BURST_ANSWER
            await asyncio.Event().wait()
        finally:
            state["closed"] = True


def test_burst_model_is_paced_word_by_word_and_interruptible(monkeypatch) -> None:
    _BurstProvider.streams = []
    monkeypatch.setattr(runtime_module, "build_provider", _BurstProvider)

    with TestClient(app) as client:
        ws, ready = _connect(client)
        try:
            ws.send_json({"t": "config", "token_delay_ms": 25})
            assert ws.receive_json()["detail"] == "Settings applied."
            ws.send_json({"t": "final", "text": "how much is a flight from delhi to hyderabad", "modality": "text"})
            tokens: list[str] = []
            for _ in range(_MAX_FRAMES):
                frame = ws.receive_json()
                if frame.get("t") == "token":
                    tokens.append(frame["text"])
                    assert len(frame["text"].split()) <= 1, "model output arrived as one burst, not word by word"
                    if len(tokens) == 5:
                        break
            ws.send_json({"t": "interrupt", "reason": "barge_in"})
            after: list[dict[str, Any]] = []
            for _ in range(_MAX_FRAMES):
                frame = ws.receive_json()
                after.append(frame)
                if frame.get("t") == "metric" and frame["name"] == "time_to_yield":
                    break
            ws.send_json({"t": "ping"})  # everything the turn emitted arrives before the pong
            for _ in range(_MAX_FRAMES):
                frame = ws.receive_json()
                after.append(frame)
                if frame.get("t") == "stage" and frame.get("detail") == "pong":
                    break
        finally:
            ws.__exit__(None, None, None)

    late_tokens = [f for f in after if f.get("t") == "token"]
    checkpoints = [f for f in after if f.get("t") == "checkpoint"]
    words = _BURST_ANSWER.split()
    assert tokens == [w + " " for w in words[:4]] + [words[4] + " "]  # one frame per word, not one burst
    assert len(late_tokens) <= 1, "tokens kept streaming after the barge-in"
    shown = "".join(tokens + [f["text"] for f in late_tokens])
    assert shown and len(shown) < len(_BURST_ANSWER)
    assert len(checkpoints) == 1
    assert f"{len(shown.split())} words kept" in checkpoints[0]["summary"]  # exactly what was shown
    assert _BurstProvider.streams and _BurstProvider.streams[0]["closed"], "model stream left open after barge-in"


def test_speaking_pace_setting_controls_streaming_speed(monkeypatch) -> None:
    import time

    monkeypatch.setattr(runtime_module, "build_provider", _BurstProvider)

    def stream_duration(delay_ms: int) -> tuple[float, int]:
        _BurstProvider.streams = []
        with TestClient(app) as client:
            ws, _ = _connect(client)
            try:
                ws.send_json({"t": "config", "token_delay_ms": delay_ms})
                assert ws.receive_json()["detail"] == "Settings applied."
                ws.send_json({"t": "final", "text": "delhi to hyderabad fare", "modality": "text"})
                count, started = 0, None
                for _ in range(_MAX_FRAMES):
                    frame = ws.receive_json()
                    if frame.get("t") == "token":
                        assert len(frame["text"].split()) <= 1, "model output arrived as one burst, unpaced"
                        started = started or time.monotonic()
                        count += 1
                        if count == 12:
                            return time.monotonic() - started, count
                raise AssertionError("stream never produced 12 tokens")
            finally:
                ws.send_json({"t": "interrupt", "reason": "stop"})
                ws.__exit__(None, None, None)

    fast, _ = stream_duration(0)
    slow, _ = stream_duration(40)
    assert slow >= 0.25, f"11 gaps at 40 ms/token took only {slow:.3f}s - pace setting ignored"
    assert fast < slow
