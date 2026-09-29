"""The free benchmark pipeline: Groq Whisper -> NVIDIA-hosted LLM -> local Piper voice.

Offline: the LLM's HTTP traffic goes to a mock transport and the voice is a
stand-in, so these tests need no key, no network and no voice download.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import httpx
import pytest
from app.fdb import local_tts
from app.fdb import runner as fdb_runner
from livekit.agents import llm as lk_llm
from openai import AsyncOpenAI


@pytest.fixture
def nvidia_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LK_PROVIDER", "nvidia")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-a-real-key")
    for key in ("FDB_LLM_MODEL", "FDB_LLM_BASE_URL", "FDB_LLM_API_KEY_ENV", "FDB_TTS", "FDB_LLM_SEED",
                "FDB_REASONING_EFFORT", "FDB_LLM_STRICT_TOOLS"):
        monkeypatch.delenv(key, raising=False)


class FakeVoice:
    """Stands in for a loaded PiperVoice: one PCM chunk per sentence."""

    class config:
        sample_rate = 22050

    def __init__(self) -> None:
        self.threads: list[str] = []

    def synthesize(self, text: str):
        self.threads.append(threading.current_thread().name)
        for sentence in [s for s in re.split(r"(?<=[.!?])\s+", text) if s]:
            yield type("Chunk", (), {"audio_int16_bytes": b"\x01\x00" * (len(sentence) * 100)})()


def test_nvidia_components_are_whisper_nvidia_llm_and_local_voice(nvidia_env, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setattr(fdb_runner, "build_tts", lambda: "piper-voice")
    components = fdb_runner.resolve_session_components()
    assert set(components) == {"stt", "llm", "tts", "vad"}
    assert components["tts"] == "piper-voice"
    llm = components["llm"]
    assert llm.model == "openai/gpt-oss-20b"
    assert str(llm._client.base_url) == "https://integrate.api.nvidia.com/v1/"
    assert llm._opts.temperature == 0.0 and llm._opts.extra_body == {"seed": 7}
    assert llm._opts.reasoning_effort == "low"  # what the Groq pipeline was tuned with
    assert llm._strict_tool_schema is False
    assert "GEMINI_API_KEY" not in os.environ


def test_the_model_and_endpoint_are_settings(nvidia_env, monkeypatch) -> None:
    monkeypatch.setenv("FDB_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("FDB_LLM_BASE_URL", "https://api.cerebras.ai/v1")
    monkeypatch.setenv("FDB_LLM_API_KEY_ENV", "CEREBRAS_API_KEY")
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test")
    llm = fdb_runner.build_llm()
    assert llm.model == "nvidia/nemotron-3-super-120b-a12b"
    assert str(llm._client.base_url) == "https://api.cerebras.ai/v1/"
    assert not isinstance(llm._opts.reasoning_effort, str)  # only gpt-oss models get it


def test_a_missing_key_names_the_variable(nvidia_env, monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY")
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is not set"):
        fdb_runner.build_llm()


def test_voice_choice_per_pipeline(nvidia_env, monkeypatch) -> None:
    monkeypatch.setattr(local_tts, "PiperTTS", lambda: "piper-voice")
    assert fdb_runner.build_tts() == "piper-voice"  # nvidia: local voice by default
    monkeypatch.setenv("FDB_TTS", "orpheus")
    assert fdb_runner.build_tts().model == "canopylabs/orpheus-v1-english"
    monkeypatch.setenv("LK_PROVIDER", "groq")
    monkeypatch.delenv("FDB_TTS")
    assert fdb_runner.build_tts().model == "canopylabs/orpheus-v1-english"  # groq: unchanged default
    monkeypatch.setenv("FDB_TTS", "piper")
    assert fdb_runner.build_tts() == "piper-voice"


async def test_the_request_carries_seed_effort_and_non_strict_tools(nvidia_env, tmp_path) -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}]}
        return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
                              headers={"content-type": "text/event-stream"})

    llm = fdb_runner.build_llm()
    llm._client = AsyncOpenAI(api_key="x", base_url=fdb_runner.NVIDIA_BASE_URL,
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    ctx = fdb_runner.create_fdb_runner_context("room", telemetry_path=str(tmp_path / "t.log"))
    chat = lk_llm.ChatContext.empty()
    chat.add_message(role="user", content="find hiking boots under 80 dollars")
    async with llm.chat(chat_ctx=chat, tools=ctx.tools) as stream:
        text = "".join([c.delta.content or "" async for c in stream if c.delta])
    await ctx.runtime.shutdown()
    assert text == "Done."
    body = captured[0]
    assert body["model"] == "openai/gpt-oss-20b" and body["seed"] == 7
    assert body["reasoning_effort"] == "low" and body["temperature"] == 0
    tools = {t["function"]["name"]: t["function"] for t in body["tools"]}
    assert len(tools) == 12 and not tools["search_products"].get("strict")
    # optional arguments may be sent as null, as on the Groq path
    assert {"type": "null"} in tools["search_products"]["parameters"]["properties"]["max_price"]["anyOf"]


def test_nvidia_pipeline_can_be_built_inside_a_job_thread(nvidia_env, monkeypatch) -> None:
    monkeypatch.setattr(fdb_runner, "build_tts", lambda: "piper-voice")
    outcome: dict[str, object] = {}

    def job() -> None:
        try:
            outcome["components"] = sorted(fdb_runner.resolve_session_components())
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    t = threading.Thread(target=job)
    t.start()
    t.join(timeout=60)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["components"] == ["llm", "stt", "tts", "vad"]


async def test_piper_voice_streams_pcm_rendered_off_the_event_loop() -> None:
    voice = FakeVoice()
    tts = local_tts.PiperTTS(voice=voice)
    assert tts.sample_rate == 22050 and tts.provider == "Piper (local)"
    frames = [ev.frame async for ev in tts.synthesize("Your order is on its way. It arrives tomorrow.")]
    samples = sum(f.samples_per_channel for f in frames)
    rendered = len("Your order is on its way.") * 100 + len("It arrives tomorrow.") * 100
    # every rendered sample is played; LiveKit may pad the last frame (a few ms)
    assert rendered <= samples < rendered + 22050 * 0.05
    assert all(f.sample_rate == 22050 and f.num_channels == 1 for f in frames)
    assert voice.threads and voice.threads[0] != threading.main_thread().name  # rendered in a worker thread
    await tts.aclose()


def test_voice_download_is_pinned_and_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = {"a.onnx": b"model-bytes", "a.onnx.json": b"{}"}
    monkeypatch.setattr(local_tts, "VOICE_NAME", "a")
    monkeypatch.setattr(local_tts, "VOICE_FILES", {n: hashlib.sha256(b).hexdigest() for n, b in good.items()})
    served = dict(good)
    fetched: list[str] = []

    def fake_retrieve(url: str, path: Path) -> None:
        name = url.rsplit("/", 1)[1]
        fetched.append(name)
        Path(path).write_bytes(served[name])

    monkeypatch.setattr(local_tts.urllib.request, "urlretrieve", fake_retrieve)
    assert local_tts.ensure_voice(tmp_path) == tmp_path / "a.onnx"
    assert fetched == ["a.onnx", "a.onnx.json"]
    local_tts.ensure_voice(tmp_path)
    assert len(fetched) == 2  # verified files are not downloaded again

    (tmp_path / "a.onnx").write_bytes(b"corrupted")
    served["a.onnx"] = b"tampered"
    with pytest.raises(RuntimeError, match="does not match the pinned"):
        local_tts.ensure_voice(tmp_path)
    assert not (tmp_path / "a.onnx.part").exists()


def test_rule_examples_never_quote_the_benchmark_data() -> None:
    """The rules' made-up examples must not echo wording from the test items."""
    data = fdb_runner._FDB_V3_DIR / "benchmark_data_v2.json"
    if not data.exists():
        pytest.skip("FDB-v3 checkout not present")
    blob = data.read_text(encoding="utf-8").lower()
    examples = re.findall(r'"([^"]{2,40})"', fdb_runner.ARGUMENT_RULES)
    assert len(examples) >= 8
    echoed = [e for e in examples
              if re.search(rf"(?<![a-z0-9]){re.escape(e.lower())}(?![a-z0-9])", blob)]
    assert not echoed, echoed


def test_runs_record_the_pipeline_that_actually_ran(nvidia_env, monkeypatch) -> None:
    from app.fdb import pipeline_config

    desc = pipeline_config.describe()
    assert desc["llm_model"] == "openai/gpt-oss-20b" and desc["llm_endpoint"] == fdb_runner.NVIDIA_BASE_URL
    assert desc["llm_seed"] == 7 and desc["tts"].startswith("piper ")
    assert desc["stt"] == "groq whisper-large-v3-turbo"
    monkeypatch.setenv("FDB_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    assert pipeline_config.describe()["llm_model"] == "nvidia/nemotron-3-super-120b-a12b"
    monkeypatch.setenv("LK_PROVIDER", "groq")
    desc = pipeline_config.describe()
    assert desc["llm_model"] == "openai/gpt-oss-120b" and desc["tts"].startswith("groq canopylabs/orpheus")
