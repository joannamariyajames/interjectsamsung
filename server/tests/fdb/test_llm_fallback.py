"""The benchmark agent's backup LLM (FDB_LLM_FALLBACK=1) - offline, mock HTTP only.

NVIDIA first for the benchmark (Groq only when NVIDIA fails), Groq first for
demos (NVIDIA once Groq's limit is reached); every answered request is logged
with the model that served it.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from app.fdb import bench_utils, pipeline_config
from app.fdb import runner as fdb_runner
from livekit.agents import llm as lk_llm
from openai import AsyncOpenAI


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-a-real-key")
    monkeypatch.setenv("FDB_LLM_FALLBACK", "1")
    for key in ("FDB_LLM_MODEL", "GROQ_LLM_MODEL", "FDB_LLM_BASE_URL", "FDB_LLM_API_KEY_ENV",
                "FDB_LLM_ATTEMPT_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)


def _models(adapter) -> list[str]:
    return [i.model for i in adapter._llm_instances]


def test_benchmark_order_is_nvidia_then_groq(keys, monkeypatch) -> None:
    monkeypatch.setenv("LK_PROVIDER", "nvidia")
    adapter = fdb_runner.build_llm()
    assert isinstance(adapter, lk_llm.FallbackAdapter)
    assert _models(adapter) == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    nvidia, groq = adapter._llm_instances
    assert str(nvidia._client.base_url) == "https://integrate.api.nvidia.com/v1/"
    assert nvidia._opts.extra_body == {"seed": 7} and "groq" in str(groq._client.base_url)
    assert adapter._attempt_timeout == 10.0 and adapter._max_retry_per_llm == 0


def test_demo_order_is_groq_then_nvidia(keys, monkeypatch) -> None:
    monkeypatch.setenv("LK_PROVIDER", "groq")
    monkeypatch.setenv("FDB_LLM_ATTEMPT_TIMEOUT", "6")
    adapter = fdb_runner.build_llm()
    assert _models(adapter) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert adapter._attempt_timeout == 6.0


def test_without_the_setting_a_single_model_answers(keys, monkeypatch) -> None:
    monkeypatch.setenv("LK_PROVIDER", "nvidia")
    monkeypatch.delenv("FDB_LLM_FALLBACK")
    llm = fdb_runner.build_llm()
    assert not isinstance(llm, lk_llm.FallbackAdapter) and llm.model == "openai/gpt-oss-20b"
    assert pipeline_config.describe()["llm_fallback"] == "off"


def test_a_missing_backup_key_runs_on_the_primary_alone(keys, monkeypatch) -> None:
    monkeypatch.setenv("LK_PROVIDER", "groq")
    monkeypatch.delenv("NVIDIA_API_KEY")
    llm = fdb_runner.build_llm()
    assert not isinstance(llm, lk_llm.FallbackAdapter) and llm.model == "openai/gpt-oss-120b"


def test_the_run_config_records_the_order(keys, monkeypatch) -> None:
    monkeypatch.setenv("LK_PROVIDER", "nvidia")
    desc = pipeline_config.describe()
    assert desc["llm_model"] == "openai/gpt-oss-20b"  # the primary, as before
    assert desc["llm_fallback"] == {
        "order": ["nvidia openai/gpt-oss-20b", "groq openai/gpt-oss-120b"],
        "attempt_timeout_s": 10.0,
    }


def _mock_client(base_url: str, handler) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="x", base_url=base_url,
                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _answer(text: str):
    def handler(request: httpx.Request) -> httpx.Response:
        chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
        return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
                              headers={"content-type": "text/event-stream"})
    return handler


async def test_a_failing_nvidia_request_is_answered_by_groq_and_logged(keys, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LK_PROVIDER", "nvidia")
    log = tmp_path / "served.log"
    monkeypatch.setattr(fdb_runner, "LLM_SERVED_LOG", log)
    adapter = fdb_runner.build_llm()
    nvidia, groq = adapter._llm_instances
    nvidia_calls: list[str] = []

    def unavailable(request: httpx.Request) -> httpx.Response:
        nvidia_calls.append(request.url.host)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    nvidia._client = _mock_client(fdb_runner.NVIDIA_BASE_URL, unavailable)
    groq._client = _mock_client("https://api.groq.com/openai/v1", _answer("From Groq."))

    chat = lk_llm.ChatContext.empty()
    chat.add_message(role="user", content="what is the weather like")
    async with adapter.chat(chat_ctx=chat) as stream:
        text = "".join([c.delta.content or "" async for c in stream if c.delta])
    # NVIDIA is re-checked in the background; let that finish (it fails again) before the test ends
    await asyncio.gather(*(s.recovering_task for s in adapter._status if s.recovering_task), return_exceptions=True)
    await adapter.aclose()

    assert text == "From Groq." and nvidia_calls
    assert [s.available for s in adapter._status] == [False, True]  # the next request goes straight to Groq
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["model"] for r in records] == ["groq openai/gpt-oss-120b"]
    assert bench_utils.llm_served_counts(log) == {"groq openai/gpt-oss-120b": 1}


def test_served_counts_skip_unreadable_lines(tmp_path) -> None:
    log = tmp_path / "served.log"
    log.write_text('{"model": "nvidia a"}\nnot json\n{"model": "nvidia a"}\n{"model": "groq b"}\n', encoding="utf-8")
    assert bench_utils.llm_served_counts(log) == {"nvidia a": 2, "groq b": 1}
    assert bench_utils.llm_served_counts(tmp_path / "missing.log") == {}
