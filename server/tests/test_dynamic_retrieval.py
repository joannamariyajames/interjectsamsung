"""Tests for external search adapter, fallback behavior, and provider neutrality."""

from __future__ import annotations

import pytest
import httpx
from app.config import Settings
from app.search_adapter import fetch_external_search, parse_search_response
from app.tools import _search_corpus


def test_parse_search_response_serper_organic() -> None:
    data = {
        "organic": [
            {
                "title": "Python Asyncio Docs",
                "snippet": "Event loop management in Python asyncio.",
                "link": "https://docs.python.org/3/library/asyncio.html",
            },
            {
                "title": "Asyncio Tutorial",
                "snippet": "Understanding coroutines and tasks.",
                "link": "https://realpython.com/async-io-python/",
            },
        ]
    }
    results = parse_search_response(data, provider="serper")
    assert len(results) == 2
    assert results[0]["doc_id"] == "web#0"
    assert results[0]["title"] == "Python Asyncio Docs"
    assert results[0]["snippet"] == "Event loop management in Python asyncio."
    assert results[0]["score"] == 1.0
    assert results[1]["doc_id"] == "web#1"
    assert results[1]["score"] == 0.9


def test_parse_search_response_tavily_results() -> None:
    data = {
        "results": [
            {
                "title": "Tavily Search Result",
                "content": "Deep web search result for open domain queries.",
                "url": "https://tavily.com/result",
            }
        ]
    }
    results = parse_search_response(data, provider="tavily")
    assert len(results) == 1
    assert results[0]["doc_id"] == "web#0"
    assert results[0]["title"] == "Tavily Search Result"
    assert results[0]["snippet"] == "Deep web search result for open domain queries."


def test_parse_search_response_empty_or_invalid() -> None:
    assert parse_search_response({}) == []
    assert parse_search_response({"organic": []}) == []
    assert parse_search_response("not a dict") == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fetch_external_search_no_api_key() -> None:
    results = await fetch_external_search("any query", api_key=None)
    assert results == []
    results_empty_str = await fetch_external_search("any query", api_key="   ")
    assert results_empty_str == []


@pytest.mark.asyncio
async def test_fetch_external_search_serper_http_mock() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-API-KEY") == "test-serper-key"
        assert str(request.url) == "https://google.serper.dev/search"
        return httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Serper Open Domain Result",
                        "snippet": "Informative snippet about open domain topic.",
                        "link": "https://example.com/item",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        results = await fetch_external_search(
            "open domain topic query",
            api_key="test-serper-key",
            provider="serper",
            client=client,
        )
        assert len(results) == 1
        assert results[0]["doc_id"] == "web#0"
        assert results[0]["title"] == "Serper Open Domain Result"


@pytest.mark.asyncio
async def test_fetch_external_search_http_error_graceful_fallback() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "Internal Server Error"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        results = await fetch_external_search(
            "query causing error",
            api_key="test-key",
            provider="serper",
            client=client,
        )
        assert results == []


@pytest.mark.asyncio
async def test_search_corpus_local_hits_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Query matching local corpus (e.g. baggage limit)
    res = await _search_corpus("baggage limits")
    assert len(res["results"]) > 0
    assert any("baggage" in r["snippet"].lower() or "cabin" in r["snippet"].lower() for r in res["results"])
    assert res["results"][0]["doc_id"].startswith(("flights", "policy", "support", "hotels"))


@pytest.mark.asyncio
async def test_search_corpus_fallback_to_external_on_zero_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.tools.settings",
        Settings(
            search_api_key="mock-key",
            search_provider="serper",
        ),
    )

    async def mock_fetch_external(
        query: str,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        provider: str = "serper",
        timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> list[dict]:
        return [
            {
                "doc_id": "web#0",
                "title": "External Result for Off Corpus Topic",
                "snippet": "Detailed snippet about open domain topic.",
                "score": 1.0,
            }
        ]

    monkeypatch.setattr("app.tools.fetch_external_search", mock_fetch_external)

    res = await _search_corpus("zqx99 vkw88 plm77 quantum_xyz_99")
    assert len(res["results"]) == 1
    assert res["results"][0]["doc_id"] == "web#0"
    assert res["results"][0]["title"] == "External Result for Off Corpus Topic"


@pytest.mark.asyncio
async def test_search_corpus_zero_hits_unconfigured_external_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.tools.settings", Settings(search_api_key=None))

    query = "zqx99 vkw88 plm77 quantum_xyz_99"
    res = await _search_corpus(query)
    assert res["query"] == query
    assert res["results"] == []
