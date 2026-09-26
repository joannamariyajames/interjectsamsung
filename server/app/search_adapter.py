"""Provider-neutral external search adapter.

Isolates external search HTTP details (Serper, Tavily, generic) behind a clean boundary,
normalizing external search results into the standard doc/result schema.
"""

from __future__ import annotations

import logging
from typing import Any
import httpx

logger = logging.getLogger(__name__)

SERPER_DEFAULT_URL = "https://google.serper.dev/search"
TAVILY_DEFAULT_URL = "https://api.tavily.com/search"


async def fetch_external_search(
    query: str,
    *,
    api_key: str | None = None,
    api_url: str | None = None,
    provider: str = "serper",
    timeout_s: float = 5.0,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch external web search results and normalize them into result dicts.

    Each result dict contains:
        doc_id: str (e.g. "web#0")
        title: str
        snippet: str
        score: float
        link: str
    """
    if not api_key or not api_key.strip():
        return []

    provider_normalized = provider.strip().lower()

    if provider_normalized == "serper":
        url = api_url or SERPER_DEFAULT_URL
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        json_payload = {"q": query}
    elif provider_normalized == "tavily":
        url = api_url or TAVILY_DEFAULT_URL
        headers = {"Content-Type": "application/json"}
        json_payload = {"api_key": api_key, "query": query}
    else:
        # Generic HTTP POST endpoint fallback
        url = api_url or SERPER_DEFAULT_URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        json_payload = {"q": query, "query": query}

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout_s)
        close_client = True

    try:
        response = await client.post(url, headers=headers, json=json_payload)
        if response.status_code != 200:
            logger.warning("External search HTTP %s from %s", response.status_code, url)
            return []
        data = response.json()
    except Exception as exc:
        logger.warning("External search request failed: %s", exc)
        return []
    finally:
        if close_client:
            await client.aclose()

    return parse_search_response(data, provider=provider_normalized)


def parse_search_response(data: dict[str, Any], provider: str = "serper") -> list[dict[str, Any]]:
    """Parse raw JSON search payload into normalized result dicts."""
    results: list[dict[str, Any]] = []

    if not isinstance(data, dict):
        return []

    raw_items = data.get("organic") or data.get("results") or data.get("items") or []
    if isinstance(raw_items, list):
        for idx, item in enumerate(raw_items[:3]):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or "Web Search Result"
            snippet = item.get("snippet") or item.get("content") or item.get("description") or ""
            link = item.get("link") or item.get("url") or ""
            doc_id = f"web#{idx}"
            score = round(1.0 - (idx * 0.1), 2)
            results.append(
                {
                    "doc_id": doc_id,
                    "title": str(title),
                    "snippet": str(snippet),
                    "score": score,
                    "link": str(link),
                }
            )

    return results
