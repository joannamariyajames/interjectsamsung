"""Optional real-model provider, OpenAI-compatible.

Enabled by setting ``LLM_API_KEY``. Works against OpenAI, Together, Groq,
vLLM, Ollama's compat endpoint, or anything else speaking ``/chat/completions``.
Streaming is line-delimited SSE, and each yielded chunk is a cancellation point,
so interruption behaves identically to the local engine.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import httpx

from ..config import settings
from .base import SYSTEM, GenerationRequest


class OpenAICompatProvider:
    name = "openai-compatible"

    def plan(self, request: GenerationRequest) -> list[str]:
        steps = [f"Read the goal: {request.goal[:60]}"]
        if request.constraints:
            steps.append(f"Honour constraints: {', '.join(request.constraints)}")
        if request.facts:
            steps.append(f"Ground in {len(request.facts)} session fact(s)")
        steps.append(f"Call {settings.llm_model} with {len(request.evidence)} grounded passage(s)")
        steps.append("Stream the answer, staying interruptible")
        return steps

    def _messages(self, request: GenerationRequest) -> list[dict[str, str]]:
        evidence = "\n\n".join(
            f"[{item['doc_id']}] {item['title']}: {item['snippet']}" for item in request.evidence
        ) or "(no matching passages)"
        messages = [{"role": "system", "content": SYSTEM}]
        messages.extend(request.context)
        user = (
            f"Active goal: {request.goal}\n"
            f"Constraints: {', '.join(request.constraints) or 'none'}\n"
        )
        if request.facts:
            facts_lines = "\n".join(f"  {k}: {v}" for k, v in request.facts.items())
            user += f"Session facts (canonical):\n{facts_lines}\n"
        user += (
            f"Evidence (Interject Travel demo knowledge base - fictional sample data):\n{evidence}\n\n"
            f"User just said: {request.utterance}"
        )
        if request.notice:
            user += (
                "\n\nThe safety harness refused part of this request. Tell the user plainly, "
                f"in your own words: {request.notice}"
            )
        messages.append({"role": "user", "content": user})
        if request.resume_from:
            # The interrupted words go in as the assistant's own turn, then an
            # explicit ask for the rest. Quoted inside a user message instead,
            # real models (seen with Groq) re-answered from the top.
            messages.append({"role": "assistant", "content": request.resume_from})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You were interrupted mid-answer. Continue your previous message from "
                        "exactly where it was cut off. Output only the remaining text: do not "
                        "repeat anything you already said and do not add a preamble."
                    ),
                }
            )
        return messages

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        payload = {
            "model": settings.llm_model,
            "messages": self._messages(request),
            "stream": True,
            "temperature": 0.3,
        }
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"

        # A rate limit (429) or a transient server error (5xx) is retried, but
        # only while the provider has sent nothing yet: an error status always
        # arrives before the first streamed chunk, so already-delivered text is
        # never re-sent. The server's Retry-After is honoured; a wait longer than
        # _MAX_RETRY_WAIT_S (e.g. a spent daily quota) fails at once instead.
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code < 400:
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                return
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(event, dict) and event.get("error"):
                                raise RuntimeError(f"provider stream error: {event['error']}")
                            try:
                                delta = event["choices"][0]["delta"].get("content")
                            except (KeyError, IndexError, TypeError, AttributeError):
                                continue
                            if delta:
                                yield delta
                        return

                    detail = (await response.aread()).decode("utf-8", "replace")[:400]
                    wait = _retry_wait(response.headers.get("retry-after"), attempt)
                    retryable = response.status_code in _RETRYABLE_STATUS and wait <= _MAX_RETRY_WAIT_S
                    if not retryable or attempt == _MAX_ATTEMPTS:
                        raise RuntimeError(f"provider returned {response.status_code}: {detail}")
                await asyncio.sleep(wait)


_MAX_ATTEMPTS = 3
_MAX_RETRY_WAIT_S = 10.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _retry_wait(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait before the next attempt: Retry-After if given, else 1s, 2s."""
    if retry_after:
        try:
            return max(0.0, float(retry_after.strip().rstrip("s")))
        except ValueError:
            pass
    return float(attempt)
