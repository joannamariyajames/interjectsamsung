"""Optional real-model provider, OpenAI-compatible.

Enabled by setting ``LLM_API_KEY``. Works against OpenAI, Together, Groq,
vLLM, Ollama's compat endpoint, or anything else speaking ``/chat/completions``.
Streaming is line-delimited SSE, and each yielded chunk is a cancellation point,
so interruption behaves identically to the local engine.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from ..config import settings
from .base import GenerationRequest

SYSTEM = (
    "You are a real-time assistant that can be interrupted at any moment. "
    "Answer briefly and concretely, grounded in the evidence provided. "
    "Cite evidence inline as [doc_id]. If you are resuming after an interruption, "
    "continue from where you stopped instead of restarting. Never invent facts "
    "that are not in the evidence. When session facts are provided, treat them "
    "as the canonical user context — they reflect what the user has told you "
    "across the conversation so far."
)


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
            f"Evidence:\n{evidence}\n\n"
            f"User just said: {request.utterance}"
        )
        if request.notice:
            user += (
                "\n\nThe safety harness refused part of this request. Tell the user plainly, "
                f"in your own words: {request.notice}"
            )
        if request.resume_from:
            user += (
                "\n\nYou were interrupted mid-answer. Here is what you had already said; "
                f"continue it, do not repeat it:\n\"\"\"{request.resume_from}\"\"\""
            )
        messages.append({"role": "user", "content": user})
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

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:200]
                    raise RuntimeError(f"provider returned {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
