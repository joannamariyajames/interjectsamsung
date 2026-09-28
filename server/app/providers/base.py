from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

# System prompt shared by the real-model providers (Gemini, OpenAI-compatible).
# The agent is a general assistant: retrieval always returns the closest
# knowledge-base passages, even for questions the knowledge base does not
# cover, so the model must use them only when they actually answer the
# question - never substitute a nearby route or product for the one asked about.
SYSTEM = (
    "You are Interject, a helpful general-purpose assistant in a live, full-duplex "
    "conversation: the user can interrupt you at any moment. Answer whatever the user "
    "asks, the way a knowledgeable assistant would, using your general knowledge. "
    "Lead with the direct answer, then give brief supporting detail; keep it concise "
    "and conversational. Write plain text, as you would speak it: no markdown - no "
    "**bold**, headings or tables. For a list, put each item on its own line starting "
    "with \"- \".\n\n"
    "Each message includes evidence passages retrieved from a travel knowledge base, "
    "with IDs like [flights#3]. Use a passage only when it actually answers the "
    "question, and cite it inline as [doc_id] when you do. Ignore passages that are "
    "not relevant and do not cite them. Never swap in a different route, place, "
    "product or policy just because a passage mentions it.\n\n"
    "You cannot browse the web or see live data. For things that change - fares, "
    "prices, schedules, availability - give typical ranges or general guidance and "
    "say they are estimates to confirm with the airline, hotel or provider.\n\n"
    "If you are resuming after an interruption, continue from where you stopped; do "
    "not restart or repeat what you already said. When session facts are provided, "
    "treat them as the canonical user context - what the user has told you so far - "
    "and stay consistent with them. If the safety harness refused part of a request, "
    "say so plainly."
)


@dataclass
class GenerationRequest:
    goal: str
    utterance: str
    constraints: list[str] = field(default_factory=list)
    context: list[dict[str, str]] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)
    resume_from: str | None = None
    modality: str = "text"
    # Set when the harness refused something the user asked for. The answer has
    # to say so rather than quietly pretending the request never happened.
    notice: str | None = None
    # Phase 2D: canonical session facts from BackspaceCore, keyed by fact key.
    # Providers use these to ground answers in what the user has actually said
    # across the conversation, not just the current utterance.
    facts: dict[str, object] = field(default_factory=dict)


class Provider(Protocol):
    name: str

    def plan(self, request: GenerationRequest) -> list[str]:
        """Short, human-readable plan steps shown in the UI while thinking."""

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Yield response chunks. Must be cancellable between chunks."""
