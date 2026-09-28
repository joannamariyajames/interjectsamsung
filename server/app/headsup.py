"""Heads-Up: proactive contradiction detection and early cut-in.

Core domain-generic contradiction engine.
Prioritizes PRECISION OVER RECALL.
Only triggers on explicit affirmative assertions that directly contradict
canonical evidence from the corpus or active session facts.
Never triggers on questions, hedges, uncertainty, or ambiguous statements.

Domain-specific rules are registered externally via the ContradictionRegistry
plugin mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class HeadsUpEvent:
    claim: str
    contradiction: str
    confidence: float
    source_doc_id: str
    cut_in_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "contradiction": self.contradiction,
            "confidence": self.confidence,
            "source_doc_id": self.source_doc_id,
            "cut_in_text": self.cut_in_text,
        }


# Words and patterns that indicate non-affirmative intent (questions, hedges, uncertainty)
_QUESTION_STARTS = (
    "is ", "are ", "can ", "could ", "will ", "would ", "what ", "which ",
    "how ", "when ", "where ", "why ", "do ", "does ", "did ", "should ",
)

_QUESTION_PHRASES = (
    "tell me if", "check if", "wondering if", "any idea if", "do you know if",
    "can you check", "does it have", "does it include", "is it true that",
)

_HEDGE_WORDS = (
    "maybe", "might", "perhaps", "i think", "i believe", "i guess",
    "not sure", "not certain", "possibly", "probably", "assuming", "hope",
    "wonder if", "suppose",
)


def _is_non_affirmative(text: str) -> bool:
    """Return True if text is a question, hedged statement, or expresses uncertainty."""
    cleaned = text.strip().lower()
    if cleaned.endswith("?"):
        return True
    if any(cleaned.startswith(q) for q in _QUESTION_STARTS):
        return True
    if any(phrase in cleaned for phrase in _QUESTION_PHRASES):
        return True
    if any(hedge in cleaned for hedge in _HEDGE_WORDS):
        return True
    return False


def _get_doc_content(doc: dict[str, Any]) -> str:
    """Generic helper to extract document content from an evidence dict."""
    content = doc.get("snippet") or doc.get("text") or doc.get("body") or ""
    return str(content)


class ContradictionRule(Protocol):
    """Protocol for modular, domain-agnostic contradiction detection rules."""

    name: str

    def check(
        self,
        utterance: str,
        evidence: Any,
        facts: dict[str, Any],
    ) -> HeadsUpEvent | None:
        ...


class ContradictionRegistry:
    """Registry managing an ordered sequence of pluggable contradiction detection rules."""

    def __init__(self) -> None:
        self._rules: list[ContradictionRule] = []

    def register(self, rule: ContradictionRule) -> None:
        """Register a contradiction rule to be evaluated in registration order."""
        self._rules.append(rule)

    def unregister(self, rule_name: str) -> None:
        """Remove rules with matching name."""
        self._rules = [r for r in self._rules if getattr(r, "name", None) != rule_name]

    def clear(self) -> None:
        """Clear all registered rules."""
        self._rules.clear()

    @property
    def rules(self) -> list[ContradictionRule]:
        return list(self._rules)

    def check(
        self,
        utterance: str,
        evidence: Any,
        facts: dict[str, Any],
        threshold: float = 0.90,
    ) -> HeadsUpEvent | None:
        """Iterate over registered rules and return the first valid contradiction meeting threshold."""
        for rule in self._rules:
            event = rule.check(utterance, evidence, facts)
            if event is not None and event.confidence >= threshold:
                return event
        return None


# Global default registry for plugin registration
default_contradiction_registry = ContradictionRegistry()


def detect_contradiction(
    utterance: str,
    evidence: list[dict[str, str]],
    facts: dict[str, Any],
    threshold: float = 0.90,
    registry: ContradictionRegistry | None = None,
) -> HeadsUpEvent | None:
    """Detect high-confidence contradictions between user assertion and evidence.

    Prioritizes PRECISION OVER RECALL.
    Returns None if:
    - Utterance is a question or hedged/uncertain.
    - Contradiction confidence is below threshold.
    - No direct factual contradiction exists in registered rules.
    """
    if _is_non_affirmative(utterance):
        return None

    active_registry = registry if registry is not None else default_contradiction_registry
    return active_registry.check(utterance, evidence, facts, threshold=threshold)
