"""Claim and Evidence data models.

A ``Claim`` is one unit of the answer the agent actually said (or is about to
say) - the structured counterpart of a bulleted, cited line like the ones
``providers/mock.py`` already produces (``"- sentence  [doc_id]"``). Giving
it its own lifecycle is what lets a later phase retract *part* of an answer
instead of the whole thing.

``Evidence`` mirrors the plain dict shape already produced by
``runtime.py``'s ``_hits_to_evidence`` (``doc_id`` / ``title`` / ``snippet`` /
``source``) so BACKSPACE can wrap that output without retrieval.py or
runtime.py changing at all.

Phase 2 scope only: shapes, not behaviour. No lifecycle transitions are
implemented here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClaimStatus(str, Enum):
    """The lifecycle a claim can occupy.

    Transitions between these (e.g. SPOKEN -> INVALIDATED) are claim-lifecycle
    behaviour and land in a later phase; this enum only has to be able to
    represent every state, not move between them.
    """

    GENERATED = "generated"
    SPOKEN = "spoken"
    HEARD = "heard"
    VALID = "valid"
    INVALIDATED = "invalidated"
    RETRACTED = "retracted"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class Evidence:
    """Compatible with the evidence dict already threaded through the runtime.

    Deliberately the same four fields, in the same names, as
    ``{"doc_id": ..., "title": ..., "snippet": ..., "source": ...}`` - so
    existing evidence dicts convert both ways without loss.
    """

    doc_id: str
    title: str = ""
    snippet: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls(
            doc_id=data["doc_id"],
            title=data.get("title", ""),
            snippet=data.get("snippet", ""),
            source=data.get("source", ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
        }


@dataclass
class Claim:
    """One claim the answer makes, with what it depends on to remain true."""

    text: str
    claim_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    work_id: str | None = None
    turn_id: str = ""
    goal_id: str | None = None
    evidence: Evidence | None = None
    status: ClaimStatus = ClaimStatus.GENERATED
    depends_on_facts: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    spoken_at: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "work_id": self.work_id,
            "turn_id": self.turn_id,
            "goal_id": self.goal_id,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "status": self.status.value,
            "depends_on_facts": list(self.depends_on_facts),
            "created_at": self.created_at,
            "spoken_at": self.spoken_at,
            "provenance": dict(self.provenance),
        }
