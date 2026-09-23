"""Composite result/domain-event models, and the BackspaceCore facade.

``FactUpdate`` and ``Retraction`` are composite result objects; ``BackspaceEvent``
is an internal domain concept only - not a ``ServerFrame`` (see
``app/schemas.py``), does not know how to serialise onto a WebSocket, and
nothing here imports ``asyncio`` or anything FastAPI-shaped. Translating a
``BackspaceEvent`` into a wire frame is the runtime layer's job, not this
package's.

``BackspaceCore`` (from Phase 3 onward) is the facade: it owns a session-local
``FactNotebook`` and ``DependencyGraph`` and exposes their operations as one
surface. As of Phase 5 it also exposes ``invalidate``/``invalidate_many``, but
deliberately does *not* call them from ``assert_fact`` - that wiring, and
recomputation planning, are later phases' work. Every facade method here is a
thin delegation; this file decides nothing on its own that ``facts.py``,
``graph.py`` or ``invalidation.py`` doesn't already decide.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from .changes import ChangeKind, ChangeSet
from .claims import Claim
from .facts import Fact, FactNotebook
from .graph import Dependency, DependencyGraph, DependencyKind, WorkItem
from .invalidation import Invalidation
from .invalidation import invalidate as _invalidate
from .invalidation import invalidate_many as _invalidate_many
from .planner import RecomputationPlan


@dataclass
class FactUpdate:
    """The result ``assert_fact``/``update_fact`` will eventually return.

    ``status`` reuses ``ChangeKind`` rather than a second new/unchanged/
    changed enum - the two questions ("what changed?" and "did this fact
    observation change anything?") are the same question asked from two
    sides, and giving them separate enums would just invite the two to drift.
    """

    fact: Fact
    status: ChangeKind
    previous: Fact | None = None
    changeset: ChangeSet | None = None
    invalidation: Invalidation | None = None
    plan: RecomputationPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "status": self.status.value,
            "previous": self.previous.to_dict() if self.previous is not None else None,
            "changeset": self.changeset.to_dict() if self.changeset is not None else None,
            "invalidation": self.invalidation.to_dict() if self.invalidation is not None else None,
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


@dataclass
class Retraction:
    """The result a future ``retract_claim`` will return.

    Distinct from ``Invalidation``: an invalidation is graph-wide bookkeeping
    triggered by a fact change; a retraction is the user-facing act of taking
    back one already-spoken claim, and carries the reason a person reading
    the transcript would need.
    """

    claim_id: str
    reason: str
    retraction_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    previous_claim: Claim | None = None
    changed_facts: list[Fact] = field(default_factory=list)
    source: str = "unknown"
    created_at: float = field(default_factory=time.time)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retraction_id": self.retraction_id,
            "claim_id": self.claim_id,
            "reason": self.reason,
            "previous_claim": self.previous_claim.to_dict() if self.previous_claim is not None else None,
            "changed_facts": [f.to_dict() for f in self.changed_facts],
            "source": self.source,
            "created_at": self.created_at,
            "provenance": dict(self.provenance),
        }


class BackspaceEventType(str, Enum):
    """Internal domain events a future pipeline will raise.

    Not wire frames. A later phase may translate a subset of these into
    ``ServerFrame`` subclasses in ``app/schemas.py``; that translation does
    not live here.
    """

    FACT_ADDED = "fact_added"
    FACT_CHANGED = "fact_changed"
    WORK_REGISTERED = "work_registered"
    DEPENDENCY_REGISTERED = "dependency_registered"
    CLAIM_REGISTERED = "claim_registered"
    WORK_INVALIDATED = "work_invalidated"
    CLAIM_INVALIDATED = "claim_invalidated"
    CLAIM_RETRACTED = "claim_retracted"
    RECOMPUTATION_PLANNED = "recomputation_planned"
    BACKSPACE_COMPLETED = "backspace_completed"


@dataclass
class BackspaceEvent:
    """One internal domain event, carrying whatever payload it describes."""

    event_type: BackspaceEventType
    turn_id: str = ""
    goal_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "turn_id": self.turn_id,
            "goal_id": self.goal_id,
            "payload": dict(self.payload),
            "created_at": self.created_at,
        }


class BackspaceCore:
    """The facade: owns a session-local ``FactNotebook`` and exposes the
    public fact API.

    This phase implements fact operations only. It composes ``FactNotebook``
    rather than reimplementing anything from it - every method below is a
    thin, direct delegation, so there is exactly one place (``facts.py``)
    that actually decides NEW/UNCHANGED/CHANGED. Dependency registration,
    invalidation, recomputation planning and claim lifecycle transitions are
    not implemented here yet; they compose on top of this facade in a later
    phase.

    Phase 4 adds the dependency-graph surface: registering work/claims and
    edges, and querying dependencies/dependents. It is a thin delegation to
    ``DependencyGraph``, the same way the fact methods delegate to
    ``FactNotebook`` - this facade still decides nothing itself. Nothing
    here connects the two: ``assert_fact`` does not touch the graph, and no
    graph query triggers invalidation. That wiring is later phases' job.

    Each instance owns its own ``FactNotebook`` and ``DependencyGraph`` - two
    ``BackspaceCore()``s never share state, matching the session-scoped-
    memory constraint the rest of the server already follows.
    """

    def __init__(self) -> None:
        self.facts = FactNotebook()
        self.graph = DependencyGraph()

    # -- fact operations ---------------------------------------------------
    def assert_fact(
        self,
        key: str,
        value: Any,
        *,
        source: str,
        turn_id: str,
        goal_id: str | None = None,
        confidence: float = 1.0,
    ) -> FactUpdate:
        """Record a structured fact observation. See ``FactNotebook.observe``
        for the NEW/UNCHANGED/CHANGED semantics this delegates to."""
        return self.facts.observe(
            key, value, source=source, turn_id=turn_id, goal_id=goal_id, confidence=confidence
        )

    def update_fact(
        self,
        fact_id: str,
        value: Any,
        *,
        source: str,
        turn_id: str,
        goal_id: str | None = None,
        confidence: float = 1.0,
    ) -> FactUpdate:
        """Correct the fact identified by ``fact_id``. Raises
        ``FactNotFoundError`` (from ``facts.py``) if it does not exist."""
        return self.facts.update_fact(
            fact_id, value, source=source, turn_id=turn_id, goal_id=goal_id, confidence=confidence
        )

    def get_fact(self, key: str) -> Fact | None:
        return self.facts.get_fact(key)

    def get_fact_history(self, key: str) -> list[Fact]:
        return self.facts.get_fact_history(key)

    def snapshot(self) -> dict[str, Any]:
        """A read-only, serialisation-friendly dump of this session's facts."""
        return {"facts": self.facts.snapshot()}

    def reset(self) -> None:
        """Clear this instance's notebook and graph. Other ``BackspaceCore``
        instances are untouched."""
        self.facts.reset()
        self.graph.reset()

    # -- dependency-graph operations -----------------------------------------
    def register_work(self, work: WorkItem) -> WorkItem:
        return self.graph.register_work(work)

    def register_claim(self, claim: Claim) -> Claim:
        return self.graph.register_claim(claim)

    def register_dependency(self, kind: DependencyKind, from_id: str, to_id: str) -> Dependency:
        return self.graph.register_dependency(kind, from_id, to_id)

    def get_dependencies(self, node_id: str) -> list[str]:
        return self.graph.get_dependencies(node_id)

    def get_direct_dependents(self, node_id: str) -> list[str]:
        return self.graph.get_direct_dependents(node_id)

    def get_transitive_dependents(self, node_id: str) -> list[str]:
        return self.graph.get_transitive_dependents(node_id)

    # -- invalidation ---------------------------------------------------------
    def invalidate(self, changeset: ChangeSet) -> Invalidation:
        """Compute what becomes stale downstream of one ChangeSet, against
        this instance's own graph. Does not touch FactNotebook state and is
        not called automatically from ``assert_fact`` - that wiring is a
        later phase's job."""
        return _invalidate(self.graph, changeset)

    def invalidate_many(self, changesets: Sequence[ChangeSet]) -> Invalidation:
        """Same as ``invalidate``, merged across a batch of ChangeSets - see
        ``invalidation.invalidate_many`` for the exact semantics."""
        return _invalidate_many(self.graph, changesets)
