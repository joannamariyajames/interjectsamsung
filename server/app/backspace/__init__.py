"""BACKSPACE Core - data model layer.

Phase 2 scope only: the dataclasses and enums the approved architecture is
built from. Nothing here has behaviour yet - no notebook lookups, no graph
traversal, no invalidation, no recomputation, and nothing that reads a raw
utterance. Those land in later phases, composed on top of these models
through a ``BackspaceCore`` facade.

Deliberately independent of the rest of the server: no module in this
package imports ``app.goals``, ``app.retrieval``'s tokenizer, ``asyncio``, or
anything FastAPI/WebSocket-shaped. That independence is load-bearing, not
incidental - it is what keeps this package synchronous, deterministic and
testable without spinning up the agent runtime.
"""

from __future__ import annotations

from .changes import ChangeKind, ChangeSet
from .claims import (
    Claim,
    ClaimNotFoundError,
    ClaimNotInvalidatedError,
    ClaimNotSpokenError,
    ClaimStatus,
    Evidence,
)
from .claims import get_claim_history
from .claims import invalidate_claim
from .claims import mark_claim_spoken
from .claims import require_claim_retractable
from .claims import supersede_claim
from .core import BackspaceCore, BackspaceEvent, BackspaceEventType, FactUpdate, Retraction
from .facts import Fact, FactNotebook, FactNotFoundError, FactStatus
from .graph import (
    Dependency,
    DependencyCycleError,
    DependencyGraph,
    DependencyGraphError,
    DependencyKind,
    NodeKind,
    NodeKindMismatchError,
    NodeNotRegisteredError,
    UnsupportedDependencyKindError,
    WorkItem,
    WorkStatus,
)
from .invalidation import Invalidation, invalidate, invalidate_many
from .planner import PlanStatus, RecomputationPlan, plan_recompute

__all__ = [
    "BackspaceCore",
    "BackspaceEvent",
    "BackspaceEventType",
    "ChangeKind",
    "ChangeSet",
    "Claim",
    "ClaimNotFoundError",
    "ClaimNotInvalidatedError",
    "ClaimNotSpokenError",
    "ClaimStatus",
    "Dependency",
    "DependencyCycleError",
    "DependencyGraph",
    "DependencyGraphError",
    "DependencyKind",
    "Evidence",
    "Fact",
    "FactNotFoundError",
    "FactNotebook",
    "FactStatus",
    "FactUpdate",
    "get_claim_history",
    "Invalidation",
    "invalidate",
    "invalidate_claim",
    "invalidate_many",
    "mark_claim_spoken",
    "NodeKind",
    "NodeKindMismatchError",
    "NodeNotRegisteredError",
    "PlanStatus",
    "RecomputationPlan",
    "plan_recompute",
    "require_claim_retractable",
    "Retraction",
    "supersede_claim",
    "UnsupportedDependencyKindError",
    "WorkItem",
    "WorkStatus",
]
