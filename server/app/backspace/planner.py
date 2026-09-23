"""RecomputationPlan data model.

What deciding "which stale work actually needs rerunning this turn" will
eventually produce, given an ``Invalidation``. Ordering the work, deciding
what is safe to preserve, and noticing a dependency that went missing are all
planning behaviour and belong to a later phase; this file only defines the
shape a plan takes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from .invalidation import Invalidation


class PlanStatus(str, Enum):
    """Where a recomputation plan stands."""

    PENDING = "pending"
    READY = "ready"
    EMPTY = "empty"
    EXECUTED = "executed"


@dataclass
class RecomputationPlan:
    """An ordered set of work items to redo, and what was spared."""

    invalidation: Invalidation
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    work_items_to_recompute: list[str] = field(default_factory=list)
    preserved_work_ids: list[str] = field(default_factory=list)
    missing_dependencies: list[str] = field(default_factory=list)
    rationale: str = ""
    status: PlanStatus = PlanStatus.PENDING
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "invalidation": self.invalidation.to_dict(),
            "work_items_to_recompute": list(self.work_items_to_recompute),
            "preserved_work_ids": list(self.preserved_work_ids),
            "missing_dependencies": list(self.missing_dependencies),
            "rationale": self.rationale,
            "status": self.status.value,
            "created_at": self.created_at,
        }
