from __future__ import annotations

import pytest

from app.backspace import ChangeKind, ChangeSet, Fact, Invalidation, PlanStatus, RecomputationPlan


def _invalidation() -> Invalidation:
    new_fact = Fact(key="party_size", value=5, fact_id="f2")
    changeset = ChangeSet(key="party_size", kind=ChangeKind.CHANGED, new_fact=new_fact)
    return Invalidation(changeset=changeset, invalidated_work_ids=["w1"], kept_work_ids=["w2"])


def test_recomputation_plan_defaults():
    plan = RecomputationPlan(invalidation=_invalidation())
    assert plan.status is PlanStatus.PENDING
    assert plan.work_items_to_recompute == []
    assert plan.preserved_work_ids == []
    assert plan.missing_dependencies == []
    assert plan.plan_id


def test_recomputation_plan_can_represent_a_full_plan():
    invalidation = _invalidation()
    plan = RecomputationPlan(
        invalidation=invalidation,
        work_items_to_recompute=["w1"],
        preserved_work_ids=["w2"],
        missing_dependencies=["f3"],
        rationale="only w1 depended on party_size",
        status=PlanStatus.READY,
    )
    assert plan.work_items_to_recompute == ["w1"]
    assert plan.preserved_work_ids == ["w2"]
    assert plan.missing_dependencies == ["f3"]
    assert plan.status is PlanStatus.READY


def test_plan_status_enum_values():
    assert {s.value for s in PlanStatus} == {"pending", "ready", "empty", "executed"}


def test_plan_status_rejects_unknown_value():
    with pytest.raises(ValueError):
        PlanStatus("bogus")


def test_recomputation_plan_serialization_nests_invalidation():
    plan = RecomputationPlan(
        invalidation=_invalidation(),
        work_items_to_recompute=["w1"],
        rationale="r",
        created_at=7.0,
    )
    data = plan.to_dict()
    assert data["work_items_to_recompute"] == ["w1"]
    assert data["invalidation"]["invalidated_work_ids"] == ["w1"]
    assert data["created_at"] == 7.0
