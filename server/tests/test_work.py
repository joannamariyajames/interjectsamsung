from __future__ import annotations

from app.goals import Goal, GoalTracker
from app.work import (
    ConstraintPredicate,
    ReconciliationResult,
    WorkItem,
    WorkStatus,
    extract_work_items_from_evidence,
    format_reconciliation_summary,
    reconcile_work_items,
    validate_work_item,
)


def test_work_item_defaults_to_valid():
    item = WorkItem(
        item_id="item-1",
        goal_id="goal-1",
        kind="flight",
        title="Flight to Goa",
        data={"price": 6000},
        depends_on=["budget", "destination"],
    )
    assert item.status == WorkStatus.VALID
    assert item.status == "valid"
    assert item.stale_reason is None
    assert item.source_doc_id is None


def test_budget_change_makes_expensive_item_stale():
    item = WorkItem(
        item_id="hotel-1",
        goal_id="goal-1",
        kind="hotel",
        title="Luxury Resort",
        data={"price": 25000},
        depends_on=["budget"],
    )
    status, reason = validate_work_item(item, {"budget": 20000})
    assert status == WorkStatus.STALE
    assert item.status == WorkStatus.STALE
    assert reason is not None
    assert "25000" in reason
    assert "20000" in reason
    assert item.stale_reason == reason


def test_unrelated_item_remains_valid():
    # Item depends on destination, not budget
    item = WorkItem(
        item_id="sightseeing-1",
        goal_id="goal-1",
        kind="activity",
        title="Scuba Diving in Goa",
        data={"destination": "Goa", "price": 4000},
        depends_on=["destination"],
    )
    # Budget changes to 2000 (lower than price 4000), but item does not depend on budget
    status, reason = validate_work_item(item, {"budget": 2000, "destination": "Goa"})
    assert status == WorkStatus.VALID
    assert item.status == WorkStatus.VALID
    assert reason is None
    assert item.stale_reason is None


def test_people_change_makes_dependent_item_stale():
    item = WorkItem(
        item_id="cab-1",
        goal_id="goal-1",
        kind="transport",
        title="Sedan Taxi for 2",
        data={"people": 2, "price": 1500},
        depends_on=["people"],
    )
    status, reason = validate_work_item(item, {"people": 5})
    assert status == WorkStatus.STALE
    assert item.status == WorkStatus.STALE
    assert reason is not None
    assert "2" in reason
    assert "5" in reason
    assert item.stale_reason == reason


def test_destination_change_makes_dependent_item_stale():
    item = WorkItem(
        item_id="hotel-2",
        goal_id="goal-1",
        kind="hotel",
        title="Beach Villa Goa",
        data={"destination": "Goa", "price": 8000},
        depends_on=["destination"],
    )
    status, reason = validate_work_item(item, {"destination": "Bangalore"})
    assert status == WorkStatus.STALE
    assert item.status == WorkStatus.STALE
    assert reason is not None
    assert "Goa" in reason
    assert "Bangalore" in reason
    assert item.stale_reason == reason


def test_matching_facts_keep_item_valid():
    item = WorkItem(
        item_id="pkg-1",
        goal_id="goal-1",
        kind="package",
        title="Goa Holiday Package",
        data={"destination": "Goa", "people": 2, "price": 18000},
        depends_on=["destination", "people", "budget"],
    )
    status, reason = validate_work_item(
        item,
        {"destination": "Goa", "people": 2, "budget": 20000},
    )
    assert status == WorkStatus.VALID
    assert item.status == WorkStatus.VALID
    assert reason is None
    assert item.stale_reason is None


def test_stale_reason_is_populated_correctly():
    item = WorkItem(
        item_id="item-test",
        goal_id="goal-1",
        kind="flight",
        title="Flight Indigo",
        data={"price": 12000},
        depends_on=["budget"],
    )
    status, reason = validate_work_item(item, {"budget": 10000})
    assert status == WorkStatus.STALE
    assert item.stale_reason == reason
    assert "12000" in item.stale_reason and "10000" in item.stale_reason

    # If facts now increase budget to 15000, revalidating clears stale status and reason
    status_reval, reason_reval = validate_work_item(item, {"budget": 15000})
    assert status_reval == WorkStatus.VALID
    assert item.status == WorkStatus.VALID
    assert item.stale_reason is None


def test_goal_with_work_items_compatibility():
    # Existing construction without work_items argument
    goal = Goal(text="Find flights to Mumbai")
    assert goal.work_items == []

    # Construction with work_items
    item = WorkItem(
        item_id="w-1",
        goal_id=goal.goal_id,
        kind="flight",
        title="Flight 6E-201",
        data={"price": 5000},
        depends_on=["budget"],
    )
    goal.work_items.append(item)
    assert len(goal.work_items) == 1
    d = goal.to_dict()
    assert "work_items" in d
    assert len(d["work_items"]) == 1
    assert d["work_items"][0]["item_id"] == "w-1"


# ----------------------------------------------------------------------
# Phase 4B tests: Extraction from evidence and runtime integration
# ----------------------------------------------------------------------


def test_hotel_evidence_creates_work_item():
    evidence = [
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": (
                "The Harbour House in Mumbai sits 20 minutes from the domestic "
                "terminal, rates from 8,900 INR, rooftop pool, 24-hour gym, free "
                "airport shuttle at :15 past the hour."
            ),
            "source": "hotels",
        }
    ]
    items = extract_work_items_from_evidence(
        evidence, "goal-1", {"destination": "Mumbai", "budget": 10000}
    )
    assert len(items) == 1
    item = items[0]
    assert item.kind == "hotel"
    assert item.status == WorkStatus.VALID
    assert item.goal_id == "goal-1"


def test_correct_name_city_price_doc_id():
    evidence = [
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": (
                "The Harbour House in Mumbai sits 20 minutes from the domestic "
                "terminal, rates from 8,900 INR, rooftop pool, 24-hour gym, free "
                "airport shuttle at :15 past the hour."
            ),
            "source": "hotels",
        }
    ]
    items = extract_work_items_from_evidence(
        evidence, "goal-1", {"destination": "Mumbai", "budget": 10000}
    )
    assert len(items) == 1
    item = items[0]
    assert item.data["name"] == "The Harbour House"
    assert item.data["city"] == "Mumbai"
    assert item.data["destination"] == "Mumbai"
    assert item.data["price"] == 8900
    assert item.source_doc_id == "hotels#1"
    assert item.title == "The Harbour House, Mumbai"


def test_correct_dependencies():
    evidence = [
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": (
                "The Harbour House in Mumbai sits 20 minutes from the domestic "
                "terminal, rates from 8,900 INR, rooftop pool."
            ),
            "source": "hotels",
        }
    ]
    items = extract_work_items_from_evidence(evidence, "goal-1")
    assert len(items) == 1
    assert items[0].depends_on == ["destination", "budget"]


def test_duplicate_evidence_does_not_create_duplicates():
    evidence = [
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": (
                "The Harbour House in Mumbai sits 20 minutes from the domestic "
                "terminal, rates from 8,900 INR."
            ),
            "source": "hotels",
        },
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": (
                "The Harbour House in Mumbai sits 20 minutes from the domestic "
                "terminal, rates from 8,900 INR."
            ),
            "source": "hotels",
        },
    ]
    items = extract_work_items_from_evidence(evidence, "goal-1")
    assert len(items) == 1

    # Also test Goal work_items deduplication logic
    goal = Goal(text="Book Mumbai hotel")
    existing_ids = {i.item_id for i in goal.work_items}
    for item in items:
        if item.item_id not in existing_ids:
            goal.work_items.append(item)
            existing_ids.add(item.item_id)

    # Re-extract and re-add
    more_items = extract_work_items_from_evidence(evidence, goal.goal_id)
    for item in more_items:
        if item.item_id not in existing_ids:
            goal.work_items.append(item)
            existing_ids.add(item.item_id)

    assert len(goal.work_items) == 1


def test_budget_change_marks_existing_item_stale():
    evidence = [
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": "Harbour House, Mumbai, 8900 INR",
            "source": "hotels",
        }
    ]
    # Turn 1: budget 10000 -> item is VALID
    facts_turn1 = {"destination": "Mumbai", "budget": 10000}
    items = extract_work_items_from_evidence(evidence, "goal-1", facts_turn1)
    assert len(items) == 1
    item = items[0]
    assert item.status == WorkStatus.VALID

    # Turn 2: budget drops to 8000 -> item becomes STALE
    facts_turn2 = {"destination": "Mumbai", "budget": 8000}
    validate_work_item(item, facts_turn2)
    assert item.status == WorkStatus.STALE
    assert item.stale_reason is not None
    assert "8900" in item.stale_reason
    assert "8000" in item.stale_reason


def test_matching_budget_keeps_item_valid():
    evidence = [
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": (
                "Lantern Court in Bengaluru is 35 minutes from the airport, "
                "rates from 6,200 INR."
            ),
            "source": "hotels",
        }
    ]
    facts = {"destination": "Bengaluru", "budget": 7000}
    items = extract_work_items_from_evidence(evidence, "goal-1", facts)
    assert len(items) == 1
    assert items[0].status == WorkStatus.VALID
    assert items[0].stale_reason is None


def test_work_item_remains_attached_to_goal():
    goal = Goal(text="Book Mumbai under 10k")
    evidence = [
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": (
                "The Harbour House in Mumbai sits 20 minutes from the domestic "
                "terminal, rates from 8,900 INR."
            ),
            "source": "hotels",
        }
    ]
    items = extract_work_items_from_evidence(
        evidence, goal.goal_id, {"destination": "Mumbai", "budget": 10000}
    )
    goal.work_items.extend(items)

    assert len(goal.work_items) == 1
    assert goal.work_items[0].title == "The Harbour House, Mumbai"
    assert goal.work_items[0].status == WorkStatus.VALID

    # Revalidation when fact changes
    validate_work_item(goal.work_items[0], {"destination": "Mumbai", "budget": 8000})
    assert goal.work_items[0].status == WorkStatus.STALE
    assert len(goal.work_items) == 1

    # Verify serialization in to_dict
    d = goal.to_dict()
    assert len(d["work_items"]) == 1
    assert d["work_items"][0]["status"] == "stale"
    assert d["work_items"][0]["item_id"] == goal.work_items[0].item_id


async def test_runtime_turn_end_to_end_creates_and_updates_work_items():
    from app.runtime import AgentRuntime
    from app.session import Session
    from tests.collector import Collector

    session = Session("test-session")
    collector = Collector()
    runtime = AgentRuntime(session, collector)

    # Turn 1: "Book hotel in Mumbai under 10000."
    await runtime.on_final("Book hotel in Mumbai under 10000.")
    await runtime._task

    active_goal = session.goals.active
    assert active_goal is not None
    assert len(active_goal.work_items) >= 1
    mumbai_item = next(
        (w for w in active_goal.work_items if w.data.get("city") == "Mumbai"),
        None,
    )
    assert mumbai_item is not None
    assert mumbai_item.status == WorkStatus.VALID
    initial_count = len(active_goal.work_items)

    # Turn 2: "Budget 8000."
    await runtime.on_final("Budget 8000.")
    await runtime._task

    # No duplicate items added for Mumbai hotel
    assert len(active_goal.work_items) == initial_count
    assert mumbai_item.status == WorkStatus.STALE
    assert mumbai_item.stale_reason is not None
    assert "8900" in mumbai_item.stale_reason and "8000" in mumbai_item.stale_reason


# ----------------------------------------------------------------------
# Phase 5 tests: Bookmark Reconciliation
# ----------------------------------------------------------------------


def test_valid_work_items_remain_valid_after_reconciliation():
    item = WorkItem(
        item_id="hotel-harbour-house-mumbai",
        goal_id="g1",
        kind="hotel",
        title="The Harbour House, Mumbai",
        data={"name": "The Harbour House", "destination": "Mumbai", "price": 8900},
        depends_on=["destination", "budget"],
        status=WorkStatus.VALID,
    )
    facts = {"destination": "Mumbai", "budget": 10000}
    result = reconcile_work_items([item], facts)

    assert len(result.valid) == 1
    assert len(result.stale) == 0
    assert result.valid[0] is item
    assert item.status == WorkStatus.VALID
    assert item.stale_reason is None


def test_budget_change_makes_expensive_item_stale_in_reconciliation():
    item = WorkItem(
        item_id="hotel-harbour-house-mumbai",
        goal_id="g1",
        kind="hotel",
        title="The Harbour House, Mumbai",
        data={"name": "The Harbour House", "destination": "Mumbai", "price": 8900},
        depends_on=["destination", "budget"],
        status=WorkStatus.VALID,
    )
    # Budget lowered to 8000 (below price 8900)
    facts = {"destination": "Mumbai", "budget": 8000}
    result = reconcile_work_items([item], facts)

    assert len(result.stale) == 1
    assert len(result.valid) == 0
    assert result.stale[0] is item
    assert item.status == WorkStatus.STALE
    assert item in result.became_stale
    assert "8900" in item.stale_reason and "8000" in item.stale_reason


def test_people_change_makes_incompatible_item_stale_in_reconciliation():
    item = WorkItem(
        item_id="car-sedan-for-2",
        goal_id="g1",
        kind="transport",
        title="Sedan Transfer",
        data={"people": 2, "price": 2000},
        depends_on=["people", "budget"],
        status=WorkStatus.VALID,
    )
    facts = {"people": 5, "budget": 10000}
    result = reconcile_work_items([item], facts)

    assert len(result.stale) == 1
    assert result.stale[0].status == WorkStatus.STALE
    assert "2" in item.stale_reason and "5" in item.stale_reason
    assert item in result.became_stale


def test_destination_change_makes_incompatible_item_stale_in_reconciliation():
    item = WorkItem(
        item_id="hotel-harbour-house-mumbai",
        goal_id="g1",
        kind="hotel",
        title="The Harbour House, Mumbai",
        data={"destination": "Mumbai", "price": 8900},
        depends_on=["destination", "budget"],
        status=WorkStatus.VALID,
    )
    facts = {"destination": "Goa", "budget": 10000}
    result = reconcile_work_items([item], facts)

    assert len(result.stale) == 1
    assert result.stale[0].status == WorkStatus.STALE
    assert "Mumbai" in item.stale_reason and "Goa" in item.stale_reason
    assert item in result.became_stale


def test_unchanged_facts_preserve_valid_items():
    item = WorkItem(
        item_id="hotel-lantern-court-bengaluru",
        goal_id="g1",
        kind="hotel",
        title="Lantern Court, Bengaluru",
        data={"destination": "Bengaluru", "price": 6200},
        depends_on=["destination", "budget"],
        status=WorkStatus.VALID,
    )
    facts = {"destination": "Bengaluru", "budget": 10000}
    result = reconcile_work_items([item], facts)

    assert len(result.valid) == 1
    assert result.valid[0].status == WorkStatus.VALID
    assert len(result.became_stale) == 0


def test_multiple_work_items_partitioned_correctly():
    expensive = WorkItem(
        item_id="hotel-harbour-house",
        goal_id="g1",
        kind="hotel",
        title="The Harbour House",
        data={"destination": "Mumbai", "price": 8900},
        depends_on=["destination", "budget"],
        status=WorkStatus.VALID,
    )
    affordable = WorkItem(
        item_id="hotel-palm-resort",
        goal_id="g1",
        kind="hotel",
        title="Hotel Palm",
        data={"destination": "Mumbai", "price": 6500},
        depends_on=["destination", "budget"],
        status=WorkStatus.VALID,
    )
    facts = {"destination": "Mumbai", "budget": 8000}
    result = reconcile_work_items([expensive, affordable], facts)

    assert len(result.valid) == 1
    assert result.valid[0] is affordable
    assert affordable.status == WorkStatus.VALID

    assert len(result.stale) == 1
    assert result.stale[0] is expensive
    assert expensive.status == WorkStatus.STALE

    # Tuple unpacking support
    valid_list, stale_list = result
    assert valid_list == [affordable]
    assert stale_list == [expensive]


def test_stale_items_are_preserved_not_deleted():
    item1 = WorkItem(
        item_id="item-1",
        goal_id="g1",
        kind="hotel",
        title="Hotel 1",
        data={"price": 10000},
        depends_on=["budget"],
    )
    item2 = WorkItem(
        item_id="item-2",
        goal_id="g1",
        kind="hotel",
        title="Hotel 2",
        data={"price": 5000},
        depends_on=["budget"],
    )
    items = [item1, item2]

    # Budget is 6000, item1 becomes stale, item2 remains valid
    result = reconcile_work_items(items, {"budget": 6000})

    # Both items are preserved in items and partitioned
    assert len(items) == 2
    assert len(result.valid) + len(result.stale) == 2
    assert item1.status == WorkStatus.STALE
    assert item2.status == WorkStatus.VALID


def test_stale_reason_cleared_if_item_becomes_valid_again():
    item = WorkItem(
        item_id="hotel-1",
        goal_id="g1",
        kind="hotel",
        title="Resort",
        data={"price": 8900},
        depends_on=["budget"],
        status=WorkStatus.VALID,
    )
    # Becomes stale
    res1 = reconcile_work_items([item], {"budget": 8000})
    assert item.status == WorkStatus.STALE
    assert item.stale_reason is not None
    assert item in res1.became_stale

    # Budget increases back to 10000 -> item becomes valid again
    res2 = reconcile_work_items([item], {"budget": 10000})
    assert item.status == WorkStatus.VALID
    assert item.stale_reason is None
    assert item in res2.became_valid


def test_reconciliation_works_when_parked_goal_is_resumed():
    tracker = GoalTracker()
    # Goal 1: Book Goa for two under 20k
    g1 = tracker.apply("Book Goa for two under 20k", tracker.classify("Book Goa for two under 20k"))
    item1 = WorkItem(
        item_id="hotel-goa-luxury",
        goal_id=g1.goal_id,
        kind="hotel",
        title="Goa Luxury Resort",
        data={"name": "Goa Luxury Resort", "destination": "Goa", "price": 15000, "people": 2},
        depends_on=["destination", "budget", "people"],
        status=WorkStatus.VALID,
    )
    item2 = WorkItem(
        item_id="hotel-goa-inn",
        goal_id=g1.goal_id,
        kind="hotel",
        title="Goa Budget Inn",
        data={"name": "Goa Budget Inn", "destination": "Goa", "price": 6000, "people": 2},
        depends_on=["destination", "budget", "people"],
        status=WorkStatus.VALID,
    )
    g1.work_items.extend([item1, item2])

    # Goal 2: Switch to refund policy -> Goal 1 parked
    tracker.apply(
        "actually, what is the refund policy?",
        tracker.classify("actually, what is the refund policy?"),
    )
    assert g1.status.value == "parked"

    # User changes fact: budget is now 8k
    new_facts = {"destination": "Goa", "people": 2, "budget": 8000}

    # Resume Goal 1
    resumed = tracker.apply("anyway, back to the hotel", tracker.classify("anyway, back to the hotel"))
    assert resumed.goal_id == g1.goal_id
    assert resumed.status.value == "active"

    # Reconcile Goal 1 work items
    result = reconcile_work_items(resumed.work_items, new_facts)
    assert len(result.stale) == 1
    assert result.stale[0].item_id == "hotel-goa-luxury"
    assert result.stale[0].status == WorkStatus.STALE

    assert len(result.valid) == 1
    assert result.valid[0].item_id == "hotel-goa-inn"
    assert result.valid[0].status == WorkStatus.VALID

    # Ensure items remain attached
    assert len(resumed.work_items) == 2


def test_existing_work_items_not_duplicated_during_reconciliation():
    goal = Goal(text="Book Mumbai under 10k")
    item = WorkItem(
        item_id="hotel-harbour-house-mumbai",
        goal_id=goal.goal_id,
        kind="hotel",
        title="The Harbour House, Mumbai",
        data={"name": "The Harbour House", "destination": "Mumbai", "price": 8900},
        depends_on=["destination", "budget"],
        status=WorkStatus.VALID,
    )
    goal.work_items.append(item)

    # Reconcile twice
    facts = {"destination": "Mumbai", "budget": 10000}
    reconcile_work_items(goal.work_items, facts)
    reconcile_work_items(goal.work_items, facts)
    assert len(goal.work_items) == 1

    # Extract items from evidence again and ensure deduplication
    evidence = [
        {
            "doc_id": "hotels#1",
            "title": "Properties",
            "snippet": "Harbour House, Mumbai, 8900 INR",
            "source": "hotels",
        }
    ]
    new_items = extract_work_items_from_evidence(evidence, goal.goal_id, facts)
    existing_ids = {w.item_id for w in goal.work_items}
    for ni in new_items:
        if ni.item_id not in existing_ids:
            goal.work_items.append(ni)
            existing_ids.add(ni.item_id)

    assert len(goal.work_items) == 1


async def test_end_to_end_parked_goal_reconciliation_flow():
    from app.runtime import AgentRuntime
    from app.session import Session
    from tests.collector import Collector

    session = Session("test-recon-session")
    collector = Collector()
    runtime = AgentRuntime(session, collector)

    # Turn 1: "Book hotel in Mumbai for two under 20000."
    await runtime.on_final("Book hotel in Mumbai for two under 20000.")
    await runtime._task

    goal_a = session.goals.active
    assert goal_a is not None
    assert len(goal_a.work_items) >= 1
    mumbai_hotel = next(
        (w for w in goal_a.work_items if w.data.get("city") == "Mumbai"),
        None,
    )
    assert mumbai_hotel is not None
    assert mumbai_hotel.status == WorkStatus.VALID
    original_work_count = len(goal_a.work_items)

    # Turn 2: Switch to another goal
    await runtime.on_final("actually, what is the refund policy if I cancel?")
    await runtime._task

    assert goal_a.status.value == "parked"
    goal_b = session.goals.active
    assert goal_b is not None and goal_b.goal_id != goal_a.goal_id

    # Turn 3: "Budget 8000."
    await runtime.on_final("Budget 8000.")
    await runtime._task

    assert session.backspace.get_fact("budget") is not None
    assert session.backspace.get_fact("budget").value == 8000

    # Turn 4: "anyway, back to the hotel"
    await runtime.on_final("anyway, back to the hotel")
    await runtime._task

    # Goal A must now be resumed
    active_now = session.goals.active
    assert active_now is not None
    assert active_now.goal_id == goal_a.goal_id
    assert active_now.status.value == "active"

    # WorkItems must still exist, not duplicated, and Mumbai hotel must now be STALE
    assert len(active_now.work_items) == original_work_count
    assert mumbai_hotel.status == WorkStatus.STALE
    assert mumbai_hotel.stale_reason is not None
    assert "8900" in mumbai_hotel.stale_reason and "8000" in mumbai_hotel.stale_reason

    # Verify reconciliation frame metadata was emitted
    messages = collector.of("message")
    last_agent_msg = [m for m in messages if m.get("role") == "agent"][-1]
    assert "reconciliation" in last_agent_msg.get("meta", {})
    recon_meta = last_agent_msg["meta"]["reconciliation"]
    assert recon_meta is not None
    stale_ids = [s["item_id"] for s in recon_meta.get("stale", [])]
    assert mumbai_hotel.item_id in stale_ids


# ----------------------------------------------------------------------
# Phase A.2: Generic ConstraintPredicate & Non-Travel Domain Tests
# ----------------------------------------------------------------------


def test_predicate_less_than_or_equal():
    """Verify <= operator marks stale when value exceeds limit."""
    pred = ConstraintPredicate(fact_key="max_cost_hourly", operator="<=", item_field="cost_hourly")
    item = WorkItem(
        item_id="compute-c5-large",
        goal_id="infra-goal-1",
        kind="compute_instance",
        title="AWS c5.large instance",
        data={"cost_hourly": 0.085, "ram_gb": 4},
        predicates=[pred],
    )

    # Within limit: valid
    status, reason = validate_work_item(item, {"max_cost_hourly": 0.10})
    assert status == WorkStatus.VALID
    assert reason is None
    assert item.status == WorkStatus.VALID

    # Exceeds limit: stale
    status, reason = validate_work_item(item, {"max_cost_hourly": 0.05})
    assert status == WorkStatus.STALE
    assert item.status == WorkStatus.STALE
    assert reason is not None
    assert "0.085" in reason and "0.05" in reason


def test_predicate_greater_than_or_equal():
    """Verify >= operator marks stale when value is below requirement."""
    pred = ConstraintPredicate(fact_key="min_ram_gb", operator=">=", item_field="ram_gb")
    item = WorkItem(
        item_id="compute-r5-xlarge",
        goal_id="infra-goal-1",
        kind="compute_instance",
        title="AWS r5.xlarge instance",
        data={"ram_gb": 32, "vcpus": 4},
        predicates=[pred],
    )

    # Meets required RAM: valid
    status, reason = validate_work_item(item, {"min_ram_gb": 16})
    assert status == WorkStatus.VALID
    assert reason is None

    # Below required RAM: stale
    status, reason = validate_work_item(item, {"min_ram_gb": 64})
    assert status == WorkStatus.STALE
    assert reason is not None
    assert "ram_gb 32 is below required 64" in reason


def test_predicate_equality():
    """Verify == operator matches exact strings/values case-insensitively for strings."""
    pred = ConstraintPredicate(fact_key="required_tier", operator="==", item_field="tier")
    item = WorkItem(
        item_id="db-prod-cluster",
        goal_id="db-goal-1",
        kind="database",
        title="Aurora PostgreSQL Cluster",
        data={"tier": "Enterprise", "replicas": 3},
        predicates=[pred],
    )

    # Exact match: valid
    status, reason = validate_work_item(item, {"required_tier": "enterprise"})
    assert status == WorkStatus.VALID
    assert reason is None

    # Mismatch: stale
    status, reason = validate_work_item(item, {"required_tier": "Standard"})
    assert status == WorkStatus.STALE
    assert reason is not None
    assert "tier 'Enterprise' does not match 'Standard'" in reason


def test_predicate_inequality():
    """Verify != operator rejects matching forbidden values."""
    pred = ConstraintPredicate(fact_key="forbidden_status", operator="!=", item_field="lifecycle")
    item = WorkItem(
        item_id="node-v1",
        goal_id="cluster-goal-1",
        kind="server_node",
        title="Legacy Worker Node",
        data={"lifecycle": "deprecated", "ip": "10.0.0.12"},
        predicates=[pred],
    )

    # Value equals forbidden: stale
    status, reason = validate_work_item(item, {"forbidden_status": "deprecated"})
    assert status == WorkStatus.STALE
    assert reason is not None
    assert "must not equal" in reason

    # Value does not equal forbidden: valid
    status, reason = validate_work_item(item, {"forbidden_status": "terminated"})
    assert status == WorkStatus.VALID
    assert reason is None


def test_predicate_in_collection():
    """Verify in operator checks membership in list/tuple/set."""
    pred = ConstraintPredicate(fact_key="allowed_regions", operator="in", item_field="region")
    item = WorkItem(
        item_id="bucket-assets",
        goal_id="storage-goal-1",
        kind="storage_bucket",
        title="Media Assets S3 Bucket",
        data={"region": "eu-central-1", "storage_class": "STANDARD"},
        predicates=[pred],
    )

    # In allowed list: valid
    status, reason = validate_work_item(item, {"allowed_regions": ["us-east-1", "eu-central-1"]})
    assert status == WorkStatus.VALID
    assert reason is None

    # Not in allowed list: stale
    status, reason = validate_work_item(item, {"allowed_regions": ["us-east-1", "us-west-2"]})
    assert status == WorkStatus.STALE
    assert reason is not None
    assert "not in allowed" in reason


def test_predicate_contains():
    """Verify contains operator checks presence of item in collection or substring."""
    pred = ConstraintPredicate(fact_key="required_feature", operator="contains", item_field="features")
    item = WorkItem(
        item_id="saas-crm",
        goal_id="saas-goal-1",
        kind="saas_license",
        title="Enterprise CRM Plan",
        data={"features": ["sso", "audit_log", "scim", "custom_domains"]},
        predicates=[pred],
    )

    # Contains required feature: valid
    status, reason = validate_work_item(item, {"required_feature": "sso"})
    assert status == WorkStatus.VALID
    assert reason is None

    # Missing required feature: stale
    status, reason = validate_work_item(item, {"required_feature": "hipaa_compliance"})
    assert status == WorkStatus.STALE
    assert reason is not None
    assert "does not contain 'hipaa_compliance'" in reason


def test_predicate_missing_fact_passes_without_invalidation():
    """If referenced fact is absent from facts, predicate passes without marking item stale."""
    pred = ConstraintPredicate(fact_key="max_latency_ms", operator="<=", item_field="latency_ms")
    item = WorkItem(
        item_id="api-edge-gateway",
        goal_id="api-goal-1",
        kind="gateway",
        title="Edge Gateway Node",
        data={"latency_ms": 120},
        predicates=[pred],
    )

    # Fact not present in facts: remains VALID
    status, reason = validate_work_item(item, {"other_fact": "value"})
    assert status == WorkStatus.VALID
    assert reason is None


def test_predicate_missing_item_field_passes_without_invalidation():
    """If item data does not have the specified field, predicate passes without invalidation."""
    pred = ConstraintPredicate(fact_key="min_ram", operator=">=", item_field="ram_gb")
    item = WorkItem(
        item_id="serverless-fn",
        goal_id="fn-goal-1",
        kind="lambda",
        title="Serverless Handler",
        data={"concurrency": 100},  # No ram_gb field
        predicates=[pred],
    )

    status, reason = validate_work_item(item, {"min_ram": 16})
    assert status == WorkStatus.VALID
    assert reason is None


def test_multiple_predicates_identifies_exact_failing_predicate():
    """Verify that when one of multiple predicates fails, item is STALE and reason identifies it."""
    pred_budget = ConstraintPredicate(fact_key="max_hourly", operator="<=", item_field="hourly_cost")
    pred_ram = ConstraintPredicate(fact_key="min_ram", operator=">=", item_field="ram_gb")
    pred_region = ConstraintPredicate(fact_key="allowed_regions", operator="in", item_field="region")

    item = WorkItem(
        item_id="cloud-vm-prod",
        goal_id="vm-goal-1",
        kind="vm",
        title="Production Compute Node",
        data={"hourly_cost": 2.50, "ram_gb": 64, "region": "ap-south-1"},
        predicates=[pred_budget, pred_ram, pred_region],
    )

    facts = {
        "max_hourly": 5.00,  # 2.50 <= 5.00 (pass)
        "min_ram": 128,      # 64 >= 128 (FAIL)
        "allowed_regions": ["ap-south-1", "us-east-1"],  # in allowed (pass)
    }

    status, reason = validate_work_item(item, facts)
    assert status == WorkStatus.STALE
    assert reason is not None
    assert "ram_gb 64 is below required 128" in reason


def test_predicate_valid_stale_valid_reconciliation_cycle():
    """Verify non-destructive reconciliation transition VALID -> STALE -> VALID."""
    pred = ConstraintPredicate(fact_key="max_p99_ms", operator="<=", item_field="p99_latency")
    item = WorkItem(
        item_id="service-auth",
        goal_id="svc-goal-1",
        kind="microservice",
        title="Auth Microservice",
        data={"p99_latency": 80},
        predicates=[pred],
    )

    # Step 1: Initial valid fact (max_p99 = 100ms)
    res1 = reconcile_work_items([item], {"max_p99_ms": 100})
    assert len(res1.valid) == 1
    assert len(res1.stale) == 0
    assert item.status == WorkStatus.VALID

    # Step 2: Strict requirement (max_p99 = 50ms) -> becomes STALE
    res2 = reconcile_work_items([item], {"max_p99_ms": 50})
    assert len(res2.valid) == 0
    assert len(res2.stale) == 1
    assert len(res2.became_stale) == 1
    assert item.status == WorkStatus.STALE
    assert item.stale_reason is not None

    # Step 3: Relaxed requirement (max_p99 = 120ms) -> transitions back to VALID
    res3 = reconcile_work_items([item], {"max_p99_ms": 120})
    assert len(res3.valid) == 1
    assert len(res3.stale) == 0
    assert len(res3.became_valid) == 1
    assert item.status == WorkStatus.VALID
    assert item.stale_reason is None


def test_arbitrary_non_travel_meeting_room_domain():
    """Test meeting room booking with capacity and equipment constraints."""
    pred_capacity = ConstraintPredicate(fact_key="attendees", operator=">=", item_field="capacity")
    pred_equipment = ConstraintPredicate(fact_key="required_gear", operator="contains", item_field="amenities")

    room = WorkItem(
        item_id="boardroom-4b",
        goal_id="meeting-goal-1",
        kind="meeting_room",
        title="Executive Boardroom 4B",
        data={"capacity": 12, "amenities": ["video_conferencing", "whiteboard", "mic_array"]},
        predicates=[pred_capacity, pred_equipment],
    )

    # Valid: 12 >= 8 and contains "video_conferencing"
    status, reason = validate_work_item(room, {"attendees": 8, "required_gear": "video_conferencing"})
    assert status == WorkStatus.VALID
    assert reason is None

    # Stale: requires "vr_rig" which is missing
    status, reason = validate_work_item(room, {"attendees": 8, "required_gear": "vr_rig"})
    assert status == WorkStatus.STALE
    assert "does not contain 'vr_rig'" in reason


def test_mixed_legacy_and_generic_predicate_items_in_same_goal():
    """Verify goals containing both legacy items and generic predicate items reconcile seamlessly."""
    legacy_item = WorkItem(
        item_id="hotel-taj-mumbai",
        goal_id="mixed-goal",
        kind="hotel",
        title="Taj Hotel",
        data={"destination": "Mumbai", "price": 12000},
        depends_on=["destination", "budget"],
    )
    generic_item = WorkItem(
        item_id="rental-car-ev",
        goal_id="mixed-goal",
        kind="rental_car",
        title="Tesla Model 3 Rental",
        data={"range_km": 400, "daily_inr": 4000},
        predicates=[
            ConstraintPredicate(fact_key="min_range", operator=">=", item_field="range_km"),
            ConstraintPredicate(fact_key="max_car_daily", operator="<=", item_field="daily_inr"),
        ],
    )

    facts = {
        "destination": "Mumbai",
        "budget": 15000,
        "min_range": 300,
        "max_car_daily": 5000,
    }

    res = reconcile_work_items([legacy_item, generic_item], facts)
    assert len(res.valid) == 2
    assert len(res.stale) == 0

    # Budget drops: legacy hotel exceeds budget, car remains valid
    facts["budget"] = 10000
    res2 = reconcile_work_items([legacy_item, generic_item], facts)
    assert len(res2.valid) == 1
    assert res2.valid[0].item_id == "rental-car-ev"
    assert len(res2.stale) == 1
    assert res2.stale[0].item_id == "hotel-taj-mumbai"
