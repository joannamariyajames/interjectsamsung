from __future__ import annotations

from typing import Any

from app.headsup import (
    ContradictionRegistry,
    ContradictionRule,
    HeadsUpEvent,
    default_contradiction_registry,
    detect_contradiction,
)
from app.headsup_travel import (
    BaggageContradictionRule,
    BudgetContradictionRule,
    RefundabilityContradictionRule,
    create_travel_registry,
    register_travel_rules,
)
from app.runtime import AgentRuntime
from app.session import Session
from tests.collector import Collector


# ----------------------------------------------------------------------
# A. Positive contradiction tests
# ----------------------------------------------------------------------


def test_positive_refundability_contradiction():
    evidence = [
        {
            "doc_id": "flights#0",
            "title": "Cabin classes",
            "snippet": "Economy Saver is the cheapest bucket. It is non-refundable but changeable once for a fee.",
            "source": "flights",
        }
    ]
    utterance = "Book Economy Saver because it's fully refundable."
    event = detect_contradiction(utterance, evidence, {})

    assert event is not None
    assert event.claim == "Economy Saver is refundable"
    assert "non-refundable" in event.contradiction
    assert event.confidence >= 0.90
    assert event.source_doc_id == "flights#0"
    assert "Hold on" in event.cut_in_text
    assert "non-refundable" in event.cut_in_text


def test_positive_baggage_contradiction():
    evidence = [
        {
            "doc_id": "flights#2",
            "title": "Baggage",
            "snippet": "Economy Saver includes 15 kg checked and 7 kg cabin. Economy Flex includes 25 kg checked.",
            "source": "flights",
        }
    ]
    utterance = "Book Economy Saver since it includes 25 kg checked baggage."
    event = detect_contradiction(utterance, evidence, {})

    assert event is not None
    assert "25 kg" in event.claim
    assert "15 kg" in event.contradiction
    assert event.confidence >= 0.90
    assert event.source_doc_id == "flights#2"
    assert "Hold on" in event.cut_in_text
    assert "15 kg" in event.cut_in_text


def test_positive_budget_contradiction():
    evidence = [
        {
            "doc_id": "hotels#5",
            "title": "Properties",
            "snippet": "The Harbour House in Mumbai sits 20 minutes from the domestic terminal, rates from 8,900 INR, rooftop pool.",
            "source": "hotels",
        }
    ]
    facts = {"budget": 8000, "destination": "Mumbai"}
    utterance = "The Harbour House fits our budget so let's book it."
    event = detect_contradiction(utterance, evidence, facts)

    assert event is not None
    assert "The Harbour House" in event.claim
    assert "8,900" in event.contradiction
    assert "8,000" in event.contradiction
    assert event.confidence >= 0.90
    assert event.source_doc_id == "hotels#5"
    assert "8,900" in event.cut_in_text


def test_advance_purchase_non_refundable_contradiction():
    evidence = [
        {
            "doc_id": "hotels#0",
            "title": "Rate types",
            "snippet": "Advance Purchase rates are 22 percent cheaper and non-refundable.",
            "source": "hotels",
        }
    ]
    utterance = "Advance Purchase is refundable if plans change."
    event = detect_contradiction(utterance, evidence, {})

    assert event is not None
    assert event.claim == "Advance Purchase is refundable"
    assert "non-refundable" in event.contradiction


# ----------------------------------------------------------------------
# B. False-positive tests (MUST return None)
# ----------------------------------------------------------------------


def test_question_is_economy_saver_refundable_returns_none():
    evidence = [
        {
            "doc_id": "flights#0",
            "title": "Cabin classes",
            "snippet": "Economy Saver is the cheapest bucket. It is non-refundable.",
            "source": "flights",
        }
    ]
    assert detect_contradiction("Is Economy Saver refundable?", evidence, {}) is None


def test_question_what_are_baggage_limits_returns_none():
    evidence = [
        {
            "doc_id": "flights#2",
            "title": "Baggage",
            "snippet": "Economy Saver includes 15 kg checked.",
            "source": "flights",
        }
    ]
    assert detect_contradiction("What are the baggage limits?", evidence, {}) is None


def test_question_can_i_cancel_saver_returns_none():
    evidence = [
        {
            "doc_id": "flights#0",
            "title": "Cabin classes",
            "snippet": "Economy Saver is non-refundable.",
            "source": "flights",
        }
    ]
    assert detect_contradiction("Can I cancel Economy Saver?", evidence, {}) is None


def test_hedged_maybe_refundable_returns_none():
    evidence = [
        {
            "doc_id": "flights#0",
            "title": "Cabin classes",
            "snippet": "Economy Saver is non-refundable.",
            "source": "flights",
        }
    ]
    assert detect_contradiction("Maybe Economy Saver is refundable.", evidence, {}) is None
    assert detect_contradiction("I think Saver might be refundable.", evidence, {}) is None


def test_unsupported_unknown_claims_return_none():
    evidence = [
        {
            "doc_id": "hotels#5",
            "title": "Properties",
            "snippet": "Lantern Court in Bengaluru is 35 minutes from the airport.",
            "source": "hotels",
        }
    ]
    assert detect_contradiction("The hotel has a tennis court.", evidence, {}) is None


def test_claims_consistent_with_evidence_return_none():
    evidence = [
        {
            "doc_id": "flights#0",
            "title": "Cabin classes",
            "snippet": "Economy Flex is fully refundable up to 4 hours before departure.",
            "source": "flights",
        }
    ]
    # Flex IS refundable, so asserting it is refundable is completely consistent
    assert detect_contradiction("Economy Flex is fully refundable.", evidence, {}) is None


# ----------------------------------------------------------------------
# C. Threshold test
# ----------------------------------------------------------------------


def test_threshold_gating_returns_none():
    evidence = [
        {
            "doc_id": "flights#0",
            "title": "Cabin classes",
            "snippet": "Economy Saver is non-refundable.",
            "source": "flights",
        }
    ]
    utterance = "Book Economy Saver because it's fully refundable."
    # If required threshold is higher than confidence (e.g. 1.05), it must return None
    assert detect_contradiction(utterance, evidence, {}, threshold=1.05) is None


# ----------------------------------------------------------------------
# D. Runtime integration test
# ----------------------------------------------------------------------


async def test_runtime_heads_up_cut_in_emits_frame_and_surfaces_correction():
    session = Session("test-headsup-runtime")
    collector = Collector()
    runtime = AgentRuntime(session, collector)

    # User assertion that contradicts corpus (Economy Saver is non-refundable)
    await runtime.on_final("Book Economy Saver because it's fully refundable.")
    await runtime._task

    # 1. HeadsUpFrame was emitted
    headsup_frames = collector.of("headsup")
    assert len(headsup_frames) >= 1
    frame = headsup_frames[0]
    assert frame["claim"] == "Economy Saver is refundable"
    assert "non-refundable" in frame["contradiction"]
    assert "Hold on" in frame["cut_in_text"]

    # 2. Stage includes headsup
    assert "headsup" in collector.stages()

    # 3. Agent response starts with the cut-in correction, not proceeding blindly on false premise
    agent_messages = [m for m in collector.of("message") if m.get("role") == "agent"]
    assert agent_messages
    last_msg = agent_messages[-1]
    assert "Hold on — Economy Saver fares are non-refundable" in last_msg["content"]
    assert last_msg["meta"]["headsup"] is not None
    assert last_msg["meta"]["headsup"]["claim"] == "Economy Saver is refundable"


# ----------------------------------------------------------------------
# E. Bookmark regression test
# ----------------------------------------------------------------------


async def test_heads_up_does_not_corrupt_bookmark_state():
    session = Session("test-bookmark-regression")
    collector = Collector()
    runtime = AgentRuntime(session, collector)

    # Turn 1: Create a goal with work items
    await runtime.on_final("Book hotel in Mumbai for two under 20000.")
    await runtime._task

    goal = session.goals.active
    assert goal is not None
    assert len(goal.work_items) >= 1
    assert session.backspace.get_fact("budget") is not None
    assert session.backspace.get_fact("budget").value == 20000
    work_items_count = len(goal.work_items)

    # Turn 2: User makes a contradictory claim on another topic
    await runtime.on_final("Book Economy Saver because it is refundable.")
    await runtime._task

    # Turn 2 triggered Heads-Up
    assert collector.of("headsup")

    # Turn 3: Bookmark state must remain intact
    # Facts are not erased
    assert session.backspace.get_fact("budget") is not None
    assert session.backspace.get_fact("budget").value == 20000
    # Resuming original goal
    await runtime.on_final("anyway, back to the hotel")
    await runtime._task

    resumed_goal = session.goals.active
    assert resumed_goal is not None
    # Goal work items are preserved and not deleted
    assert len(resumed_goal.work_items) == work_items_count


# ----------------------------------------------------------------------
# F. Hotel budget specific variations & end-to-end flow tests
# ----------------------------------------------------------------------


def test_hotel_budget_contradiction_variations():
    evidence = [
        {
            "doc_id": "hotels#5",
            "title": "Properties",
            "snippet": "The Harbour House in Mumbai sits 20 minutes from the domestic terminal, rates from 8,900 INR, rooftop pool.",
            "source": "hotels",
        }
    ]
    facts = {"budget": 8000, "destination": "Mumbai"}
    variations = [
        "Book The Harbour House in Mumbai, it is within my 8k budget.",
        "Book The Harbour House, it's within my 8k budget.",
        "The Harbour House is under 8000.",
        "Harbour House fits my 8k budget.",
        "Book that hotel, it is within my ₹8,000 budget.",
    ]

    for utterance in variations:
        event = detect_contradiction(utterance, evidence, facts)
        assert event is not None, f"Failed on variation: {utterance}"
        assert "The Harbour House" in event.claim or "within budget" in event.claim
        assert "8,900" in event.contradiction
        assert "8,000" in event.cut_in_text
        assert event.confidence >= 0.90
        assert event.source_doc_id == "hotels#5"


def test_hotel_budget_false_positives_return_none():
    evidence = [
        {
            "doc_id": "hotels#5",
            "title": "Properties",
            "snippet": "The Harbour House in Mumbai sits 20 minutes from the domestic terminal, rates from 8,900 INR, rooftop pool.",
            "source": "hotels",
        }
    ]
    facts = {"budget": 8000, "destination": "Mumbai"}

    # Questions and hedges must return None
    assert detect_contradiction("Is The Harbour House within my 8k budget?", evidence, facts) is None
    assert detect_contradiction("I think The Harbour House might be within my 8k budget.", evidence, facts) is None
    # Claim consistent with 20k budget must return None
    assert detect_contradiction("The Harbour House is within my 20k budget.", evidence, {"budget": 20000}) is None


async def test_hotel_budget_end_to_end_flow_and_constraint_integrity():
    session = Session("test-hotel-e2e")
    collector = Collector()
    runtime = AgentRuntime(session, collector)

    # Turn 1: Initial goal with 20000 budget
    await runtime.on_final("Book a hotel in Mumbai for two people under 20000")
    await runtime._task
    assert session.backspace.get_fact("budget") is not None
    assert session.backspace.get_fact("budget").value == 20000
    assert session.backspace.get_fact("destination") is not None
    assert session.backspace.get_fact("destination").value == "Mumbai"
    assert session.backspace.get_fact("party_size") is not None
    assert session.backspace.get_fact("party_size").value == 2
    g1 = session.goals.active
    assert g1 is not None
    assert len(g1.work_items) >= 1

    # Turn 2: Budget update to 8000
    await runtime.on_final("Actually, my budget is 8000 now")
    await runtime._task
    assert session.backspace.get_fact("budget") is not None
    assert session.backspace.get_fact("budget").value == 8000

    # Turn 3: Contradictory assertion that Harbour House (8,900) is within 8k budget
    await runtime.on_final("Book The Harbour House in Mumbai, it is within my 8k budget.")
    await runtime._task

    # 1. HeadsUpFrame was emitted
    headsup_frames = collector.of("headsup")
    assert len(headsup_frames) >= 1
    last_headsup = headsup_frames[-1]
    assert "The Harbour House" in last_headsup["claim"]
    assert "8,900" in last_headsup["contradiction"]
    assert "8,000" in last_headsup["cut_in_text"]

    # 2. Stage updated to headsup
    assert "headsup" in collector.stages()

    # 3. Cut-in notice surfaced in response
    agent_msgs = [m for m in collector.of("message") if m.get("role") == "agent"]
    assert agent_msgs
    turn3_msg = agent_msgs[-1]
    assert "Hold on — The Harbour House starts at ₹8,900" in turn3_msg["content"]
    assert turn3_msg["meta"]["headsup"] is not None

    # 4. Constraints must NOT be malformed (e.g. no "within my")
    for g in session.goals.stack:
        for c in g.constraints:
            assert c != "within my", f"Malformed constraint found: {c}"
            assert not c.endswith(" my"), f"Malformed constraint found: {c}"

    # 5. Budget fact must be preserved at 8000, not reverted to 20000
    assert session.backspace.get_fact("budget") is not None
    assert session.backspace.get_fact("budget").value == 8000

    # Turn 4: Resume
    await runtime.on_final("anyway, back to the hotel")
    await runtime._task

    resumed = session.goals.active
    assert resumed is not None
    assert "Mumbai" in resumed.text
    assert session.backspace.get_fact("budget") is not None
    assert session.backspace.get_fact("budget").value == 8000
    assert len(resumed.work_items) >= 1


# ----------------------------------------------------------------------
# G. Phase A.1 ContradictionRegistry & Generic Rule Extensibility Tests
# ----------------------------------------------------------------------


class ServerRAMContradictionRule:
    """Generic non-travel test rule for server hardware contradictions."""

    name: str = "server_ram"

    def check(self, utterance: str, evidence: Any, facts: dict[str, Any]) -> HeadsUpEvent | None:
        import re

        m = re.search(r"server\s+(\w+)\s+has\s+(\d+)\s*gb", utterance.lower())
        if not m:
            return None
        server_id, claimed_ram = m.group(1), int(m.group(2))
        for doc in (evidence or []):
            content = doc.get("snippet") or doc.get("text") or ""
            doc_id = doc.get("doc_id") or "servers#0"
            m_doc = re.search(rf"server\s+{server_id}\s+equipped\s+with\s+(\d+)\s*gb", content.lower())
            if m_doc:
                actual_ram = int(m_doc.group(1))
                if claimed_ram != actual_ram:
                    return HeadsUpEvent(
                        claim=f"Server {server_id} has {claimed_ram}GB RAM",
                        contradiction=f"Server {server_id} is equipped with {actual_ram}GB RAM",
                        confidence=1.0,
                        source_doc_id=doc_id,
                        cut_in_text=f"Hold on — Server {server_id} actually has {actual_ram}GB RAM, not {claimed_ram}GB.",
                    )
        return None


def test_core_engine_has_no_travel_rules_by_default():
    """Verify core contradiction engine is completely domain-generic and unopinionated by default."""
    reg = ContradictionRegistry()
    assert len(reg.rules) == 0
    assert reg.check("Book Economy Saver because it's fully refundable.", [], {}) is None


def test_travel_registry_contains_travel_rules():
    """Verify travel extension registry correctly bundles travel rules."""
    reg = create_travel_registry()
    rule_names = [r.name for r in reg.rules]
    assert "refundability" in rule_names
    assert "baggage" in rule_names
    assert "budget" in rule_names
    assert len(reg.rules) == 3


def test_registry_fallback_when_rule_returns_none():
    """Verify that when earlier rules return None, subsequent registered rules run."""
    reg = ContradictionRegistry()
    reg.register(RefundabilityContradictionRule())
    reg.register(BaggageContradictionRule())

    # Baggage utterance will return None from RefundabilityContradictionRule and match BaggageContradictionRule
    evidence = [
        {
            "doc_id": "flights#2",
            "title": "Baggage",
            "snippet": "Economy Saver includes 15 kg checked.",
            "source": "flights",
        }
    ]
    utterance = "Book Economy Saver since it includes 30 kg checked baggage."
    event = reg.check(utterance, evidence, {})
    assert event is not None
    assert "30 kg" in event.claim
    assert "15 kg" in event.contradiction


def test_registry_deterministic_first_match_ordering():
    """Verify that the first matching rule in registration order is returned deterministically."""
    reg = ContradictionRegistry()

    class RuleA:
        name = "rule_a"

        def check(self, utterance, evidence, facts):
            return HeadsUpEvent("claim A", "contra A", 1.0, "doc#A", "cut A")

    class RuleB:
        name = "rule_b"

        def check(self, utterance, evidence, facts):
            return HeadsUpEvent("claim B", "contra B", 1.0, "doc#B", "cut B")

    reg.register(RuleA())
    reg.register(RuleB())

    event = reg.check("test assertion", [], {})
    assert event is not None
    assert event.claim == "claim A"


def test_custom_server_ram_rule_in_registry():
    """Verify that an arbitrary non-travel rule functions seamlessly through ContradictionRegistry."""
    reg = ContradictionRegistry()
    reg.register(ServerRAMContradictionRule())

    evidence = [
        {
            "doc_id": "compute#prod-1",
            "snippet": "Server prod1 equipped with 32 GB RAM and 8 vCPUs.",
        }
    ]

    # Contradictory assertion: prod1 has 64GB
    event = reg.check("We know server prod1 has 64 GB RAM so it should handle the load.", evidence, {})
    assert event is not None
    assert "64GB RAM" in event.claim
    assert "32GB RAM" in event.contradiction
    assert event.source_doc_id == "compute#prod-1"
    assert "Hold on" in event.cut_in_text


def test_detect_contradiction_accepts_custom_registry():
    """Verify detect_contradiction() dispatches to custom registry and still enforces false-positive gating."""
    reg = ContradictionRegistry()
    reg.register(ServerRAMContradictionRule())

    evidence = [{"doc_id": "compute#prod-1", "snippet": "Server prod1 equipped with 32 GB RAM."}]

    # Affirmative contradiction fires
    event = detect_contradiction(
        "Server prod1 has 64 GB RAM.",
        evidence,
        {},
        registry=reg,
    )
    assert event is not None
    assert "32GB" in event.contradiction

    # Question with same keywords returns None (gated before registry check)
    assert detect_contradiction(
        "Does server prod1 have 64 GB RAM?",
        evidence,
        {},
        registry=reg,
    ) is None

    # Hedged statement returns None (gated before registry check)
    assert detect_contradiction(
        "I think server prod1 has 64 GB RAM.",
        evidence,
        {},
        registry=reg,
    ) is None

    # Consistent statement returns None (rule returns None)
    assert detect_contradiction(
        "Server prod1 has 32 GB RAM.",
        evidence,
        {},
        registry=reg,
    ) is None


def test_registry_unregister_and_clear():
    reg = create_travel_registry()
    assert len(reg.rules) == 3

    reg.unregister("baggage")
    assert [r.name for r in reg.rules] == ["refundability", "budget"]

    reg.clear()
    assert len(reg.rules) == 0
    assert reg.check("Book Economy Saver because it's fully refundable.", [], {}) is None


