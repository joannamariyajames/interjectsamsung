"""Focused Step B integration tests: Fact correction & WorkItem invalidation lifecycle.

Verifies:
1. Nike -> Adidas correction lifecycle:
   - Fact asserted in BACKSPACE
   - Nike-dependent WorkItem registered
   - Product fact corrected to Adidas
   - BACKSPACE reports ChangeKind.CHANGED
   - Nike WorkItem transitions to STALE
   - Harness blocks the stale Nike WorkItem
   - New Adidas-dependent WorkItem registered against new fact ID and is not stale
2. Unaffected facts:
   - Unrelated facts (user_id, currency) remain CURRENT when product changes
   - Work dependent only on unaffected facts remains valid
3. Runtime multi-turn correction:
   - End-to-end multi-turn execution via AgentRuntime.on_final()
   - Turn 1 establishes Nike
   - Turn 2 corrects to Adidas
   - Verifies the resulting BACKSPACE Core and WorkItem lifecycle
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.backspace import (
    ChangeKind,
    DependencyKind,
    FactStatus,
    WorkItem,
    WorkStatus,
)
from app.extraction import ExtractedFact
from app.harness import Harness
from app.runtime import AgentRuntime
from app.session import Session
from tests.collector import Collector


def make_runtime() -> tuple[AgentRuntime, Collector, Session]:
    collector = Collector()
    session = Session(session_id="test-correction-session")
    return AgentRuntime(session, collector), collector, session


# ===========================================================================
# 1. Nike -> Adidas correction lifecycle
# ===========================================================================


@pytest.mark.asyncio
async def test_nike_to_adidas_correction_lifecycle():
    runtime, _, session = make_runtime()

    # 1. Create product="Nike shoes"
    r1 = runtime.observe_fact("product", "Nike shoes", source="user")
    assert r1.change_kind == ChangeKind.NEW
    nike_fact = session.backspace.get_fact("product")
    assert nike_fact is not None
    assert nike_fact.value == "Nike shoes"
    assert nike_fact.version == 1
    assert nike_fact.status == FactStatus.CURRENT

    # 2. Create/register Nike-dependent WorkItem
    work_nike = runtime._create_and_register_work("checkout", fact_keys=["product"])
    assert work_nike.status == WorkStatus.PENDING
    assert nike_fact.fact_id in work_nike.depends_on_facts

    # 3. Change product to "Adidas shoes"
    r2 = runtime.observe_fact("product", "Adidas shoes", source="user")

    # 4. Verify BACKSPACE reports CHANGED
    assert r2.change_kind == ChangeKind.CHANGED
    assert r2.changeset is not None
    assert r2.changeset.key == "product"
    assert r2.changeset.previous_fact is not None
    assert r2.changeset.previous_fact.fact_id == nike_fact.fact_id

    # 5. Verify Nike WorkItem becomes STALE
    assert work_nike.status == WorkStatus.STALE
    assert r2.invalidation is not None
    assert work_nike.work_id in r2.invalidation.invalidated_work_ids

    # 6. Verify Harness blocks the stale WorkItem
    harness = Harness(strict=True)
    budget = harness.new_budget()
    outcome = await harness.call("send_booking", budget, work=work_nike)
    assert outcome.status == "blocked"
    assert "stale" in outcome.verdict.lower()
    assert work_nike.work_id in outcome.verdict

    # 7. Create/register new Adidas-dependent WorkItem
    adidas_fact = session.backspace.get_fact("product")
    assert adidas_fact is not None
    assert adidas_fact.value == "Adidas shoes"
    assert adidas_fact.version == 2
    assert adidas_fact.status == FactStatus.CURRENT
    assert adidas_fact.fact_id != nike_fact.fact_id

    work_adidas = runtime._create_and_register_work("checkout", fact_keys=["product"])

    # 8. Verify new WorkItem uses the new product fact ID and is not stale
    assert work_adidas.work_id != work_nike.work_id
    assert work_adidas.status == WorkStatus.PENDING
    assert adidas_fact.fact_id in work_adidas.depends_on_facts
    assert nike_fact.fact_id not in work_adidas.depends_on_facts
    assert work_adidas.status != WorkStatus.STALE


# ===========================================================================
# 2. Unaffected facts preserved during correction
# ===========================================================================


def test_unaffected_facts_preserved_during_product_correction():
    runtime, _, session = make_runtime()

    # Seed user_id and currency
    r_user = runtime.observe_fact("user_id", "usr_1001", source="auth")
    r_curr = runtime.observe_fact("currency", "USD", source="settings")
    r_prod = runtime.observe_fact("product", "Nike shoes", source="user")

    assert r_user.change_kind == ChangeKind.NEW
    assert r_curr.change_kind == ChangeKind.NEW
    assert r_prod.change_kind == ChangeKind.NEW

    # Register work items: one depends on product, one depends on user_id
    work_product = runtime._create_and_register_work("checkout", fact_keys=["product"])
    work_user = runtime._create_and_register_work("user_profile", fact_keys=["user_id"])

    assert work_product.status == WorkStatus.PENDING
    assert work_user.status == WorkStatus.PENDING

    # Change only product
    r_corr = runtime.observe_fact("product", "Adidas shoes", source="user")
    assert r_corr.change_kind == ChangeKind.CHANGED

    # Verify user_id and currency remain current and unaffected
    user_fact = session.backspace.get_fact("user_id")
    curr_fact = session.backspace.get_fact("currency")

    assert user_fact is not None
    assert user_fact.status == FactStatus.CURRENT
    assert user_fact.version == 1
    assert user_fact.value == "usr_1001"

    assert curr_fact is not None
    assert curr_fact.status == FactStatus.CURRENT
    assert curr_fact.version == 1
    assert curr_fact.value == "USD"

    # Work depending on user_id remains PENDING/VALID
    assert work_user.status == WorkStatus.PENDING

    # Only product work becomes STALE
    assert work_product.status == WorkStatus.STALE


# ===========================================================================
# 3. Runtime multi-turn correction via on_final()
# ===========================================================================


@pytest.mark.asyncio
async def test_runtime_multi_turn_correction():
    collector = Collector()
    session = Session(session_id="multi-turn-correction-test")
    runtime = AgentRuntime(session, collector)

    async def mock_extract(utterance: str, goal_text: str):
        if "Nike" in utterance:
            return [ExtractedFact(key="product", value="Nike shoes", confidence=1.0)]
        elif "Adidas" in utterance:
            return [ExtractedFact(key="product", value="Adidas shoes", confidence=1.0)]
        return []

    with patch("app.runtime.extract_facts_with_fallback", side_effect=mock_extract):
        # Turn 1: user expresses Nike intent
        await runtime.on_final("I want to buy Nike shoes.")
        assert runtime._task is not None
        await runtime._task

        # Verify fact was asserted into BACKSPACE
        nike_fact = session.backspace.get_fact("product")
        assert nike_fact is not None
        assert nike_fact.value == "Nike shoes"
        assert nike_fact.version == 1

        # Register work dependent on product in Turn 1
        work_nike = runtime._create_and_register_work("checkout", fact_keys=["product"], turn_id="t1")
        assert work_nike.status == WorkStatus.PENDING
        assert nike_fact.fact_id in work_nike.depends_on_facts

        # Turn 2: user corrects to Adidas
        await runtime.on_final("Actually, I want Adidas shoes.")
        assert runtime._task is not None
        await runtime._task

        # Verify product was corrected in BACKSPACE
        adidas_fact = session.backspace.get_fact("product")
        assert adidas_fact is not None
        assert adidas_fact.value == "Adidas shoes"
        assert adidas_fact.version == 2
        assert adidas_fact.status == FactStatus.CURRENT

        # Verify previous Nike WorkItem became STALE through observation invalidation
        assert work_nike.status == WorkStatus.STALE

        # Verify Harness blocks the stale Nike WorkItem
        budget = runtime.harness.new_budget()
        outcome = await runtime.harness.call("send_booking", budget, work=work_nike)
        assert outcome.status == "blocked"
        assert "stale" in outcome.verdict.lower()

        # Create/register new Adidas-dependent work
        work_adidas = runtime._create_and_register_work("checkout", fact_keys=["product"], turn_id="t2")
        assert work_adidas.status == WorkStatus.PENDING
        assert adidas_fact.fact_id in work_adidas.depends_on_facts
        assert nike_fact.fact_id not in work_adidas.depends_on_facts
        assert work_adidas.status != WorkStatus.STALE


@pytest.mark.asyncio
async def test_runtime_multi_turn_with_unaffected_facts():
    collector = Collector()
    session = Session(session_id="multi-turn-unaffected-test")
    runtime = AgentRuntime(session, collector)

    # Pre-seed unaffected fact into session
    runtime.observe_fact("user_id", "u_42", source="system", turn_id="t0")
    user_fact = session.backspace.get_fact("user_id")
    assert user_fact is not None

    async def mock_extract(utterance: str, goal_text: str):
        if "Nike" in utterance:
            return [ExtractedFact(key="product", value="Nike shoes", confidence=1.0)]
        elif "Adidas" in utterance:
            return [ExtractedFact(key="product", value="Adidas shoes", confidence=1.0)]
        return []

    with patch("app.runtime.extract_facts_with_fallback", side_effect=mock_extract):
        # Turn 1
        await runtime.on_final("I want to buy Nike shoes.")
        await runtime._task

        work_nike = runtime._create_and_register_work("checkout", fact_keys=["product"], turn_id="t1")
        work_user = runtime._create_and_register_work("user_sync", fact_keys=["user_id"], turn_id="t1")

        # Turn 2: correction
        await runtime.on_final("Actually, I want Adidas shoes.")
        await runtime._task

        # Verify product updated and work_nike became STALE
        assert session.backspace.get_fact("product").value == "Adidas shoes"
        assert work_nike.status == WorkStatus.STALE

        # Verify unaffected fact and work remain intact
        u_fact_after = session.backspace.get_fact("user_id")
        assert u_fact_after.version == 1
        assert u_fact_after.value == "u_42"
        assert u_fact_after.status == FactStatus.CURRENT
        assert work_user.status == WorkStatus.PENDING
