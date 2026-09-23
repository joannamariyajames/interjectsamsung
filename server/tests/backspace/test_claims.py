from __future__ import annotations

import pytest

from app.backspace import Claim, ClaimStatus, Evidence


def test_evidence_matches_the_existing_runtime_shape():
    raw = {
        "doc_id": "flights#0",
        "title": "Fares",
        "snippet": "Economy fares start at...",
        "source": "flights",
    }
    evidence = Evidence.from_dict(raw)
    assert evidence.to_dict() == raw


def test_evidence_from_dict_tolerates_missing_optional_fields():
    evidence = Evidence.from_dict({"doc_id": "policy#1"})
    assert evidence.doc_id == "policy#1"
    assert evidence.title == ""
    assert evidence.snippet == ""
    assert evidence.source == ""


def test_claim_creation_defaults():
    claim = Claim(text="Baggage is 15kg in economy.")
    assert claim.text == "Baggage is 15kg in economy."
    assert claim.status is ClaimStatus.GENERATED
    assert claim.evidence is None
    assert claim.depends_on_facts == []
    assert claim.goal_id is None
    assert claim.claim_id


def test_claim_can_carry_evidence_and_fact_dependencies():
    evidence = Evidence(doc_id="hotels#2", title="Properties", snippet="...", source="hotels")
    claim = Claim(
        text="A room under 9000 is available.",
        work_id="w1",
        turn_id="t1",
        goal_id="g1",
        evidence=evidence,
        depends_on_facts=["f1", "f2"],
    )
    assert claim.evidence is evidence
    assert claim.depends_on_facts == ["f1", "f2"]
    assert claim.work_id == "w1"


def test_claim_lifecycle_states_are_all_representable():
    assert {s.value for s in ClaimStatus} == {
        "generated", "spoken", "heard", "valid",
        "invalidated", "retracted", "superseded",
    }


def test_claim_status_rejects_unknown_value():
    with pytest.raises(ValueError):
        ClaimStatus("bogus")


def test_claim_serialization_is_deterministic_for_identical_input():
    evidence = Evidence(doc_id="d1", title="T", snippet="S", source="src")
    kwargs = dict(
        text="x", claim_id="c1", work_id="w1", turn_id="t1", goal_id="g1",
        evidence=evidence, status=ClaimStatus.SPOKEN, depends_on_facts=["f1"],
        created_at=10.0, spoken_at=11.0, provenance={"a": 1},
    )
    first = Claim(**kwargs).to_dict()
    second = Claim(**kwargs).to_dict()
    assert first == second
    assert first["evidence"] == evidence.to_dict()
