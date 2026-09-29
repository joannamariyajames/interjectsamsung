"""The demo knowledge base must read as a fictional agency's own data, never as
real-world facts, and must keep the shape the interruption machinery relies on."""

from __future__ import annotations

import re

from app.headsup import detect_contradiction
from app.retrieval import corpus
from app.work import _extract_hotel_tuples


def test_every_passage_is_scoped_to_the_fictional_agency() -> None:
    for doc in corpus.docs:
        assert "Interject Travel" in doc.text, f"{doc.doc_id} ({doc.title}) reads as a real-world fact"


def test_no_invented_schedules_or_flight_counts() -> None:
    text = " ".join(d.text for d in corpus.docs).lower()
    assert not re.search(r"\b\d+ daily departures\b", text)
    assert "on-time" not in text and "block time" not in text
    schedules = next(d for d in corpus.docs if d.title == "Schedules and live fares")
    assert "keeps no flight schedules" in schedules.text and "live" in schedules.text


def test_policies_apply_only_to_the_agencys_corporate_clients() -> None:
    approval = next(d for d in corpus.docs if d.title == "Approval thresholds")
    assert "corporate clients enrolled with Interject Travel" in approval.text
    assert "never to anyone else by default" in approval.text


def test_document_ids_and_order_are_unchanged() -> None:
    assert [d.doc_id for d in corpus.docs][:4] == ["flights#0", "flights#1", "flights#2", "flights#3"]
    assert [d.title for d in corpus.docs] == [
        "Cabin classes", "Change and cancellation windows", "Baggage", "Schedules and live fares",
        "Rate types", "Properties", "Check-in rules",
        "Approval thresholds", "Per diem", "Reimbursement", "Refunds",
        "Delays", "Missed connections", "Contact", "Insurance",
    ]


def test_hotel_extraction_and_heads_up_still_work_on_the_snippets() -> None:
    docs = {d.doc_id: d for d in corpus.docs}
    hotels = _extract_hotel_tuples(docs["hotels#5"].snippet)
    assert ("The Harbour House", "Mumbai", 8900) in hotels
    evidence = [{"doc_id": d.doc_id, "title": d.title, "snippet": d.snippet} for d in (docs["flights#0"], docs["flights#2"])]
    refund = detect_contradiction("Book Economy Saver because it's fully refundable.", evidence, {})
    assert refund is not None and refund.source_doc_id == "flights#0"
    baggage = detect_contradiction("Book Economy Saver since it includes 25 kg checked baggage.", evidence, {})
    assert baggage is not None and baggage.source_doc_id == "flights#2"
