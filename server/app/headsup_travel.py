"""Travel domain contradiction rules for Heads-Up early cut-in.

Optional extension layer that registers travel-specific contradiction rules
(Economy Saver refundability, baggage limits, hotel budget overruns) into the
core ContradictionRegistry.
"""

from __future__ import annotations

import re
from typing import Any

from .headsup import (
    ContradictionRegistry,
    ContradictionRule,
    HeadsUpEvent,
    default_contradiction_registry,
)
from .retrieval import corpus
from .work import _extract_hotel_tuples


def _get_expanded_doc_content(doc: dict[str, Any]) -> str:
    content = doc.get("snippet") or doc.get("text") or doc.get("body") or ""
    doc_id = doc.get("doc_id") or doc.get("id")
    if doc_id and len(content) < 30:
        for cd in corpus.docs:
            if cd.doc_id == doc_id:
                content = cd.text
                break
    return str(content)


class RefundabilityContradictionRule:
    """Detects explicit claims that non-refundable fares/rates are refundable."""

    name: str = "refundability"

    def check(
        self,
        utterance: str,
        evidence: Any,
        facts: dict[str, Any],
    ) -> HeadsUpEvent | None:
        cleaned_utterance = utterance.strip().lower()

        # Must NOT contain negative words (non-refundable, not refundable, no refund)
        if "non-refundable" in cleaned_utterance or "not refundable" in cleaned_utterance or "no refund" in cleaned_utterance:
            return None

        refundable_claimed = bool(re.search(
            r"\b(?:is|are|allows?|gives?|includes?|with|has)?\s*(?:fully\s+|full\s+)?refundable\b|\bcan\s+(?:be\s+)?refunded\b|\ballows?\s+cancellation\b",
            cleaned_utterance,
        ))
        if not refundable_claimed:
            return None

        # 1. Economy Saver
        saver_claimed = bool(re.search(r"\b(?:economy\s+)?saver\b", cleaned_utterance))
        if saver_claimed:
            for doc in (evidence or []):
                doc_text = _get_expanded_doc_content(doc).lower()
                doc_id = doc.get("doc_id") or "flights#0"
                if "economy saver" in doc_text and "non-refundable" in doc_text:
                    return HeadsUpEvent(
                        claim="Economy Saver is refundable",
                        contradiction="Economy Saver is non-refundable",
                        confidence=1.0,
                        source_doc_id=doc_id,
                        cut_in_text="Hold on — Economy Saver fares are non-refundable.",
                    )

        # 2. Advance Purchase rates (hotels)
        advance_claimed = bool(re.search(r"\badvance\s+purchase\b", cleaned_utterance))
        if advance_claimed:
            for doc in (evidence or []):
                doc_text = _get_expanded_doc_content(doc).lower()
                doc_id = doc.get("doc_id") or "hotels#0"
                if "advance purchase" in doc_text and "non-refundable" in doc_text:
                    return HeadsUpEvent(
                        claim="Advance Purchase is refundable",
                        contradiction="Advance Purchase rates are non-refundable",
                        confidence=1.0,
                        source_doc_id=doc_id,
                        cut_in_text="Hold on — Advance Purchase rates are non-refundable.",
                    )

        return None


class BaggageContradictionRule:
    """Detects claims of baggage allowance for Economy Saver exceeding 15 kg."""

    name: str = "baggage"

    def check(
        self,
        utterance: str,
        evidence: Any,
        facts: dict[str, Any],
    ) -> HeadsUpEvent | None:
        cleaned_utterance = utterance.strip().lower()

        baggage_match = re.search(
            r"\b(?:economy\s+)?saver\b[^.\n]*?\b(?:includes?|has|with|allows?|gives?)\s+([0-9]+)\s*(?:kg|kilos|kilograms)\b",
            cleaned_utterance,
        )
        if not baggage_match:
            baggage_match = re.search(
                r"\b([0-9]+)\s*(?:kg|kilos|kilograms)\b[^.\n]*?\b(?:economy\s+)?saver\b",
                cleaned_utterance,
            )

        if baggage_match:
            claimed_kg = int(baggage_match.group(1))
            if claimed_kg > 15:
                for doc in (evidence or []):
                    doc_text = _get_expanded_doc_content(doc).lower()
                    doc_id = doc.get("doc_id") or "flights#2"
                    if "economy saver" in doc_text and "15 kg" in doc_text:
                        return HeadsUpEvent(
                            claim=f"Economy Saver includes {claimed_kg} kg",
                            contradiction="Economy Saver includes 15 kg checked",
                            confidence=1.0,
                            source_doc_id=doc_id,
                            cut_in_text=f"Hold on — Economy Saver only includes 15 kg checked baggage, not {claimed_kg} kg.",
                        )

        return None


class BudgetContradictionRule:
    """Detects hotel budget overruns contradicting explicit limits or session budget."""

    name: str = "budget"

    def check(
        self,
        utterance: str,
        evidence: Any,
        facts: dict[str, Any],
    ) -> HeadsUpEvent | None:
        cleaned_utterance = utterance.strip().lower()
        budget_val = facts.get("budget")

        # 1. Explicit amount claimed for the hotel / budget
        amount_match = re.search(
            r"\b(?:under|below|within|cheaper\s+than|less\s+than|up\s+to|max|maximum|fits?(?:\s+in|\s+into|\s+within|\s+under)?)\s+"
            r"(?:(?:our|my|the)\s+)?(?:budget\s+(?:of|is)?\s*)?"
            r"(?:[₹\u20b9]|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\b",
            cleaned_utterance,
        )
        claimed_limit: int | None = None
        if amount_match:
            val = float(amount_match.group(1).replace(",", ""))
            if amount_match.group(2):
                val *= 1000
            claimed_limit = int(val)

        # 2. General budget keyword claims (e.g. "fits our budget", "within my budget", "is under budget")
        budget_keyword_claim = bool(re.search(
            r"\b(?:fits?(?:\s+in|\s+into|\s+within|\s+under)?|is\s+within|is\s+under|works?\s+because\s+it(?:'s|\s+is)\s+(?:under|within))\s+"
            r"(?:(?:our|my|the)\s+)?(?:[\d,k₹\u20b9rs.inr]+\s+)?budget\b",
            cleaned_utterance,
        ))

        if claimed_limit is not None or budget_keyword_claim:
            for doc in (evidence or []):
                content = _get_expanded_doc_content(doc)
                doc_id = doc.get("doc_id") or ""
                tuples = _extract_hotel_tuples(content)
                for name, city, price in tuples:
                    name_clean = re.sub(r"^the\s+", "", name, flags=re.IGNORECASE).strip().lower()
                    refers_to_hotel = (name_clean in cleaned_utterance) or bool(
                        re.search(r"\b(?:that|the|this)\s+(?:hotel|property)\b|\bbook\s+it\b", cleaned_utterance)
                    )
                    if not refers_to_hotel:
                        continue

                    contradiction_found = False
                    effective_limit: int | None = None

                    if claimed_limit is not None and price > claimed_limit:
                        contradiction_found = True
                        effective_limit = claimed_limit
                    elif claimed_limit is None and budget_keyword_claim and budget_val is not None and price > budget_val:
                        contradiction_found = True
                        effective_limit = budget_val

                    if contradiction_found:
                        limit_to_show = budget_val if budget_val is not None else effective_limit
                        limit_str = f"₹{limit_to_show:,}" if limit_to_show else ""
                        return HeadsUpEvent(
                            claim=f"{name} fits within budget",
                            contradiction=f"{name} rates start at ₹{price:,}, which exceeds budget of {limit_str}",
                            confidence=1.0,
                            source_doc_id=doc_id,
                            cut_in_text=f"Hold on — {name} starts at ₹{price:,}, which is above your {limit_str} budget."
                            if budget_val
                            else f"Hold on — {name} starts at ₹{price:,}, which exceeds {limit_str}.",
                        )

        return None


def register_travel_rules(registry: ContradictionRegistry | None = None) -> None:
    """Register travel contradiction rules into the target registry (defaulting to global registry)."""
    target = registry if registry is not None else default_contradiction_registry
    existing_names = {r.name for r in target.rules}
    if "refundability" not in existing_names:
        target.register(RefundabilityContradictionRule())
    if "baggage" not in existing_names:
        target.register(BaggageContradictionRule())
    if "budget" not in existing_names:
        target.register(BudgetContradictionRule())


def create_travel_registry() -> ContradictionRegistry:
    """Create a new ContradictionRegistry pre-populated with travel rules."""
    reg = ContradictionRegistry()
    register_travel_rules(reg)
    return reg
