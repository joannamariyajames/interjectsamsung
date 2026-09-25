"""Persistent WorkItem model for BACKSPACE.

Tracks discrete units of agent work (retrievals, quotes, options, steps)
and their dependency on session facts so that when facts change, only
affected work is invalidated (marked STALE) rather than restarting the entire goal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkStatus(str, Enum):
    VALID = "valid"
    STALE = "stale"
    RETRACTED = "retracted"


@dataclass
class ConstraintPredicate:
    fact_key: str
    operator: str  # "<=", ">=", "==", "!=", "in", "contains"
    item_field: str | None = None
    target_value: Any | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_key": self.fact_key,
            "operator": self.operator,
            "item_field": self.item_field,
            "target_value": self.target_value,
            "description": self.description,
        }

    def evaluate(self, item_data: dict[str, Any], facts: dict[str, Any]) -> tuple[bool, str | None]:
        """Evaluate the predicate against item data and active facts.

        Returns:
            (is_valid: bool, stale_reason: str | None)
        """
        # Determine fact value
        if self.fact_key in facts and facts[self.fact_key] is not None:
            fact_val = facts[self.fact_key]
        elif self.target_value is not None:
            fact_val = self.target_value
        else:
            # Fact is absent and no target_value fallback: passes without invalidation
            return True, None

        field_name = self.item_field or self.fact_key
        if field_name not in item_data or item_data[field_name] is None:
            # Item field is absent: passes without invalidation
            return True, None

        item_val = item_data[field_name]
        op = self.operator.strip().lower()

        try:
            if op == "<=":
                if item_val > fact_val:
                    reason = (
                        self.description
                        or f"{field_name} {item_val} exceeds limit {fact_val}"
                    )
                    return False, reason
            elif op == ">=":
                if item_val < fact_val:
                    reason = (
                        self.description
                        or f"{field_name} {item_val} is below required {fact_val}"
                    )
                    return False, reason
            elif op == "==":
                if isinstance(item_val, str) and isinstance(fact_val, str):
                    is_valid = item_val.strip().lower() == fact_val.strip().lower()
                else:
                    is_valid = item_val == fact_val
                if not is_valid:
                    reason = (
                        self.description
                        or f"{field_name} '{item_val}' does not match '{fact_val}'"
                    )
                    return False, reason
            elif op == "!=":
                if isinstance(item_val, str) and isinstance(fact_val, str):
                    is_valid = item_val.strip().lower() != fact_val.strip().lower()
                else:
                    is_valid = item_val != fact_val
                if not is_valid:
                    reason = (
                        self.description
                        or f"{field_name} '{item_val}' must not equal '{fact_val}'"
                    )
                    return False, reason
            elif op == "in":
                if isinstance(fact_val, (list, tuple, set)):
                    if isinstance(item_val, str):
                        is_valid = any(
                            item_val.strip().lower() == str(v).strip().lower()
                            for v in fact_val
                        )
                    else:
                        is_valid = item_val in fact_val
                elif isinstance(fact_val, str) and isinstance(item_val, str):
                    is_valid = item_val.strip().lower() in fact_val.strip().lower()
                else:
                    is_valid = item_val in fact_val
                if not is_valid:
                    reason = (
                        self.description
                        or f"{field_name} '{item_val}' is not in allowed {fact_val}"
                    )
                    return False, reason
            elif op == "contains":
                if isinstance(item_val, (list, tuple, set)):
                    if isinstance(fact_val, str):
                        is_valid = any(
                            str(v).strip().lower() == fact_val.strip().lower()
                            for v in item_val
                        )
                    else:
                        is_valid = fact_val in item_val
                elif isinstance(item_val, str) and isinstance(fact_val, str):
                    is_valid = fact_val.strip().lower() in item_val.strip().lower()
                else:
                    is_valid = fact_val in item_val
                if not is_valid:
                    reason = (
                        self.description
                        or f"{field_name} does not contain '{fact_val}'"
                    )
                    return False, reason
        except (TypeError, ValueError):
            return True, None

        return True, None


@dataclass
class WorkItem:
    item_id: str
    goal_id: str
    kind: str
    title: str
    data: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    predicates: list[ConstraintPredicate] = field(default_factory=list)
    source_doc_id: str | None = None
    status: WorkStatus = WorkStatus.VALID
    stale_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "goal_id": self.goal_id,
            "kind": self.kind,
            "title": self.title,
            "data": dict(self.data),
            "depends_on": list(self.depends_on),
            "predicates": [
                p.to_dict() if hasattr(p, "to_dict") else vars(p)
                for p in self.predicates
            ],
            "source_doc_id": self.source_doc_id,
            "status": self.status.value if isinstance(self.status, WorkStatus) else str(self.status),
            "stale_reason": self.stale_reason,
        }


def validate_work_item(item: WorkItem, facts: dict[str, Any]) -> tuple[WorkStatus, str | None]:
    """Validate a WorkItem against active session facts.

    Rules:
    - If item has predicates:
        Evaluates each ConstraintPredicate. The first failing predicate marks
        the item STALE with its reason.
    - If item has no predicates (legacy slot validation):
        - If item depends on "budget" and its price exceeds facts["budget"]:
            STALE + clear reason
        - If item depends on "people" and its stored people count differs from facts["people"]:
            STALE + clear reason
        - If item depends on "destination" and its stored destination differs from facts["destination"]:
            STALE + clear reason
    - Otherwise:
        VALID, None

    Returns:
        (WorkStatus, reason | None)
    """
    # A. Generic predicate validation (preferred)
    if item.predicates:
        for pred in item.predicates:
            is_valid, reason = pred.evaluate(item.data, facts)
            if not is_valid:
                item.status = WorkStatus.STALE
                item.stale_reason = reason
                return WorkStatus.STALE, reason

        item.status = WorkStatus.VALID
        item.stale_reason = None
        return WorkStatus.VALID, None

    # B. Legacy slot validation (backward compatibility for items without predicates)
    # 1. Budget check
    if "budget" in item.depends_on and "budget" in facts and facts["budget"] is not None:
        price = item.data.get("price")
        if price is None:
            price = item.data.get("cost")
        if price is None:
            price = item.data.get("quote_inr")
        if price is None:
            price = item.data.get("amount")

        if price is not None and price > facts["budget"]:
            reason = f"Price {price} exceeds budget of {facts['budget']}"
            item.status = WorkStatus.STALE
            item.stale_reason = reason
            return WorkStatus.STALE, reason

    # 2. People check
    if "people" in item.depends_on and "people" in facts and facts["people"] is not None:
        people = item.data.get("people")
        if people is None:
            people = item.data.get("party_size")
        if people is None:
            people = item.data.get("capacity")
        if people is None:
            people = item.data.get("count")

        if people is not None and people != facts["people"]:
            reason = f"Party size {people} does not match current facts ({facts['people']})"
            item.status = WorkStatus.STALE
            item.stale_reason = reason
            return WorkStatus.STALE, reason

    # 3. Destination check
    if "destination" in item.depends_on and "destination" in facts and facts["destination"] is not None:
        dest = item.data.get("destination")
        if dest is None:
            dest = item.data.get("city")
        if dest is None:
            dest = item.data.get("location")

        if dest is not None and str(dest).strip().lower() != str(facts["destination"]).strip().lower():
            reason = f"Destination '{dest}' does not match current destination '{facts['destination']}'"
            item.status = WorkStatus.STALE
            item.stale_reason = reason
            return WorkStatus.STALE, reason

    # Otherwise: VALID, None
    item.status = WorkStatus.VALID
    item.stale_reason = None
    return WorkStatus.VALID, None


_INVALID_CITIES = {
    "rates", "from", "price", "cost", "terminal", "airport", "hours",
    "night", "hotel", "hotels", "domestic", "standard", "flexible",
    "corporate", "advance",
}

_HOTEL_PATTERNS = [
    # 1. Corpus style: "The Harbour House in Mumbai sits ... rates from 8,900 INR"
    re.compile(
        r"\b([A-Z][A-Za-z0-9\s'&]+?)\s+in\s+([A-Z][a-zA-Z]+)\b[^.\n]*?rates\s+from\s+(?:INR|Rs\.?|₹)?\s*([0-9,]+)\s*(?:INR|rupees)?"
    ),
    # 2. CSV style: "Harbour House, Mumbai, 8900 INR"
    re.compile(
        r"\b([A-Z][A-Za-z0-9\s'&]+?),\s*([A-Z][a-zA-Z]+),\s*(?:INR|Rs\.?|₹)?\s*([0-9,]+)\s*(?:INR|rupees)?"
    ),
    # 3. Comma-separated with "rates from": "Harbour House, Mumbai, rates from 8,900 INR"
    re.compile(
        r"\b([A-Z][A-Za-z0-9\s'&]+?),\s*([A-Z][a-zA-Z]+),\s*rates\s+from\s+(?:INR|Rs\.?|₹)?\s*([0-9,]+)\s*(?:INR|rupees)?"
    ),
]


def _extract_hotel_tuples(text: str) -> list[tuple[str, str, int]]:
    results: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()

    for pattern in _HOTEL_PATTERNS:
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            city = m.group(2).strip()
            price_str = m.group(3).strip()

            if city.lower() in _INVALID_CITIES or name.lower() in _INVALID_CITIES:
                continue
            if len(city) < 3 or len(name) < 3:
                continue

            try:
                price = int(price_str.replace(",", ""))
            except ValueError:
                continue

            name_clean = re.sub(r"^the\s+", "", name, flags=re.IGNORECASE).strip()
            key = (name_clean.lower(), city.lower())
            if key not in seen:
                seen.add(key)
                results.append((name, city, price))

    return results


def extract_work_items_from_evidence(
    evidence: list[dict[str, Any]],
    goal_id: str,
    facts: dict[str, Any] | None = None,
) -> list[WorkItem]:
    """Inspect retrieved evidence dictionaries and extract WorkItems for supported hotel results.

    - Uses doc_id as source_doc_id
    - Extracts hotel name, destination/city, and price
    - Sets kind = 'hotel', depends_on = ['destination', 'budget']
    - Validates against current facts
    - Avoids duplicate WorkItems for the same hotel
    """
    if facts is None:
        facts = {}

    work_items: list[WorkItem] = []
    seen_ids: set[str] = set()

    for item in evidence:
        doc_id = item.get("doc_id") or item.get("id")
        content = item.get("snippet") or item.get("text") or item.get("body") or ""

        tuples = _extract_hotel_tuples(content)
        for name, city, price in tuples:
            name_clean = re.sub(r"^the\s+", "", name, flags=re.IGNORECASE).strip()
            name_slug = re.sub(r"[^a-z0-9]+", "-", name_clean.lower()).strip("-")
            city_slug = re.sub(r"[^a-z0-9]+", "-", city.lower()).strip("-")
            item_id = f"hotel-{name_slug}-{city_slug}"

            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            title = f"{name}, {city}" if city.lower() not in name.lower() else name
            work_item = WorkItem(
                item_id=item_id,
                goal_id=goal_id,
                kind="hotel",
                title=title,
                data={
                    "name": name,
                    "destination": city,
                    "city": city,
                    "price": price,
                },
                depends_on=["destination", "budget"],
                source_doc_id=doc_id,
                status=WorkStatus.VALID,
            )
            validate_work_item(work_item, facts)
            work_items.append(work_item)

    return work_items


@dataclass
class ReconciliationResult:
    valid: list[WorkItem] = field(default_factory=list)
    stale: list[WorkItem] = field(default_factory=list)
    became_stale: list[WorkItem] = field(default_factory=list)
    became_valid: list[WorkItem] = field(default_factory=list)

    def __iter__(self):
        # Allow tuple unpacking: valid, stale = reconcile_work_items(...)
        return iter((self.valid, self.stale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": [w.to_dict() for w in self.valid],
            "stale": [w.to_dict() for w in self.stale],
            "became_stale": [w.to_dict() for w in self.became_stale],
            "became_valid": [w.to_dict() for w in self.became_valid],
        }


def reconcile_work_items(
    items: list[WorkItem],
    facts: dict[str, Any],
) -> ReconciliationResult:
    """Reconcile existing WorkItems against current session facts.

    - Validates every existing WorkItem using validate_work_item()
    - Preserves VALID items
    - Marks invalid items STALE with a clear stale_reason
    - Preserves stale_reason
    - Never deletes WorkItems
    - Returns partitioned ReconciliationResult
    """
    valid: list[WorkItem] = []
    stale: list[WorkItem] = []
    became_stale: list[WorkItem] = []
    became_valid: list[WorkItem] = []

    for item in items:
        old_status = item.status
        validate_work_item(item, facts)

        if item.status == WorkStatus.VALID:
            valid.append(item)
            if old_status == WorkStatus.STALE:
                became_valid.append(item)
        else:
            stale.append(item)
            if old_status == WorkStatus.VALID:
                became_stale.append(item)

    return ReconciliationResult(
        valid=valid,
        stale=stale,
        became_stale=became_stale,
        became_valid=became_valid,
    )


def format_reconciliation_summary(result: ReconciliationResult) -> str:
    """Format a clear explanation of reconciliation for generation."""
    lines: list[str] = []
    if result.became_stale:
        for item in result.became_stale:
            lines.append(f"- {item.title}: STALE ({item.stale_reason})")
    elif result.stale and not result.valid:
        for item in result.stale:
            lines.append(f"- {item.title}: STALE ({item.stale_reason})")

    if result.became_valid:
        for item in result.became_valid:
            lines.append(f"- {item.title}: VALID")

    if lines:
        return "Reconciled previous options against current facts:\n" + "\n".join(lines)
    return ""

