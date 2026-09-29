"""Deterministic clean-up of tool arguments, applied to every benchmark tool call.

Two of the agent's instructions (``ARGUMENT_RULES`` 2 and 3) are about the form
of a value, not its meaning: pass a date the way it was said, without a year the
user never gave, and join a code that was spelled out character by character.
Models follow them unreliably - a smaller model in particular writes
"2026-08-20" for "August twentieth" and keeps "A-B-C-1-2-3" as spelled - so the
same two rules are also enforced here, by parameter kind, for every call.

Nothing here knows about benchmark items: it looks only at the parameter's name
(an identifier, a date) and at what the user actually said.
"""

from __future__ import annotations

import re
from typing import Any

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")

# identifier-like parameters: order_id, product_id, doc_number, confirmation_code...
_ID_PARAM = re.compile(r"(?:^|_)(?:id|number|code)$")
_DATE_PARAM = re.compile(r"(?:^|_)date$")
# single characters separated by dashes, dots, underscores or spaces: "A-B-C-1-2-3", "Q 7 X 2"
_SPELLED = re.compile(r"^[A-Za-z0-9](?:[\s.\-_]+[A-Za-z0-9])+$")
_SEPARATORS = re.compile(r"[\s.\-_]+")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_ORDINAL = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)


_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def _typed_literal(text: str) -> Any:
    """``"true"`` -> True, ``"42"`` -> 42, ``"2.5"`` -> 2.5; anything else unchanged."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if _NUMBER.match(text):
        return float(text) if "." in text else int(text)
    return text


def normalize_arguments(
    args: dict[str, Any], heard: str = "", untyped: frozenset[str] | set[str] = frozenset()
) -> dict[str, Any]:
    """Return ``args`` with spelled-out codes joined and dates kept as spoken.

    ``heard`` is what the user said in this conversation; a year is only removed
    from a date when the user never said it. ``untyped`` names parameters the tool
    declares without a type (``value: Any``): a model can only send those as
    text, so a literal boolean or number there is passed as one.
    """
    out = dict(args)
    for name, value in args.items():
        if not isinstance(value, str):
            continue
        text = value.strip()
        if name in untyped:
            out[name] = _typed_literal(text)
        elif _ID_PARAM.search(name) and _SPELLED.match(text):
            out[name] = _SEPARATORS.sub("", text)
        elif _DATE_PARAM.search(name):
            iso = _ISO_DATE.match(text)
            if iso and iso.group(1) not in heard:
                month, day = int(iso.group(2)), int(iso.group(3))
                if 1 <= month <= 12 and 1 <= day <= 31:
                    text = f"{MONTHS[month - 1]} {day}"
            out[name] = _ORDINAL.sub(r"\1", text)
    return out
