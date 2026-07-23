"""Parse a free-text retention period into a machine-computable duration — block B1.

The CNIL register states retention in prose (``"5 years from the payment of the
salary"``, ``"24 mesi"``, ``"a criterio"``). Block B7 needs a number to answer
"past retention?" as a comparison rather than a prose read, so
:func:`parse_retention` extracts the first ``<number> <unit>`` it finds and
converts it to whole months; text without a numeric duration (a criterion,
``"N/A"``, empty) yields ``None``.

Both English and Italian units are recognized, since the target registers are
Italian/Swiss while the reference CNIL template is written in English.
"""

from __future__ import annotations

import re

#: Recognized unit names (English + Italian) mapped to their length in months.
_MONTHS_PER_UNIT: dict[str, int] = {
    "year": 12,
    "years": 12,
    "anno": 12,
    "anni": 12,
    "month": 1,
    "months": 1,
    "mese": 1,
    "mesi": 1,
}

_DURATION_RE = re.compile(
    r"(\d+)\s*(" + "|".join(_MONTHS_PER_UNIT) + r")\b",
    re.IGNORECASE,
)


def parse_retention(text: str) -> int | None:
    """Extract a retention duration in whole months from free text.

    Recognizes the first ``<number> <unit>`` occurrence, with English or Italian
    units (``year(s)``/``anno``/``anni``, ``month(s)``/``mese``/``mesi``). Days
    are intentionally not parsed: registers express retention in years or months
    and B7 reasons in months (see the open point in ``doc/plans/B1-ropa-ingestion.md``).

    :param text: the raw retention wording from the register, e.g.
        ``"5 years from the payment of the salary"``.
    :returns: the duration in months (e.g. ``60``), or ``None`` when the text
        states a criterion rather than a fixed duration, or is empty.
    """
    match = _DURATION_RE.search(text)
    if match is None:
        return None
    return int(match.group(1)) * _MONTHS_PER_UNIT[match.group(2).lower()]


__all__ = ["parse_retention"]
