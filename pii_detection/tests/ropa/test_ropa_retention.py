"""Tests for the retention parser (free-text period → whole months)."""

from __future__ import annotations

import pytest

from pii_detection.ropa.ingestion.retention import parse_retention


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5 years from the payment of the salary", 60),  # English, trailing prose
        ("2 anni", 24),  # Italian years
        ("6 months", 6),  # English months
        ("24 mesi", 24),  # Italian months
        ("1 year", 12),  # singular unit
        ("5 ans à compter du versement du salaire", 60),  # French years, CNIL wording
        ("1 an", 12),  # French singular
        ("6 mois", 6),  # French months
        ("2 anni e 6 mesi", 24),  # first duration wins, as documented
        ("a criterio", None),  # criterion, no fixed duration
        ("N/A", None),
        ("", None),  # empty cell
    ],
)
def test_parse_retention(text: str, expected: int | None) -> None:
    assert parse_retention(text) == expected
