"""Tests for the CNIL-sheet normalizer — block B1.

The normalizer reads one full CNIL fiche and keeps only its identity plus the
"Categories of personal data" block; every other section of the form is ignored.
"""

from __future__ import annotations

import pytest

from pii_detection.ropa.ingestion.normalizer import normalize

# A grid shaped like the real CNIL activity sheet: identity, stakeholders,
# purposes, the categories block, then sections we do not model.
_FULL_SHEET = [
    ["Description of the processing operation"],
    ["Name of the processing operation", "Payroll management"],
    ["N° / REF", "RT-001"],
    [],
    ["Stakeholders", "Name", "Address"],
    ["Controller", "Louise DUPONT", "1 rue Rivoli"],
    [],
    ["Main purpose", "Payroll management"],
    ["Sub-purpose 1", "Calculation of remuneration"],
    [],
    ["Categories of personal data", "Description", "Data retention period"],
    ["Identity", "Last names and addresses", "5 years"],
    ["Economic information", "Bank account details", "5 years"],
    [],
    ["Categories of data subjects", "Description", "Details"],
    ["Category 1", "Employees"],
    [],
    ["Recipients", "Type of recipient", "Details"],
    ["Recipient 1", "Administrative Department"],
]


def test_reads_identity_and_only_the_categories_block() -> None:
    activity = normalize(_FULL_SHEET)

    assert activity.name == "Payroll management"
    assert activity.purpose == "Payroll management"

    # Exactly the two rows of the categories block — the "data subjects" and
    # "recipients" sections after the blank line are not mistaken for categories.
    assert [m.raw_text for m in activity.macro_categories] == ["Identity", "Economic information"]
    assert all(m.retention_months == 60 for m in activity.macro_categories)
    assert [m.categories[0].raw_text for m in activity.macro_categories] == [
        "Last names and addresses",
        "Bank account details",
    ]


def test_sheet_without_name_is_rejected() -> None:
    # The blank template fiche has the labels but no value — not an activity.
    with pytest.raises(ValueError):
        normalize([["Name of the processing operation"], ["Categories of personal data"]])
