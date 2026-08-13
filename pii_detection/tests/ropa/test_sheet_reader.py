"""Tests for the spreadsheet reader — block B1.

Focus on the ODS quirk the CNIL template exposes: cell comments (the "red
triangle" help notes) live in an ``<office:annotation>`` inside the cell, and
must not leak into the cell value.
"""

from __future__ import annotations

from pathlib import Path

from pii_detection.ropa.ingestion.sheet_reader import read_sheet

_FIXTURE = Path(__file__).parent / "record-processing-activities.ods"


def test_cell_annotations_are_excluded_from_values() -> None:
    rows = read_sheet(_FIXTURE, "4_-_Example_")

    # The Social Security Number cell carries a "Cf. Article 87..." comment; the
    # value must be the label alone, not the comment glued in front of it.
    nir = next(row for row in rows if row and row[0].startswith("Social Security"))
    assert nir[0] == "Social Security Number (or NIR)"

    # No cell anywhere on the sheet may contain leaked comment prose.
    assert not any("Article 87" in cell for row in rows for cell in row)
