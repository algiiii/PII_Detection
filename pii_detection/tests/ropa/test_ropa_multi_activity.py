"""Tests for multi-activity ingestion over a whole CNIL workbook."""

from __future__ import annotations

from pathlib import Path

from pii_detection.ropa.ingestion.pipeline import ingest_file
from pii_detection.ropa.ingestion.sheet_reader import sheet_names
from pii_detection.ropa.repository import ROPARepository

_FIXTURE = Path(__file__).parent / "record-processing-activities.ods"


def test_sheet_names_lists_every_tab() -> None:
    names = sheet_names(_FIXTURE)
    assert "4_-_Example_" in names  # the real activity sheet
    assert "3_-_Template_" in names  # the blank template (must be skipped on ingest)


def test_ingest_keeps_only_real_activities(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'ropa.db'}"

    activities = ingest_file(_FIXTURE, db_url)
    assert len(activities) == 1  # tutorial/lists (no section) and template (no name) skipped

    saved = ROPARepository(db_url).load()
    assert [a.id for a in saved] == ["payroll-management"]
    # retention was parsed from prose into whole months (block B7 needs a number)
    assert saved[0].macro_categories[0].retention_months == 60
