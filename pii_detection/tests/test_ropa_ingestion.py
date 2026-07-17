"""Simple tests for the ROPA ingestion (reader + normalizer)."""

from __future__ import annotations

from pathlib import Path

from pii_detection.ropa.ingestion.excel_reader import read_records
from pii_detection.ropa.ingestion.normalizer import normalize, split_multi

XLSX = Path(__file__).with_name("ROPA.xlsx")


def test_split_multi() -> None:
    assert split_multi(None) == []
    assert split_multi("a; b ;;c") == ["a", "b", "c"]


def test_read_records() -> None:
    table = read_records(XLSX)
    assert "name" in table.columns
    assert len(table.records) == 6
    assert table.records[0]["name"] == "Gestione del personale"


def test_normalize_activity_count_and_ids() -> None:
    ropa = normalize(read_records(XLSX))
    assert len(ropa.activities) == 6
    ids = [a.activity_id for a in ropa.activities]
    assert len(set(ids)) == 6  # generated ids are unique


def test_normalize_categories_and_retention() -> None:
    ropa = normalize(read_records(XLSX))
    health = next(a for a in ropa.activities if "sanitaria" in a.name)
    assert "health_data" in health.declared_pii_types()
    assert health.recipients == ["Medico competente"]

    marketing = next(a for a in ropa.activities if "Marketing" in a.name)
    assert marketing.retentions[0].duration_months is None
