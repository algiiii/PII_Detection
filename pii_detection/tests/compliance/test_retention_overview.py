"""Tests for the corpus-wide retention view (B7 → B8).

The view adds no rule: it reuses the per-document verdict. What is tested here is
therefore everything *around* that reuse — which documents earn a row, in which
order, and the invariant that looking at the registry must not modify it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pii_detection.compliance.overview import (
    format_overview,
    retention_overview,
)
from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIIMatch,
    TextSpan,
)
from pii_detection.extraction.dates import reference_date
from pii_detection.registry.freshness import stamp_for
from pii_detection.registry.repository import PIIRepository
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import (
    DeclaredCategory,
    DeclaredMacroCategory,
    MappingState,
    ProcessingActivity,
)

_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _activity(
    activity_id: str, pii_types: Sequence[str], *, retention_months: int | None
) -> ProcessingActivity:
    macro = DeclaredMacroCategory(
        raw_text=f"categorie {activity_id}",
        retention_text="",
        retention_months=retention_months,
    )
    macro.categories = [
        DeclaredCategory(
            raw_text=",".join(pii_types),
            pii_types=list(pii_types),
            mapping_state=MappingState.CONFIRMED,
        )
    ]
    activity = ProcessingActivity(id=activity_id, name=activity_id, purpose="p")
    activity.macro_categories = [macro]
    return activity


def _match(pii_type: str) -> PIIMatch:
    provenance = DetectionProvenance("det.x", DetectorKind.REGEX, pii_type, 0.9)
    return PIIMatch(
        span=TextSpan(0, 10),
        text="?",
        pii_type=pii_type,
        confidence=0.9,
        confirmation_level=ConfirmationLevel.SINGLE_SOURCE,
        sources=[provenance],
        document_id="ignored",
    )


def _record(
    registry: PIIRepository,
    tmp_path: Path,
    document_id: str,
    pii_type: str,
    *,
    age_days: int,
    activities: Sequence[str] = (),
) -> None:
    """Ingest one document of a given age, optionally associated with activities."""
    source = tmp_path / document_id.replace("/", "_")
    source.write_text("x", encoding="utf-8")
    when = _NOW - timedelta(days=age_days)
    os.utime(source, (when.timestamp(), when.timestamp()))
    registry.record_scan(
        document_id,
        [_match(pii_type)],
        path=str(source),
        reference_date=reference_date(source, now=_NOW),
        stamp=stamp_for(source),
    )
    if activities:
        registry.assign_activities(document_id, list(activities))


def _repos(tmp_path: Path) -> tuple[ROPARepository, PIIRepository]:
    return (
        ROPARepository(url=f"sqlite:///{tmp_path}/ropa.db"),
        PIIRepository(url=f"sqlite:///{tmp_path}/pii.db"),
    )


def test_orders_by_severity_and_skips_the_compliant(tmp_path: Path) -> None:
    ropa, registry = _repos(tmp_path)
    ropa.save([_activity("paghe", ["iban"], retention_months=12)])
    _record(registry, tmp_path, "recente.txt", "iban", age_days=30, activities=["paghe"])
    _record(registry, tmp_path, "vecchio.txt", "iban", age_days=800, activities=["paghe"])
    _record(registry, tmp_path, "antico.txt", "iban", age_days=3000, activities=["paghe"])

    rows = retention_overview(ropa=ropa, registry=registry, now=_NOW)

    assert [row.document_id for row in rows] == ["antico.txt", "vecchio.txt"]
    assert rows[0].worst_overdue_months > rows[1].worst_overdue_months


def test_unassociated_documents_are_skipped(tmp_path: Path) -> None:
    # No activity means no declared retention to compare against: reporting these
    # would bury the real breaches under every unassociated file of the share.
    ropa, registry = _repos(tmp_path)
    ropa.save([_activity("paghe", ["iban"], retention_months=12)])
    _record(registry, tmp_path, "orfano.txt", "iban", age_days=3000)

    assert retention_overview(ropa=ropa, registry=registry, now=_NOW) == []


def test_unverifiable_retention_earns_a_row_without_severity(tmp_path: Path) -> None:
    ropa, registry = _repos(tmp_path)
    ropa.save([_activity("hr", ["iban"], retention_months=None)])
    _record(registry, tmp_path, "criterio.txt", "iban", age_days=3000, activities=["hr"])

    (row,) = retention_overview(ropa=ropa, registry=registry, now=_NOW)
    assert row.flags == ()
    assert row.unresolved == ("categorie hr",)
    assert row.worst_overdue_months == 0


def test_row_carries_the_date_and_its_provenance(tmp_path: Path) -> None:
    ropa, registry = _repos(tmp_path)
    ropa.save([_activity("paghe", ["iban"], retention_months=12)])
    _record(registry, tmp_path, "vecchio.txt", "iban", age_days=800, activities=["paghe"])

    (row,) = retention_overview(ropa=ropa, registry=registry, now=_NOW)
    assert row.reference_date_source == "file_mtime"  # a .txt has no internal date
    assert row.reference_date is not None
    assert row.activity_ids == ("paghe",)


def test_the_view_does_not_write_to_the_registry(tmp_path: Path) -> None:
    # Reading a page must not have side effects: unlike check_document's default,
    # the overview never persists per-instance coverage.
    ropa, registry = _repos(tmp_path)
    ropa.save([_activity("paghe", ["iban"], retention_months=12)])
    _record(registry, tmp_path, "vecchio.txt", "iban", age_days=800, activities=["paghe"])

    retention_overview(ropa=ropa, registry=registry, now=_NOW)

    (instance,) = registry.instances_for("vecchio.txt")
    assert instance.processing_activity_id is None


def test_unknown_activity_does_not_break_the_view(tmp_path: Path) -> None:
    # The association is a plain string id across two databases with no physical
    # foreign key: an id the ROPA no longer knows must be tolerated.
    ropa, registry = _repos(tmp_path)
    ropa.save([_activity("paghe", ["iban"], retention_months=12)])
    _record(
        registry,
        tmp_path,
        "misto.txt",
        "iban",
        age_days=800,
        activities=["paghe", "sparita"],
    )

    (row,) = retention_overview(ropa=ropa, registry=registry, now=_NOW)
    assert row.flags[0].activity_id == "paghe"


def test_format_overview_reports_nothing_to_do(tmp_path: Path) -> None:
    ropa, registry = _repos(tmp_path)
    assert "No retention issue" in format_overview(
        retention_overview(ropa=ropa, registry=registry, now=_NOW)
    )
