"""Tests for the compliance block (B6 association + B7 check).

The verdict logic (:func:`build_report`) is exercised **pure**, on synthetic
domain objects with no database and an injected clock, so the buckets, the N:N
union semantics, the approximate retention and the per-instance coverage are all
deterministic. :func:`check_document` is then exercised end-to-end over two
in-memory SQLite databases, verifying the coverage is persisted.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pii_detection.compliance.checker import build_report, check_document
from pii_detection.extraction.dates import reference_date
from pii_detection.registry.freshness import stamp_for
from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIIMatch,
    TextSpan,
)
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import Document, PIIInstance
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import (
    DeclaredCategory,
    DeclaredMacroCategory,
    MappingState,
    ProcessingActivity,
)

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


# --- synthetic-object builders (no session) ---------------------------------


def _category(
    pii_types: Sequence[str], *, state: MappingState = MappingState.CONFIRMED
) -> DeclaredCategory:
    return DeclaredCategory(raw_text=",".join(pii_types) or "empty", pii_types=list(pii_types), mapping_state=state)


def _macro(
    categories: Sequence[DeclaredCategory], *, retention_months: int | None = None
) -> DeclaredMacroCategory:
    macro = DeclaredMacroCategory(
        raw_text="macro", retention_text="", retention_months=retention_months
    )
    macro.categories = list(categories)
    return macro


def _activity(activity_id: str, macros: Sequence[DeclaredMacroCategory]) -> ProcessingActivity:
    activity = ProcessingActivity(id=activity_id, name=activity_id, purpose="p")
    activity.macro_categories = list(macros)
    return activity


def _instance(instance_id: int, pii_type: str) -> PIIInstance:
    return PIIInstance(
        id=instance_id,
        document_id="doc",
        pii_type=pii_type,
        start=0,
        end=1,
        confidence=0.9,
        confirmation_level="single_source",
    )


def _document(activity_ids: Sequence[str], *, reference_date: datetime | None = None) -> Document:
    return Document(
        document_id="doc", activity_ids=list(activity_ids), reference_date=reference_date
    )


# --- pure verdict logic ------------------------------------------------------


def test_buckets_orphan_covered_missing() -> None:
    activity = _activity("paghe", [_macro([_category(["person_name", "iban", "email"])])])
    document = _document(["paghe"])
    instances = [_instance(1, "person_name"), _instance(2, "iban"), _instance(3, "health_data")]

    result = build_report(document, [activity], [], instances, now=_NOW)

    assert result.report.orphan == ("health_data",)
    assert result.report.covered == ("iban", "person_name")
    assert result.report.missing == ("email",)
    assert result.coverage == {1: "paghe", 2: "paghe", 3: None}


def test_union_semantics_under_multiple_activities() -> None:
    a = _activity("A", [_macro([_category(["person_name"])])])
    b = _activity("B", [_macro([_category(["iban"])])])
    document = _document(["A", "B"])
    instances = [_instance(1, "person_name"), _instance(2, "iban"), _instance(3, "health_data")]

    result = build_report(document, [a, b], [], instances, now=_NOW)

    # Orphan only when declared by NONE of the associated activities.
    assert result.report.orphan == ("health_data",)
    assert result.report.covered == ("iban", "person_name")
    # Coverage points at the (first) activity that declares the type.
    assert result.coverage == {1: "A", 2: "B", 3: None}


def test_proposed_ignored_by_default_counted_on_request() -> None:
    activity = _activity("A", [_macro([_category(["iban"], state=MappingState.PROPOSED)])])
    document = _document(["A"])
    instances = [_instance(1, "iban")]

    default = build_report(document, [activity], [], instances, now=_NOW)
    assert default.report.orphan == ("iban",)  # proposed mapping does not count

    included = build_report(document, [activity], [], instances, include_proposed=True, now=_NOW)
    assert included.report.covered == ("iban",)
    assert included.report.orphan == ()


def test_unresolved_category_is_flagged() -> None:
    activity = _activity("A", [_macro([_category([])])])  # declared but no pii_type
    document = _document(["A"])

    result = build_report(document, [activity], [], [], now=_NOW)

    assert result.report.unresolved == ("empty",)


def test_unknown_activity_id_is_reported() -> None:
    document = _document(["ghost"])
    result = build_report(document, [], ["ghost"], [], now=_NOW)
    assert result.report.unknown_activity_ids == ("ghost",)


def test_retention_breach_when_document_too_old() -> None:
    activity = _activity("A", [_macro([_category(["iban"])], retention_months=12)])
    old = _NOW - timedelta(days=400)  # ~13 months
    document = _document(["A"], reference_date=old)
    instances = [_instance(1, "iban")]

    result = build_report(document, [activity], [], instances, now=_NOW)

    (flag,) = result.report.retention_flags
    assert flag.activity_id == "A"
    assert flag.retention_months == 12
    assert flag.age_months > 12
    assert flag.pii_types == ("iban",)
    assert result.report.compliant is False


def test_overdue_months_measures_severity() -> None:
    activity = _activity("A", [_macro([_category(["iban"])], retention_months=12)])
    document = _document(["A"], reference_date=_NOW - timedelta(days=365 * 5))
    result = build_report(document, [activity], [], [_instance(1, "iban")], now=_NOW)

    (flag,) = result.report.retention_flags
    assert flag.overdue_months == flag.age_months - 12
    assert flag.overdue_months > 45  # ~5 years kept against a 1-year limit


def test_criterion_retention_is_reported_as_unverifiable() -> None:
    # The register states a criterion ("for the duration of the relationship"), so
    # retention_months is None and no comparison is possible. Before, the check
    # simply skipped it and the document came out clean: the case nobody verified
    # looked exactly like the case that passed.
    activity = _activity("A", [_macro([_category(["iban"])], retention_months=None)])
    document = _document(["A"], reference_date=_NOW - timedelta(days=4000))
    result = build_report(document, [activity], [], [_instance(1, "iban")], now=_NOW)

    assert result.report.retention_flags == ()
    assert result.report.retention_unresolved == ("macro",)
    # Informational, like `unresolved`: it does not condemn the document by itself.
    assert result.report.compliant is True


def test_criterion_retention_is_silent_when_its_data_is_absent() -> None:
    # Nothing of that category is in the document: there is nothing to keep too
    # long, so flagging it would be noise.
    activity = _activity("A", [_macro([_category(["iban"])], retention_months=None)])
    document = _document(["A"], reference_date=_NOW - timedelta(days=4000))
    result = build_report(document, [activity], [], [], now=_NOW)

    assert result.report.retention_unresolved == ()


def test_unverifiable_retention_does_not_need_a_document_date() -> None:
    # A missing reference date disables the age comparison, not the report of what
    # cannot be verified: that gap is in the register, not in the document.
    activity = _activity("A", [_macro([_category(["iban"])], retention_months=None)])
    document = _document(["A"], reference_date=None)
    result = build_report(document, [activity], [], [_instance(1, "iban")], now=_NOW)

    assert result.report.retention_unresolved == ("macro",)


def test_proposed_mapping_does_not_trigger_a_breach_when_excluded() -> None:
    # The coverage comparison ignores PROPOSED mappings by default; the retention
    # check used to count them anyway, so the same category was "not declared" for
    # one half of the verdict and "declared, and overdue" for the other.
    activity = _activity(
        "A",
        [_macro([_category(["iban"], state=MappingState.PROPOSED)], retention_months=12)],
    )
    document = _document(["A"], reference_date=_NOW - timedelta(days=400))
    instances = [_instance(1, "iban")]

    strict = build_report(document, [activity], [], instances, now=_NOW)
    assert strict.report.retention_flags == ()
    assert strict.report.orphan == ("iban",)  # not declared, as far as this verdict goes

    lenient = build_report(
        document, [activity], [], instances, include_proposed=True, now=_NOW
    )
    assert len(lenient.report.retention_flags) == 1
    assert lenient.report.covered == ("iban",)


def test_no_retention_breach_when_recent() -> None:
    activity = _activity("A", [_macro([_category(["iban"])], retention_months=12)])
    recent = _NOW - timedelta(days=90)  # ~3 months
    document = _document(["A"], reference_date=recent)
    instances = [_instance(1, "iban")]

    result = build_report(document, [activity], [], instances, now=_NOW)

    assert result.report.retention_flags == ()
    assert result.report.compliant is True


# --- end-to-end over two in-memory databases ---------------------------------


def _match(start: int, end: int, pii_type: str) -> PIIMatch:
    provenance = DetectionProvenance("det.x", DetectorKind.REGEX, pii_type, 0.9)
    return PIIMatch(
        span=TextSpan(start, end),
        text="?",
        pii_type=pii_type,
        confidence=0.9,
        confirmation_level=ConfirmationLevel.SINGLE_SOURCE,
        sources=[provenance],
        document_id="ignored",
    )


def test_check_document_persists_coverage(tmp_path: Path) -> None:
    ropa = ROPARepository(url=f"sqlite:///{tmp_path}/ropa.db")
    registry = PIIRepository(url=f"sqlite:///{tmp_path}/pii.db")

    ropa.save([_activity("paghe", [_macro([_category(["iban"])])])])
    registry.record_scan("doc", [_match(0, 10, "iban"), _match(20, 30, "health_data")])
    registry.assign_activities("doc", ["paghe"])

    report = check_document("doc", ropa=ropa, registry=registry)

    assert report.covered == ("iban",)
    assert report.orphan == ("health_data",)
    coverage = {i.pii_type: i.processing_activity_id for i in registry.instances_for("doc")}
    assert coverage == {"iban": "paghe", "health_data": None}


def test_check_document_reports_retention_end_to_end(tmp_path: Path) -> None:
    # The whole chain, not just the pure function: a real file dated years back,
    # ingested through the registry, checked against a register that declares a
    # one-year retention. `now` is injected, so the outcome does not depend on when
    # the suite runs.
    ropa = ROPARepository(url=f"sqlite:///{tmp_path}/ropa.db")
    registry = PIIRepository(url=f"sqlite:///{tmp_path}/pii.db")
    ropa.save([_activity("paghe", [_macro([_category(["iban"])], retention_months=12)])])

    source = tmp_path / "vecchio.txt"
    source.write_text("IBAN in archivio.", encoding="utf-8")
    old = _NOW - timedelta(days=365 * 4)
    os.utime(source, (old.timestamp(), old.timestamp()))
    registry.record_scan(
        "Archivio/vecchio.txt",
        [_match(0, 10, "iban")],
        path=str(source),
        reference_date=reference_date(source),
        stamp=stamp_for(source),
    )
    registry.assign_activities("Archivio/vecchio.txt", ["paghe"])

    report = check_document(
        "Archivio/vecchio.txt", ropa=ropa, registry=registry, now=_NOW
    )

    (flag,) = report.retention_flags
    assert flag.retention_months == 12
    assert flag.overdue_months > 30
    assert report.compliant is False
