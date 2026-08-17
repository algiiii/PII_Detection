"""Tests for the detected-PII registry (block B5, Step 1).

The repository is exercised with synthetic :class:`PIIMatch` and the ``ingest``
wiring with fake detectors on a plain-text file, so no Presidio is needed. The
key invariant checked here is **minimization**: no PII value ever reaches a column.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    PIIMatch,
    TextSpan,
)
from pii_detection.extraction.dates import DateSource
from pii_detection.registry.ingest import ingest_document
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import ChangeType, PIIInstance


def _repo(tmp_path: Path) -> PIIRepository:
    return PIIRepository(url=f"sqlite:///{tmp_path}/pii.db")


def _match(
    start: int,
    end: int,
    pii_type: str,
    *,
    value: str = "?",
    level: ConfirmationLevel = ConfirmationLevel.SINGLE_SOURCE,
) -> PIIMatch:
    provenance = DetectionProvenance("det.x", DetectorKind.REGEX, pii_type, 0.9)
    return PIIMatch(
        span=TextSpan(start, end),
        text=value,
        pii_type=pii_type,
        confidence=0.9,
        confirmation_level=level,
        sources=[provenance],
        document_id="ignored",
    )


class _SubstringDetector:
    """Emits a candidate for each given ``(pii_type, value)`` found in the text."""

    detector_id = "fake.substr"
    detector_kind = DetectorKind.REGEX

    def __init__(self, targets: list[tuple[str, str]]) -> None:
        self._targets = targets

    def detect(self, text: str) -> list[PIICandidate]:
        found: list[PIICandidate] = []
        for pii_type, value in self._targets:
            index = text.find(value)
            if index >= 0:
                found.append(
                    PIICandidate(
                        TextSpan(index, index + len(value)),
                        value,
                        DetectionProvenance("fake.substr", DetectorKind.REGEX, pii_type, 0.9),
                    )
                )
        return found


class _EmptyDetector:
    detector_id = "fake.empty"
    detector_kind = DetectorKind.NER

    def detect(self, text: str) -> list[PIICandidate]:
        return []


def test_first_scan_populates_instances_and_new_changes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    scan = repo.record_scan("doc1", [_match(0, 10, "iban"), _match(20, 30, "email")])

    instances = repo.instances_for("doc1")
    assert {i.pii_type for i in instances} == {"iban", "email"}
    for instance in instances:
        assert instance.last_scan_id == scan.id
        assert instance.sources == ["det.x"]
        assert len(instance.changes) == 1
        assert instance.changes[0].change_type is ChangeType.NEW
        assert instance.changes[0].previous_scan_id is None


def test_value_is_never_persisted(tmp_path: Path) -> None:
    secret = "IT60X0542811101000000123456"
    repo = _repo(tmp_path)
    repo.record_scan("doc", [_match(0, len(secret), "iban", value=secret)])

    (instance,) = repo.instances_for("doc")
    # No column holds the PII value, and the model has no value/text field.
    assert "text" not in PIIInstance.model_fields
    assert "value" not in PIIInstance.model_fields
    stored = {str(getattr(instance, name)) for name in PIIInstance.model_fields}
    assert secret not in stored


def test_replace_avoids_duplicates(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    matches = [_match(0, 10, "iban"), _match(20, 30, "email")]
    repo.record_scan("doc", matches)
    repo.record_scan("doc", matches, replace=True)
    assert len(repo.instances_for("doc")) == len(matches)


def _changes(instance: PIIInstance) -> list[ChangeType]:
    return [c.change_type for c in sorted(instance.changes, key=lambda c: c.id or 0)]


def test_rescan_identical_is_confirmed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    matches = [_match(0, 10, "iban"), _match(20, 30, "email")]
    repo.record_scan("doc", matches)
    repo.record_scan("doc", matches)

    instances = repo.instances_for("doc")
    assert len(instances) == 2  # no duplicates: the re-scan confirms, not re-adds
    assert all(_changes(i) == [ChangeType.NEW, ChangeType.CONFIRMED] for i in instances)


def test_rescan_added_pii_is_new(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.record_scan("doc", [_match(0, 10, "iban")])
    repo.record_scan("doc", [_match(0, 10, "iban"), _match(40, 50, "phone")])

    by_type = {i.pii_type: i for i in repo.instances_for("doc")}
    assert set(by_type) == {"iban", "phone"}
    assert _changes(by_type["iban"]) == [ChangeType.NEW, ChangeType.CONFIRMED]
    assert _changes(by_type["phone"]) == [ChangeType.NEW]


def test_rescan_missing_pii_is_removed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.record_scan("doc", [_match(0, 10, "iban"), _match(20, 30, "email")])
    repo.record_scan("doc", [_match(0, 10, "iban")])

    assert {i.pii_type for i in repo.instances_for("doc")} == {"iban"}  # current state
    all_by_type = {i.pii_type: i for i in repo.instances_for("doc", include_removed=True)}
    assert all_by_type["email"].removed is True
    assert _changes(all_by_type["email"]) == [ChangeType.NEW, ChangeType.REMOVED]


def test_rescan_shifted_pii_is_moved(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.record_scan("doc", [_match(0, 10, "iban")])
    repo.record_scan("doc", [_match(5, 15, "iban")])

    (instance,) = repo.instances_for("doc")
    assert (instance.start, instance.end) == (5, 15)  # position updated in place
    assert _changes(instance) == [ChangeType.NEW, ChangeType.MOVED]


def test_rescan_confirming_refreshes_certainty(tmp_path: Path) -> None:
    """A CONFIRMED re-scan (same span) refreshes confidence/level/sources, so a new
    AI agreement on an existing instance does not stay invisible."""
    repo = _repo(tmp_path)
    single = PIIMatch(
        span=TextSpan(0, 10),
        text="?",
        pii_type="iban",
        confidence=0.6,
        confirmation_level=ConfirmationLevel.SINGLE_SOURCE,
        sources=[DetectionProvenance("regex.x", DetectorKind.REGEX, "iban", 0.6)],
        document_id="ignored",
    )
    repo.record_scan("doc", [single])
    doubled = PIIMatch(
        span=TextSpan(0, 10),
        text="?",
        pii_type="iban",
        confidence=0.9,
        confirmation_level=ConfirmationLevel.DOUBLE_CONFIRMED,
        sources=[
            DetectionProvenance("regex.x", DetectorKind.REGEX, "iban", 0.6),
            DetectionProvenance("ai.m", DetectorKind.AI, "iban", 0.6),
        ],
        document_id="ignored",
    )
    repo.record_scan("doc", [doubled])

    (instance,) = repo.instances_for("doc")
    assert instance.confirmation_level == ConfirmationLevel.DOUBLE_CONFIRMED.value
    assert set(instance.sources) == {"regex.x", "ai.m"}
    assert instance.confidence == pytest.approx(0.9)
    assert _changes(instance) == [ChangeType.NEW, ChangeType.CONFIRMED]  # identity unchanged


def test_ingest_document_persists_detected_pii(tmp_path: Path) -> None:
    document = tmp_path / "note.txt"
    document.write_text(
        "Scrivi a mario@x.it, IBAN IT60X0542811101000000123456.", encoding="utf-8"
    )
    pattern = _SubstringDetector(
        [("email", "mario@x.it"), ("iban", "IT60X0542811101000000123456")]
    )
    repo = _repo(tmp_path)

    ingest_document(document, pattern, _EmptyDetector(), repository=repo)

    assert {i.pii_type for i in repo.instances_for("note")} == {"email", "iban"}


def test_ingest_records_reference_date_and_file_stamp(tmp_path: Path) -> None:
    # The two dates answer different questions and are stored separately: the
    # reference date is the document's age (B7), the stamp is what tells a later
    # scan whether the file changed.
    document = tmp_path / "note.txt"
    document.write_text("Nessuna PII qui.", encoding="utf-8")
    when = datetime(2019, 4, 5, 10, 0, tzinfo=timezone.utc)
    os.utime(document, (when.timestamp(), when.timestamp()))
    repo = _repo(tmp_path)

    ingest_document(document, _EmptyDetector(), _EmptyDetector(), repository=repo)

    recorded = repo.get_document("note")
    assert recorded is not None
    # A .txt carries no internal date, so both come from the file system.
    assert recorded.reference_date_source == DateSource.FILE_MTIME.value
    assert recorded.reference_date is not None
    assert recorded.reference_date.replace(tzinfo=timezone.utc) == when
    assert recorded.source_mtime is not None
    assert recorded.source_size == document.stat().st_size
    assert recorded.last_scanned_at is not None


def test_scan_without_file_evidence_leaves_dates_untouched(tmp_path: Path) -> None:
    # The prune path records an empty scan for a file that is gone: it has no file
    # to look at, and must not blank out what the last real scan observed.
    document = tmp_path / "note.txt"
    document.write_text("x", encoding="utf-8")
    repo = _repo(tmp_path)
    ingest_document(document, _EmptyDetector(), _EmptyDetector(), repository=repo)
    before = repo.get_document("note")
    assert before is not None and before.reference_date is not None

    repo.record_scan("note", [])

    after = repo.get_document("note")
    assert after is not None
    assert after.reference_date == before.reference_date
    assert after.source_size == before.source_size
