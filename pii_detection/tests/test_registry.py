"""Tests for the detected-PII registry (block B5, Step 1).

The repository is exercised with synthetic :class:`PIIMatch` and the ``ingest``
wiring with fake detectors on a plain-text file, so no Presidio is needed. The
key invariant checked here is **minimization**: no PII value ever reaches a column.
"""

from __future__ import annotations

from pathlib import Path

from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    PIIMatch,
    TextSpan,
)
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


def test_without_replace_a_rescan_appends(tmp_path: Path) -> None:
    # Documents the Step-1 limitation the Step-2 delta will remove.
    repo = _repo(tmp_path)
    matches = [_match(0, 10, "iban")]
    repo.record_scan("doc", matches)
    repo.record_scan("doc", matches)
    assert len(repo.instances_for("doc")) == 2 * len(matches)


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
