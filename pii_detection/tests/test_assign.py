"""Tests for the explicit document–activity association (block B6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_detection.compliance.assign import ExplicitAssigner, persist_assignment
from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIIMatch,
    TextSpan,
)
from pii_detection.registry.repository import PIIRepository


def _repo(tmp_path: Path) -> PIIRepository:
    return PIIRepository(url=f"sqlite:///{tmp_path}/pii.db")


def _match(pii_type: str) -> PIIMatch:
    provenance = DetectionProvenance("det.x", DetectorKind.REGEX, pii_type, 0.9)
    return PIIMatch(
        span=TextSpan(0, 5),
        text="?",
        pii_type=pii_type,
        confidence=0.9,
        confirmation_level=ConfirmationLevel.SINGLE_SOURCE,
        sources=[provenance],
        document_id="ignored",
    )


def test_explicit_assigner_returns_given_ids_ignoring_content() -> None:
    assigner = ExplicitAssigner(["a", "b"])
    assert assigner.assign("doc", ["iban", "email"]) == ["a", "b"]


def test_persist_assignment_stores_ids_on_document(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.record_scan("doc", [_match("iban")])

    persisted = persist_assignment(repo, "doc", ExplicitAssigner(["paghe", "selezione"]))

    assert persisted == ["paghe", "selezione"]
    document = repo.get_document("doc")
    assert document is not None
    assert document.activity_ids == ["paghe", "selezione"]


def test_assign_activities_rejects_empty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.record_scan("doc", [_match("iban")])
    with pytest.raises(ValueError):
        repo.assign_activities("doc", [])


def test_assign_activities_rejects_blank_id(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.record_scan("doc", [_match("iban")])
    with pytest.raises(ValueError):
        repo.assign_activities("doc", ["ok", "  "])


def test_assign_activities_unknown_document_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(KeyError):
        repo.assign_activities("missing", ["paghe"])
