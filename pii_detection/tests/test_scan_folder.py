"""Tests for the batch folder scan (block B5 batch driver).

A temporary folder tree is scanned with fake detectors (no Presidio), checking the
recursive walk and the relative-path identity, that unsupported/unreadable files are
handled without aborting, the folder-wide inventory, and the prune of a file removed
from a sub-folder.
"""

from __future__ import annotations

from pathlib import Path

from pii_detection.detection.types import (
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.scan_folder import ingest_folder

_IBAN = "IT60X0542811101000000123456"
_EMAIL = "mario@x.it"


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


def _pattern() -> _SubstringDetector:
    return _SubstringDetector([("iban", _IBAN), ("email", _EMAIL)])


def _repo(tmp_path: Path) -> PIIRepository:
    return PIIRepository(url=f"sqlite:///{tmp_path}/pii.db")


def _make_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text(f"IBAN {_IBAN} qui", encoding="utf-8")
    (folder / "b.txt").write_text(f"scrivi a {_EMAIL}", encoding="utf-8")
    (folder / "skip.bin").write_text("not a supported format", encoding="utf-8")
    (folder / "bad.txt").write_bytes(b"\xff\xfe not valid utf-8")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text(f"IBAN {_IBAN}", encoding="utf-8")  # scanned recursively
    return folder


def test_scan_is_recursive_and_ingests_files(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)

    result = ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    assert result.scanned == 3  # a.txt, b.txt, sub/c.txt (bad.txt errored, skip.bin skipped)
    ids = {document.document_id for document in repo.documents()}
    assert ids == {"a.txt", "b.txt", "sub/c.txt"}  # id = path relative to the folder (POSIX)
    assert result.by_type == {"iban": 2, "email": 1}  # iban in a.txt and sub/c.txt


def test_unsupported_and_unreadable_are_handled(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)

    result = ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    assert [p.name for p in result.skipped] == ["skip.bin"]
    assert [p.name for p, _ in result.errors] == ["bad.txt"]  # invalid UTF-8, isolated


def test_rescan_prunes_removed_file(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text(f"IBAN {_IBAN}", encoding="utf-8")
    (folder / "b.txt").write_text(f"scrivi a {_EMAIL}", encoding="utf-8")
    repo = _repo(tmp_path)
    ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    (folder / "b.txt").unlink()  # the file disappears from the folder
    result = ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    assert result.removed == ["b.txt"]
    assert repo.instances_for("b.txt") == []  # its PII is no longer present
    removed = repo.instances_for("b.txt", include_removed=True)
    assert removed and all(instance.removed for instance in removed)
    assert {i.pii_type for i in repo.instances_for("a.txt")} == {"iban"}  # still there


def test_rescan_prunes_removed_nested_file(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text(f"scrivi a {_EMAIL}", encoding="utf-8")
    (folder / "sub" / "c.txt").write_text(f"IBAN {_IBAN}", encoding="utf-8")
    repo = _repo(tmp_path)
    ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    (folder / "sub" / "c.txt").unlink()  # a nested file disappears from the tree
    result = ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    assert result.removed == ["sub/c.txt"]  # pruned by its relative-path id
    assert repo.instances_for("sub/c.txt") == []
    assert {i.pii_type for i in repo.instances_for("a.txt")} == {"email"}  # untouched
