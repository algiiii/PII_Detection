"""Tests for the batch folder scan (block B5 batch driver).

A temporary folder tree is scanned with fake detectors (no Presidio), checking the
recursive walk and the relative-path identity, that unsupported/unreadable files are
handled without aborting, the folder-wide inventory, and the prune of a file removed
from a sub-folder.
"""

from __future__ import annotations

from pathlib import Path

from pii_detection.detection.ai_detector import AITriggerPolicy
from pii_detection.detection.types import (
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.scan_folder import ingest_folder, plan_folder

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


class _CountingAIDetector:
    """Fake AI detector that records how many documents it was asked to analyse."""

    detector_id = "ai.counting"
    detector_kind = DetectorKind.AI

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, text: str) -> list[PIICandidate]:
        self.calls += 1
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


def test_plan_folder_enumerates_scannable_and_skipped(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)

    plan = plan_folder(folder)

    # extension-based: bad.txt is scannable (it only fails later, at ingest time)
    assert {doc_id for _p, doc_id in plan.scannable} == {"a.txt", "b.txt", "bad.txt", "sub/c.txt"}
    assert [p.name for p in plan.skipped] == ["skip.bin"]


def test_ingest_folder_reports_progress(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    calls: list[tuple[int, int]] = []

    ingest_folder(
        folder,
        _pattern(),
        _EmptyDetector(),
        repository=_repo(tmp_path),
        progress=lambda done, total: calls.append((done, total)),
    )

    # one call per scannable file (a.txt, b.txt, bad.txt, sub/c.txt), in order
    assert [done for done, _ in calls] == [1, 2, 3, 4]
    assert calls[-1] == (4, 4)


def _four_file_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(4):  # names sort as 0,1,2,3 -> stable sampling indices
        (folder / f"{i}.txt").write_text(f"file {i} IBAN {_IBAN}", encoding="utf-8")
    return folder


def test_ai_runs_on_every_document_without_policy(tmp_path: Path) -> None:
    """No policy: the AI detector is run on every analysed file."""
    ai = _CountingAIDetector()
    result = ingest_folder(
        _four_file_folder(tmp_path), _pattern(), _EmptyDetector(), ai, repository=_repo(tmp_path)
    )
    assert ai.calls == 4
    assert result.ai_documents == 4


def test_ai_rate_one_runs_on_every_document(tmp_path: Path) -> None:
    """Rate 1 selects every document (the sampled fractions are unit-tested on the
    policy itself; here we check the folder driver honours rate 1 = all)."""
    ai = _CountingAIDetector()
    result = ingest_folder(
        _four_file_folder(tmp_path),
        _pattern(),
        _EmptyDetector(),
        ai,
        repository=_repo(tmp_path),
        ai_policy=AITriggerPolicy(sampling_rate=1),
    )
    assert ai.calls == 4
    assert result.ai_documents == 4


def test_no_ai_detector_never_runs_ai(tmp_path: Path) -> None:
    """``ai=None`` runs no AI even when the policy would select every file."""
    result = ingest_folder(
        _four_file_folder(tmp_path),
        _pattern(),
        _EmptyDetector(),
        None,
        repository=_repo(tmp_path),
        ai_policy=AITriggerPolicy(sampling_rate=1),
    )
    assert result.ai_documents == 0


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


# --- incremental scan ---------------------------------------------------------


def test_second_scan_skips_everything_unchanged(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)
    first = ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    second = ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    assert second.scanned == 0
    assert set(second.unchanged) == {"a.txt", "b.txt", "sub/c.txt"}
    assert second.by_type == first.by_type  # the inventory still counts their PII


def test_unchanged_files_are_never_pruned(tmp_path: Path) -> None:
    # The regression the incremental path could introduce: `seen` drives the prune,
    # so a document skipped as unchanged must still count as seen — otherwise the
    # second scan would mark every untouched file's PII as REMOVED.
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)
    ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    result = ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    assert result.removed == []
    assert [i.pii_type for i in repo.instances_for("a.txt")] == ["iban"]


def test_errored_file_is_not_pruned(tmp_path: Path) -> None:
    # bad.txt is on disk but fails to extract. It used to miss `seen` entirely, so
    # the prune reported it as gone — a file that is right there.
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)
    ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    result = ingest_folder(
        folder, _pattern(), _EmptyDetector(), repository=repo, incremental=False
    )

    assert "bad.txt" not in result.removed


def test_modified_file_is_rescanned_alone(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)
    ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    (folder / "b.txt").write_text(f"ora anche IBAN {_IBAN}", encoding="utf-8")
    result = ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    assert result.scanned == 1
    assert "b.txt" not in result.unchanged
    assert {i.pii_type for i in repo.instances_for("b.txt")} == {"iban"}


def test_full_flag_reanalyses_everything(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)
    ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    result = ingest_folder(
        folder, _pattern(), _EmptyDetector(), repository=repo, incremental=False
    )

    assert result.scanned == 3
    assert result.unchanged == []


def test_changing_the_engine_invalidates_the_skip(tmp_path: Path) -> None:
    # Same bytes on disk, different detectors: the previous results no longer
    # describe what this engine would find, so the skip must not apply.
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)
    ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)

    other = _SubstringDetector([("iban", _IBAN)])
    other.detector_id = "fake.other"
    result = ingest_folder(folder, other, _EmptyDetector(), repository=repo)

    assert result.scanned == 3
    assert result.unchanged == []


def test_progress_counts_only_the_files_actually_analysed(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    repo = _repo(tmp_path)
    ingest_folder(folder, _pattern(), _EmptyDetector(), repository=repo)
    (folder / "b.txt").write_text("cambiato", encoding="utf-8")

    seen: list[tuple[int, int]] = []
    result = ingest_folder(
        folder,
        _pattern(),
        _EmptyDetector(),
        repository=repo,
        progress=lambda d, t: seen.append((d, t)),
    )

    # Two files are attempted: the modified b.txt, and bad.txt — which never got a
    # recorded stamp because it always fails to extract, so it is retried rather
    # than skipped. A file that could never be read is not a file known to be
    # unchanged, and the next run might well succeed (a missing converter installed,
    # a truncated file replaced).
    assert seen == [(1, 2), (2, 2)]
    assert result.scanned == 1
    assert [path.name for path, _ in result.errors] == ["bad.txt"]
