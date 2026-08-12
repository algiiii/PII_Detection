"""Scan the files in a folder and persist each document's PII (batch, B3+B4+B5).

The operational path behind the "point it at a folder and let it run" deployment:
read every supported document under a folder **recursively (sub-folders included)**
and keep the detected-PII registry in sync with what is on disk. It composes the
per-document pipeline (:func:`~pii_detection.registry.ingest.ingest_document`) that
already exists, adding only the tree walk, a per-file identity and the
reconciliation of files that disappeared::

    python -m pii_detection.registry.scan_folder path/to/folder [--gliner] [--no-prune]

Each file is recorded under a ``document_id`` equal to its path relative to the
scanned folder (POSIX, e.g. ``HR/contratti/mario.pdf``), so equally named files in
different sub-folders do not collide. ``--gliner`` swaps spaCy for GLiNER (heavy,
container only). By default a document already in the registry but no longer under
the folder is reconciled as gone (its PII marked ``REMOVED``); ``--no-prune``
disables it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.protocol import PIIDetector
from pii_detection.extraction import supported_suffixes
from pii_detection.registry.ingest import ingest_document
from pii_detection.registry.repository import PIIRepository


@dataclass
class FolderScanResult:
    """Summary of scanning the files in a folder into the registry.

    :ivar scanned: number of files successfully ingested.
    :ivar skipped: paths skipped because their extension is not supported.
    :ivar errors: ``(path, message)`` for files that failed to ingest.
    :ivar removed: document ids reconciled as gone from the folder (PII ``REMOVED``).
    :ivar by_type: current PII count per ``pii_type`` across the scanned files —
        the folder-wide inventory.
    """

    scanned: int = 0
    skipped: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class FolderPlan:
    """What a recursive scan of a folder would cover — computed without detection.

    :ivar scannable: ``(path, document_id)`` pairs that will be ingested;
        ``document_id`` is the path relative to the folder (POSIX).
    :ivar skipped: paths skipped because their extension is not supported.
    """

    scannable: list[tuple[Path, str]] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)


def plan_folder(folder: str | Path) -> FolderPlan:
    """Enumerate the files a recursive scan of ``folder`` would cover.

    Pure enumeration (no detection): used both to **preview** a scan and to drive
    :func:`ingest_folder`, so the two always agree on what gets scanned.

    :param folder: root directory to enumerate recursively.
    :returns: the :class:`FolderPlan` (scannable files + skipped ones), each list
        sorted by path.
    :raises NotADirectoryError: if ``folder`` is not a directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    supported = supported_suffixes()
    plan = FolderPlan()
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        if path.suffix.lower() in supported:
            plan.scannable.append((path, path.relative_to(folder).as_posix()))
        else:
            plan.skipped.append(path)
    return plan


def ingest_folder(
    folder: str | Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    *,
    repository: PIIRepository | None = None,
    merge: MergeEngine | None = None,
    prune: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> FolderScanResult:
    """Ingest every supported document under ``folder``, recursively (batch scan).

    Enumerates the tree with :func:`plan_folder` and ingests each supported file
    under a ``document_id`` equal to its path relative to ``folder`` (POSIX). Each
    file is isolated: one that fails to extract is recorded in ``errors`` and the
    scan continues. Re-scanning updates each document through the B5 delta.

    When ``prune`` is set, documents already in the registry that were **not** seen
    in this scan and still have present PII are reconciled as gone: recording an
    empty scan marks their instances ``REMOVED``. This assumes the registry watches
    a single folder tree (one registry per monitored folder).

    :param folder: root directory to scan recursively.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param repository: registry to write to; a default one is built when omitted.
    :param merge: merge engine reused across files; a default one when omitted.
    :param prune: reconcile documents gone from the folder as removed.
    :param progress: optional callback invoked ``progress(done, total)`` after each
        file (whether ingested or errored), for a UI progress bar.
    :returns: a :class:`FolderScanResult` summary, including the folder-wide inventory.
    :raises NotADirectoryError: if ``folder`` is not a directory.
    """
    repository = repository if repository is not None else PIIRepository()
    merge = merge if merge is not None else MergeEngine()

    plan = plan_folder(folder)
    result = FolderScanResult(skipped=list(plan.skipped))
    seen: set[str] = set()
    total = len(plan.scannable)
    for done, (path, document_id) in enumerate(plan.scannable, start=1):
        try:
            ingest_document(
                path,
                pattern,
                ner,
                document_id=document_id,
                repository=repository,
                merge=merge,
            )
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the batch
            result.errors.append((path, str(exc)))
        else:
            seen.add(document_id)
            result.scanned += 1
        if progress is not None:
            progress(done, total)

    if prune:
        for document in repository.documents():
            if document.document_id in seen:
                continue
            if repository.instances_for(document.document_id):  # still has present PII
                repository.record_scan(document.document_id, [])
                result.removed.append(document.document_id)

    counts: Counter[str] = Counter()
    for document_id in seen:
        for instance in repository.instances_for(document_id):
            counts[instance.pii_type] += 1
    result.by_type = dict(counts)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI: scan a folder's files, persist the PII and print the summary.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description="Scan the files in a folder and persist each document's PII (batch)."
    )
    parser.add_argument("folder", type=Path, help="root directory to scan recursively (sub-folders included)")
    parser.add_argument(
        "--gliner",
        action="store_true",
        help="use GLiNER for the NER instead of spaCy (heavy; container only)",
    )
    parser.add_argument(
        "--no-prune",
        dest="prune",
        action="store_false",
        help="do not mark documents gone from the folder as removed",
    )
    parser.add_argument(
        "--no-apply-rules",
        dest="apply_rules",
        action="store_false",
        help="do not apply folder→activity rules after the scan",
    )
    args = parser.parse_args(argv)
    # Lazy import: pulls Presidio only when actually running the scan. Detectors are
    # built once here and reused across every file (GLiNER is expensive to load).
    from pii_detection.detection.presidio_detector import build_default_detectors

    pattern, ner = build_default_detectors(use_gliner=args.gliner)
    repository = PIIRepository()
    result = ingest_folder(
        args.folder, pattern, ner, repository=repository, prune=args.prune
    )

    print(f"Scanned {result.scanned} documents in '{args.folder}'.")
    if result.skipped:
        print(f"  skipped (unsupported): {len(result.skipped)}")
    if result.removed:
        print(f"  marked removed (gone from folder): {len(result.removed)}")
        for document_id in result.removed:
            print(f"    - {document_id}")
    if result.errors:
        print(f"  errors: {len(result.errors)}")
        for path, message in result.errors:
            print(f"    - {path}: {message}")
    if result.by_type:
        total = sum(result.by_type.values())
        print(f"\nPII inventory ({total} across the folder):")
        for pii_type, count in sorted(result.by_type.items()):
            print(f"  {pii_type:<16} {count}")
    if args.apply_rules:
        applied = repository.apply_folder_rules()
        print(
            f"\nFolder rules applied: {applied.associated} associated, "
            f"{applied.skipped_manual} kept manual, {applied.unmatched} unmatched."
        )


if __name__ == "__main__":
    main()


__all__ = ["FolderScanResult", "FolderPlan", "plan_folder", "ingest_folder", "main"]
