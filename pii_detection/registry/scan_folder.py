"""Scan the files in a folder and persist each document's PII (batch, B3+B4+B5).

The operational path behind the "point it at a folder and let it run" deployment:
read every supported document **directly inside** a folder (not its sub-folders —
no file-system structure analysis here) and keep the detected-PII registry in sync
with what is on disk. It composes the per-document pipeline
(:func:`~pii_detection.registry.ingest.ingest_document`) that already exists, adding
only the folder walk, a per-file identity and the reconciliation of files that
disappeared::

    python -m pii_detection.registry.scan_folder path/to/folder [--gliner] [--no-prune]

Each file is recorded under a ``document_id`` equal to its file name (names are
unique within a folder). ``--gliner`` swaps spaCy for GLiNER (heavy, container
only). By default a document already in the registry but no longer in the folder is
reconciled as gone (its PII marked ``REMOVED``); ``--no-prune`` disables it.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


def ingest_folder(
    folder: str | Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    *,
    repository: PIIRepository | None = None,
    merge: MergeEngine | None = None,
    prune: bool = True,
) -> FolderScanResult:
    """Ingest every supported document directly inside ``folder`` (batch scan).

    Reads the files in ``folder`` (not recursively) and ingests each
    ``.pdf``/``.docx``/``.txt`` under a ``document_id`` equal to its file name. Each
    file is isolated: one that fails to extract is recorded in ``errors`` and the
    scan continues. Re-scanning updates each document through the B5 delta.

    When ``prune`` is set, documents already in the registry that were **not** seen
    in this scan and still have present PII are reconciled as gone: recording an
    empty scan marks their instances ``REMOVED``. This assumes the registry watches
    a single folder (one registry per monitored folder).

    :param folder: directory whose files to scan (not recursive).
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param repository: registry to write to; a default one is built when omitted.
    :param merge: merge engine reused across files; a default one when omitted.
    :param prune: reconcile documents gone from the folder as removed.
    :returns: a :class:`FolderScanResult` summary, including the folder-wide inventory.
    :raises NotADirectoryError: if ``folder`` is not a directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    repository = repository if repository is not None else PIIRepository()
    merge = merge if merge is not None else MergeEngine()
    supported = supported_suffixes()

    result = FolderScanResult()
    seen: set[str] = set()
    for path in sorted(item for item in folder.iterdir() if item.is_file()):
        if path.suffix.lower() not in supported:
            result.skipped.append(path)
            continue
        document_id = path.name
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
            continue
        seen.add(document_id)
        result.scanned += 1

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
    parser.add_argument("folder", type=Path, help="directory whose files to scan (not recursive)")
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
    args = parser.parse_args(argv)
    # Lazy import: pulls Presidio only when actually running the scan. Detectors are
    # built once here and reused across every file (GLiNER is expensive to load).
    from pii_detection.detection.presidio_detector import build_default_detectors

    pattern, ner = build_default_detectors(use_gliner=args.gliner)
    result = ingest_folder(args.folder, pattern, ner, prune=args.prune)

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


if __name__ == "__main__":
    main()


__all__ = ["FolderScanResult", "ingest_folder", "main"]
