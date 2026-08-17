"""Scan the files in a folder and persist each document's PII (batch, B3+B4+B5).

The operational path behind the "point it at a folder and let it run" deployment:
read every supported document under a folder **recursively (sub-folders included)**
and keep the detected-PII registry in sync with what is on disk. It composes the
per-document pipeline (:func:`~pii_detection.registry.ingest.ingest_document`) that
already exists, adding only the tree walk, a per-file identity and the
reconciliation of files that disappeared::

    python -m pii_detection.registry.scan_folder path/to/folder [--gliner] [--ai] [--no-prune] [--full]

Each file is recorded under a ``document_id`` equal to its path relative to the
scanned folder (POSIX, e.g. ``HR/contratti/mario.pdf``), so equally named files in
different sub-folders do not collide. ``--gliner`` swaps spaCy for GLiNER (heavy,
container only). ``--ai`` runs the local LLM second opinion on **every** document
(needs Ollama); without it, the ``PII_AI_SAMPLING_RATE`` environment variable can
still enable the sampled 1-in-N pass. By default a document already in the registry
but no longer under the folder is reconciled as gone (its PII marked ``REMOVED``);
``--no-prune`` disables it. Also by default the scan is **incremental**: a file whose
modification time and size are unchanged since its last scan, and whose scan ran with
the same detection engine, is not read again; ``--full`` re-analyses everything.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pii_detection.detection.ai_detector import AITriggerPolicy
from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.protocol import PIIDetector
from pii_detection.extraction import supported_suffixes
from pii_detection.extraction.dates import as_utc
from pii_detection.registry.freshness import (
    FileStamp,
    detector_signature,
    needs_rescan,
    stamp_for,
)
from pii_detection.registry.ingest import ingest_document
from pii_detection.registry.repository import PIIRepository


def _changed(
    repository: PIIRepository, path: Path, document_id: str, signature: str | None
) -> bool:
    """Whether a file must be analysed again, against what the registry recorded.

    A file the registry cannot vouch for — never seen, or seen without a stamp —
    is always analysed: the incremental path must never *invent* a reason to skip.

    :param repository: the registry holding the previous observation.
    :param path: the file on disk.
    :param document_id: its identity in the registry.
    :param signature: fingerprint of the detection engine about to run, or ``None``
        to ignore engine changes (the preview, which does not know the engine yet).
    :returns: ``True`` when the document must be re-analysed.
    """
    document = repository.get_document(document_id)
    recorded = (
        FileStamp(modified_at=as_utc(document.source_mtime), size=document.source_size)
        if document is not None
        and document.source_mtime is not None
        and document.source_size is not None
        else None
    )
    return needs_rescan(
        stamp_for(path),
        recorded,
        signature=signature,
        recorded_signature=document.detector_signature if document else None,
    )


@dataclass
class FolderScanResult:
    """Summary of scanning the files in a folder into the registry.

    :ivar scanned: number of files successfully ingested.
    :ivar unchanged: document ids skipped because the file did not change since the
        last scan — distinct from :attr:`skipped`, which is about file *formats*.
    :ivar skipped: paths skipped because their extension is not supported.
    :ivar errors: ``(path, message)`` for files that failed to ingest.
    :ivar removed: document ids reconciled as gone from the folder (PII ``REMOVED``).
    :ivar ai_documents: number of files that also ran the generative-AI pass (all of
        them under ``--ai``, the sampled subset under a sampling policy).
    :ivar by_type: current PII count per ``pii_type`` across the scanned files —
        the folder-wide inventory.
    """

    scanned: int = 0
    unchanged: list[str] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    ai_documents: int = 0
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


def count_unchanged(plan: FolderPlan, repository: PIIRepository) -> int:
    """How many of a plan's files an incremental scan would skip.

    Answers the preview's question — "how much of this is already done?" — using
    the very same predicate the scan uses, so the page cannot promise one thing and
    the run do another. The engine signature is deliberately **not** considered
    here: the preview does not know which detectors the run will use, and counting
    a file as unchanged that the run then re-analyses is the honest direction of
    the two errors.

    :param plan: the enumeration produced by :func:`plan_folder`.
    :param repository: the registry holding the previous observations.
    :returns: the number of files that would be skipped as unchanged.
    """
    return sum(
        1
        for path, document_id in plan.scannable
        if not _changed(repository, path, document_id, signature=None)
    )


def ingest_folder(
    folder: str | Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    ai: PIIDetector | None = None,
    *,
    repository: PIIRepository | None = None,
    merge: MergeEngine | None = None,
    prune: bool = True,
    incremental: bool = True,
    progress: Callable[[int, int], None] | None = None,
    ai_policy: AITriggerPolicy | None = None,
) -> FolderScanResult:
    """Ingest every supported document under ``folder``, recursively (batch scan).

    Enumerates the tree with :func:`plan_folder` and ingests each supported file
    under a ``document_id`` equal to its path relative to ``folder`` (POSIX). Each
    file is isolated: one that fails to extract is recorded in ``errors`` and the
    scan continues. Re-scanning updates each document through the B5 delta.

    When ``incremental`` is set (the default), a file whose stamp matches what the
    registry recorded — and that was analysed by the same detection engine — is
    **not** read again: it lands in ``unchanged``. Re-reading a share that has not
    moved is the dominant cost of a periodic scan and buys nothing.

    When ``prune`` is set, documents already in the registry that were **not** seen
    in this scan and still have present PII are reconciled as gone: recording an
    empty scan marks their instances ``REMOVED``. This assumes the registry watches
    a single folder tree (one registry per monitored folder). "Seen" here means
    **enumerated on disk**, not "ingested": a file skipped as unchanged, or one that
    failed to extract, is still on disk and must not be reported as removed.

    The optional ``ai`` detector runs the generative-AI second opinion. Which files
    get it is decided by ``ai_policy``: with no policy every analysed file runs it
    (the manual "AI on everything" mode); with an :class:`AITriggerPolicy` only the
    sampled ones do (``index`` taken over the **full** enumeration ``plan.scannable``,
    so the sample is stable regardless of which files the incremental pass skips).

    :param folder: root directory to scan recursively.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param ai: optional generative-AI detector for the second-opinion pass; when
        ``None`` no AI runs, whatever ``ai_policy`` says.
    :param repository: registry to write to; a default one is built when omitted.
    :param merge: merge engine reused across files; a default one when omitted.
    :param prune: reconcile documents gone from the folder as removed.
    :param incremental: skip files unchanged since their last scan; set ``False`` to
        force a full re-analysis.
    :param progress: optional callback invoked ``progress(done, total)`` after each
        file actually analysed, for a UI progress bar.
    :param ai_policy: sampling policy selecting which files get the AI pass; ``None``
        means every analysed file gets it (when ``ai`` is set).
    :returns: a :class:`FolderScanResult` summary, including the folder-wide inventory.
    :raises NotADirectoryError: if ``folder`` is not a directory.
    """
    repository = repository if repository is not None else PIIRepository()
    merge = merge if merge is not None else MergeEngine()

    plan = plan_folder(folder)
    result = FolderScanResult(skipped=list(plan.skipped))
    signature = detector_signature([pattern.detector_id, ner.detector_id])

    # "Seen" is every file the enumeration found on disk, whatever happened to it
    # afterwards: it is what protects the prune below from removing documents that
    # are still there. The sampling index is the file's position in the full
    # enumeration, so the AI sample is stable across incremental runs.
    seen: set[str] = set()
    todo: list[tuple[Path, str, int]] = []
    for index, (path, document_id) in enumerate(plan.scannable):
        seen.add(document_id)
        if incremental and not _changed(repository, path, document_id, signature):
            result.unchanged.append(document_id)
        else:
            todo.append((path, document_id, index))

    total = len(todo)
    for done, (path, document_id, index) in enumerate(todo, start=1):
        use_ai = ai if ai is not None and (ai_policy is None or ai_policy.selects(index)) else None
        try:
            ingest_document(
                path,
                pattern,
                ner,
                document_id=document_id,
                ai=use_ai,
                repository=repository,
                merge=merge,
                detector_signature=signature,
            )
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the batch
            result.errors.append((path, str(exc)))
        else:
            result.scanned += 1
            if use_ai is not None:
                result.ai_documents += 1
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
    for document_id in seen:  # includes the unchanged ones: they still hold their PII
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
        "--ai",
        action="store_true",
        help="run the local LLM second opinion on every document (needs Ollama)",
    )
    parser.add_argument(
        "--no-prune",
        dest="prune",
        action="store_false",
        help="do not mark documents gone from the folder as removed",
    )
    parser.add_argument(
        "--full",
        dest="incremental",
        action="store_false",
        help="re-analyse every file, including those unchanged since the last scan",
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
    # AI second opinion: --ai runs it on every document (no policy); otherwise the
    # sampling policy from the environment decides, and the detector is built only
    # when that policy is actually enabled (PII_AI_SAMPLING_RATE > 0).
    ai = None
    ai_policy: AITriggerPolicy | None = None
    if args.ai:
        from pii_detection.detection.ai_detector import build_ai_detector

        ai = build_ai_detector()
    else:
        ai_policy = AITriggerPolicy.from_env()
        if ai_policy.enabled:
            from pii_detection.detection.ai_detector import build_ai_detector

            ai = build_ai_detector()
    repository = PIIRepository()
    result = ingest_folder(
        args.folder,
        pattern,
        ner,
        ai,
        repository=repository,
        prune=args.prune,
        incremental=args.incremental,
        ai_policy=ai_policy,
    )

    print(f"Scanned {result.scanned} documents in '{args.folder}'.")
    if result.ai_documents:
        print(f"  AI second opinion: {result.ai_documents} documents")
    if result.unchanged:
        print(f"  unchanged (skipped, already up to date): {len(result.unchanged)}")
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


__all__ = [
    "FolderScanResult",
    "FolderPlan",
    "plan_folder",
    "count_unchanged",
    "ingest_folder",
    "main",
]
