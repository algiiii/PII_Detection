"""Document → DB pipeline: extract, detect and persist the PII (B3 + B4 + B5).

Composes the pieces that already exist — B3 extraction and B4 detection via
:func:`~pii_detection.scan.scan_document` — and persists the result into the
detected-PII registry (B5). It is the operational path that **populates the
database from a real document**::

    python -m pii_detection.registry.ingest path/to/document.pdf [--gliner] [--replace]

``--gliner`` swaps spaCy for GLiNER (heavy, container only); ``--replace`` drops
the document's existing instances first (Step-1 way to avoid duplicates on a
re-scan, until the Step-2 delta lands).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.protocol import PIIDetector
from pii_detection.extraction.dates import reference_date
from pii_detection.registry.freshness import stamp_for
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import Scan
from pii_detection.scan import scan_document


def ingest_document(
    path: str | Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    *,
    document_id: str | None = None,
    replace: bool = False,
    detector_signature: str | None = None,
    repository: PIIRepository | None = None,
    merge: MergeEngine | None = None,
) -> Scan:
    """Extract, detect and persist the PII of a document into the registry.

    :param path: path to a ``.pdf``/``.docx``/``.txt`` document.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param document_id: identifier to record the document under; defaults to the
        file stem. A batch scan passes the path relative to its root, so documents
        with the same name in different folders do not collide.
    :param replace: drop the document's existing instances first.
    :param detector_signature: fingerprint of the detection engine in use, stored
        with the document so a later scan can tell the engine changed.
    :param repository: registry to write to; a default one is built when omitted.
    :param merge: merge engine to use; a default one is built when omitted.
    :returns: the created scan.
    """
    repository = repository if repository is not None else PIIRepository()
    matches = scan_document(path, pattern, ner, merge=merge)
    return repository.record_scan(
        document_id if document_id is not None else Path(path).stem,
        matches,
        path=str(path),
        reference_date=reference_date(path),
        stamp=stamp_for(path),
        detector_signature=detector_signature,
        replace=replace,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: ingest a document into the registry and print a summary.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description="Extract, detect and persist a document's PII into the registry."
    )
    parser.add_argument("path", type=Path, help="path to a .pdf, .docx or .txt file")
    parser.add_argument(
        "--gliner",
        action="store_true",
        help="use GLiNER for the NER instead of spaCy (heavy; container only)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="drop the document's existing instances before recording this scan",
    )
    args = parser.parse_args(argv)
    # Lazy import: pulls Presidio only when actually running the CLI.
    from pii_detection.detection.presidio_detector import build_default_detectors

    pattern, ner = build_default_detectors(use_gliner=args.gliner)
    repository = PIIRepository()
    ingest_document(args.path, pattern, ner, replace=args.replace, repository=repository)

    instances = repository.instances_for(args.path.stem)
    counts = Counter(instance.pii_type for instance in instances)
    print(f"Registry now holds {len(instances)} PII for '{args.path.stem}':")
    for pii_type, count in sorted(counts.items()):
        print(f"  {pii_type:<16} {count}")


if __name__ == "__main__":
    main()


__all__ = ["ingest_document", "main"]
