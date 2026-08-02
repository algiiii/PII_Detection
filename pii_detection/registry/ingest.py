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
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.protocol import PIIDetector
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import Scan
from pii_detection.scan import scan_document


def ingest_document(
    path: str | Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    *,
    replace: bool = False,
    repository: PIIRepository | None = None,
    merge: MergeEngine | None = None,
) -> Scan:
    """Extract, detect and persist the PII of a document into the registry.

    :param path: path to a ``.pdf``/``.docx``/``.txt`` document.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param replace: drop the document's existing instances first.
    :param repository: registry to write to; a default one is built when omitted.
    :param merge: merge engine to use; a default one is built when omitted.
    :returns: the created scan.
    """
    repository = repository if repository is not None else PIIRepository()
    matches = scan_document(path, pattern, ner, merge=merge)
    modified_at = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    return repository.record_scan(
        Path(path).stem,
        matches,
        path=str(path),
        source_modified_at=modified_at,
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
