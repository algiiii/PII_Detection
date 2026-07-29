"""Scan a document for PII and print the findings (B3 + B4).

Product-facing CLI: given a path to a real document, extract its text (B3) and
run the detection pipeline (Presidio pattern + NER, merged by
:class:`~pii_detection.detection.pipeline.MergeEngine`, B4), then print the PII
found. The value of each PII is shown on screen — the whole point of the tool for
a DPO — and lives only in memory (§2.3.11), never persisted here.

    python -m pii_detection.scan path/to/document.pdf

``--gliner`` swaps spaCy for GLiNER (heavy, container only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.types import PIIMatch
from pii_detection.extraction import extract_document


def scan_document(
    path: str | Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    *,
    merge: MergeEngine | None = None,
) -> list[PIIMatch]:
    """Extract a document and detect the PII in it.

    :param path: path to a ``.pdf``/``.docx``/``.txt`` document.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param merge: merge engine to use; a default one is built when omitted.
    :returns: the merged matches, ordered by position in the document.
    """
    merge = merge if merge is not None else MergeEngine()
    document = extract_document(path)
    matches = merge.merge(
        pattern.detect(document.text),
        ner.detect(document.text),
        document_id=document.document_id,
    )
    return sorted(matches, key=lambda match: match.span.start)


def format_matches(matches: list[PIIMatch], *, document_id: str) -> str:
    """Render the found PII as a readable table.

    :param matches: the matches to show.
    :param document_id: identifier printed in the header.
    :returns: a multi-line string ready to print.
    """
    if not matches:
        return f"No PII found in '{document_id}'."
    header = f"{'pii_type':<14}{'value':<34}{'conf':>5}  {'confirmation':<17}sources"
    lines = [f"Found {len(matches)} PII in '{document_id}':", "", header, "-" * len(header)]
    for match in matches:
        # Collapse whitespace: a detected span can cross a line break and would
        # otherwise break the table layout.
        flat = " ".join(match.text.split())
        value = flat if len(flat) <= 32 else flat[:29] + "..."
        sources = ",".join(provenance.detector_id for provenance in match.sources)
        lines.append(
            f"{match.pii_type:<14}{value:<34}{match.confidence:>5.2f}  "
            f"{match.confirmation_level.value:<17}{sources}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: scan the given document and print the PII found.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Scan a document for PII (B3 + B4).")
    parser.add_argument("path", type=Path, help="path to a .pdf, .docx or .txt file")
    parser.add_argument(
        "--gliner",
        action="store_true",
        help="use GLiNER for the NER instead of spaCy (heavy; container only)",
    )
    args = parser.parse_args(argv)
    # Lazy import: pulls Presidio only when actually scanning, so importing this
    # module (for scan_document/format_matches) needs no heavy deps.
    from pii_detection.detection.presidio_detector import build_default_detectors

    pattern, ner = build_default_detectors(use_gliner=args.gliner)
    matches = scan_document(args.path, pattern, ner)
    print(format_matches(matches, document_id=args.path.stem))


if __name__ == "__main__":
    main()


__all__ = ["scan_document", "format_matches", "main"]
