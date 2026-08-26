"""End-to-end pipeline evaluation: the extraction tax (Tier 1 vs Tier 2).

Executable entry point — ``python -m pii_detection.evaluation.run_pipeline`` —
that closes the loop over a corpus of real document files. For each document it
runs the same detection pipeline (Presidio pattern + NER, merged by
:class:`~pii_detection.detection.pipeline.MergeEngine`) **twice**:

- on the clean text (Tier 1, no extraction), and
- on the text extracted from the file by B3 (Tier 2),

then scores both against the value-based gold with
:func:`~pii_detection.evaluation.scoring.evaluate_values`. The gap between the two
F1 scores is the **cost of going through extraction**.

Two corpus layouts are supported, sharing the whole scoring path:

- the **rendered** corpus (:func:`evaluate_rendered_corpus`): flat documents with
  ``clean/<id>.txt`` beside ``<id>.pdf``, as produced by
  :func:`~pii_detection.evaluation.render.render_corpus`;
- the **enterprise** corpus (:func:`evaluate_enterprise_corpus`): a folder tree of
  real files, with the clean text in ``sources.jsonl`` and each ``document_id``
  already carrying its own extension.

The Presidio setup mirrors :mod:`~pii_detection.evaluation.run_presidio_baseline`;
``--gliner`` swaps spaCy for GLiNER (heavy, container only).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.presidio_detector import build_default_detectors
from pii_detection.detection.protocol import PIIDetector
from pii_detection.evaluation.corpus import parse_annotated_text
from pii_detection.evaluation.render import default_rendered_dir, load_gold
from pii_detection.evaluation.scoring import EvaluationReport, evaluate_values, format_report
from pii_detection.extraction import extract_document


def _detect_values(
    text: str,
    pattern: PIIDetector,
    ner: PIIDetector,
    merge: MergeEngine,
    document_id: str,
    ai: PIIDetector | None = None,
) -> list[tuple[str, str]]:
    """Run the pattern+NER (+optional AI) detectors, merge, and return ``(pii_type, value)``.

    The ``ai`` slot lets the benchmark (Step 6) score the union pipeline including
    the generative pass; when ``None`` the merge receives no AI candidates.
    """
    matches = merge.merge(
        pattern.detect(text),
        ner.detect(text),
        ai.detect(text) if ai is not None else (),
        document_id=document_id,
    )
    return [(match.pii_type, match.text) for match in matches]


def _extracted_text(path: Path) -> str:
    """Extract a document's text, degrading to the empty string on failure.

    A file B3 cannot read is not an evaluation crash but a **result**: its PII go
    unfound in Tier 2 and the extraction tax records the loss. Returning ``""``
    keeps a single unreadable file from aborting a corpus-wide run.

    :param path: the document to extract.
    :returns: the extracted text, or ``""`` if extraction failed.
    """
    try:
        return extract_document(path).text
    except Exception:  # noqa: BLE001 — any reader failure is a Tier-2 miss, not a crash
        return ""


def _score_tiers(
    sources: Mapping[str, str],
    files: Mapping[str, Path],
    gold: Mapping[str, list[tuple[str, str]]],
    pattern: PIIDetector,
    ner: PIIDetector,
    merge: MergeEngine,
) -> tuple[EvaluationReport, EvaluationReport]:
    """Score one detect+merge pass per tier over the documents of ``gold``.

    The shared core of both corpus layouts: they differ only in *where* the clean
    text and the file to extract come from, never in how the two tiers are scored.

    :param sources: clean text per ``document_id`` (missing = empty text).
    :param files: file to extract per ``document_id`` (missing = no Tier-2 text).
    :param gold: value-based gold, keyed by ``document_id``.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param merge: the merge engine.
    :returns: ``(clean_report, extracted_report)``.
    """
    clean_by_doc: dict[str, list[tuple[str, str]]] = {}
    extracted_by_doc: dict[str, list[tuple[str, str]]] = {}
    for doc_id in sorted(gold):
        clean_text = sources.get(doc_id, "")
        path = files.get(doc_id)
        extracted = _extracted_text(path) if path is not None else ""
        clean_by_doc[doc_id] = _detect_values(clean_text, pattern, ner, merge, doc_id)
        extracted_by_doc[doc_id] = _detect_values(extracted, pattern, ner, merge, doc_id)
    return evaluate_values(clean_by_doc, gold), evaluate_values(extracted_by_doc, gold)


def evaluate_rendered_corpus(
    directory: Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    *,
    file_format: str = "pdf",
    merge: MergeEngine | None = None,
) -> tuple[EvaluationReport, EvaluationReport]:
    """Score a rendered corpus on both the clean and the extracted text.

    Reads the value gold and, per document, the clean text (``clean/<id>.txt``)
    and the text extracted from ``<id>.<file_format>`` (B3), running the same
    detect+merge on each.

    :param directory: a rendered-corpus directory (``gold.jsonl`` + ``clean/`` +
        the rendered files), as produced by
        :func:`~pii_detection.evaluation.render.render_corpus`.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param file_format: rendered extension to read for the extracted tier.
    :param merge: merge engine to use; a default one is built when omitted.
    :returns: ``(clean_report, extracted_report)``.
    """
    merge = merge if merge is not None else MergeEngine()
    gold = load_gold(directory / "gold.jsonl")
    sources = {
        doc_id: (directory / "clean" / f"{doc_id}.txt").read_text(encoding="utf-8")
        for doc_id in gold
    }
    files = {doc_id: directory / f"{doc_id}.{file_format}" for doc_id in gold}
    return _score_tiers(sources, files, gold, pattern, ner, merge)


def evaluate_enterprise_corpus(
    root: Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    *,
    merge: MergeEngine | None = None,
) -> tuple[EvaluationReport, EvaluationReport]:
    """Score the enterprise corpus on both the clean and the extracted text.

    Same two tiers as :func:`evaluate_rendered_corpus`, over the folder tree
    produced by the enterprise generator: the clean text comes from the annotated
    ``sources.jsonl`` (markers stripped) and the Tier-2 text from the real file at
    ``tree/<document_id>`` — the id already carries its own extension, so unlike
    the rendered corpus there is no single format to choose.

    :param root: the corpus root holding ``gold.jsonl``, ``sources.jsonl`` and ``tree/``.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param merge: merge engine to use; a default one is built when omitted.
    :returns: ``(clean_report, extracted_report)``.
    :raises FileNotFoundError: if ``gold.jsonl`` or ``sources.jsonl`` is missing.
    """
    merge = merge if merge is not None else MergeEngine()
    gold = load_gold(root / "gold.jsonl")
    sources_path = root / "sources.jsonl"
    if not sources_path.is_file():
        raise FileNotFoundError(f"corpus file not found: {sources_path}")
    sources: dict[str, str] = {}
    for line in sources_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        document_id = str(record["document_id"])
        sources[document_id] = parse_annotated_text(document_id, str(record["annotated"])).text
    files = {doc_id: root / "tree" / doc_id for doc_id in gold}
    return _score_tiers(sources, files, gold, pattern, ner, merge)


def main() -> None:
    """Build the Presidio detectors, evaluate both tiers on the corpus, print."""
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation (extraction tax).")
    parser.add_argument("--dir", type=Path, default=default_rendered_dir())
    parser.add_argument("--format", choices=["pdf", "docx"], default="pdf", dest="file_format")
    parser.add_argument(
        "--enterprise",
        type=Path,
        default=None,
        metavar="ROOT",
        help="score the enterprise corpus at ROOT (gold.jsonl + sources.jsonl + tree/) "
        "instead of the rendered corpus; --dir and --format are then ignored",
    )
    parser.add_argument(
        "--gliner",
        action="store_true",
        help="use GLiNER for the NER instead of spaCy (heavy; container only)",
    )
    args = parser.parse_args()

    pattern, ner = build_default_detectors(use_gliner=args.gliner)

    if args.enterprise is not None:
        clean_report, extracted_report = evaluate_enterprise_corpus(args.enterprise, pattern, ner)
        tier2_label = "extracted from the enterprise tree"
    else:
        clean_report, extracted_report = evaluate_rendered_corpus(
            args.dir, pattern, ner, file_format=args.file_format
        )
        tier2_label = f"extracted from {args.file_format.upper()}"

    print("=== Tier 1 — clean text (detection only) ===")
    print(format_report(clean_report))
    print(f"\n=== Tier 2 — {tier2_label} (B3 + detection) ===")
    print(format_report(extracted_report))
    tax = clean_report.overall.f1 - extracted_report.overall.f1
    print(f"\nExtraction tax (F1 clean - F1 extracted): {tax:+.3f}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_rendered_corpus", "evaluate_enterprise_corpus", "main"]
