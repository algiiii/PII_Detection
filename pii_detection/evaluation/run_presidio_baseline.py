"""Score the detection stack's pattern and NER layers separately on an annotated corpus.

Executable entry point — ``python -m pii_detection.evaluation.run_presidio_baseline``
— the twin of :mod:`~pii_detection.evaluation.run_baseline` for Presidio. It prints
three tables — the pattern layer alone, the NER alone, and their union — so the NER's
quality (recall on names, false positives on structured data) is visible on its own,
comparable to the pattern baseline.

The detectors come from
:func:`~pii_detection.detection.presidio_detector.build_default_detectors`, the same
factory the scan CLI and the other runners use. The pattern layer is therefore the
**composite** one — Presidio's built-in recognizers *plus* the config-driven rules of
``custom_patterns.yaml`` — so these tables measure the stack the system actually runs,
not a subset of it. Measuring bare Presidio instead would silently drop every custom
rule (the Swiss AVS number among them) and make this report disagree with every other
table computed from the same corpus.

Loads the real Italian spaCy model, so it is slow; run it on demand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pii_detection.detection.presidio_detector import build_default_detectors
from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.types import DetectorKind, PIICandidate
from pii_detection.evaluation.corpus import load_annotated_corpus, sample_documents
from pii_detection.evaluation.scoring import evaluate, format_report


class _UnionDetector:
    """Concatenate the candidates of several detectors, without merging.

    Stands in for "Presidio as a whole" in the benchmark: the raw union of pattern
    and NER candidates, so overlaps and NER false positives show up as they are
    (the real merge is B4's ``MergeEngine``, out of scope here).

    :ivar detector_id: fixed identifier.
    :ivar detector_kind: nominal kind (the union spans techniques).
    """

    detector_id: str = "presidio.union"
    detector_kind: DetectorKind = DetectorKind.REGEX

    def __init__(self, detectors: list[PIIDetector]) -> None:
        """:param detectors: the detectors whose candidates are concatenated."""
        self._detectors = detectors

    def detect(self, text: str) -> list[PIICandidate]:
        """:returns: every candidate of every wrapped detector, in order."""
        return [cand for detector in self._detectors for cand in detector.detect(text)]


def main() -> None:
    """Build the Presidio detectors, score pattern/NER/union on the corpus, print."""
    parser = argparse.ArgumentParser(description="Benchmark Presidio on the annotated corpus.")
    parser.add_argument(
        "--gliner",
        action="store_true",
        help="use GLiNER for the NER instead of spaCy (heavy; needs the [ner] deps)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="annotated corpus: a dir of .txt or a sources.jsonl (default: the packaged one)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="score a random subset of N documents, reproducible via --seed",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed fixing the --sample draw")
    args = parser.parse_args()

    pattern, ner = build_default_detectors(use_gliner=args.gliner)
    corpus = list(load_annotated_corpus(args.corpus))
    if args.sample is not None:
        corpus = sample_documents(corpus, args.sample, args.seed)
    occurrences = sum(len(document.spans) for document in corpus)
    print(f"corpus: {len(corpus)} documents, {occurrences} annotated PII occurrences")

    ner_label = "Presidio NER (GLiNER)" if args.gliner else "Presidio NER (spaCy)"
    rows: list[tuple[str, PIIDetector]] = [
        ("Pattern layer (Presidio + custom regex/checksum)", pattern),
        (ner_label, ner),
        ("Raw union (pattern + NER, no merge)", _UnionDetector([pattern, ner])),
    ]
    for label, detector in rows:
        print(f"\n=== {label} ===")
        print(format_report(evaluate(detector, corpus)))


if __name__ == "__main__":
    main()
