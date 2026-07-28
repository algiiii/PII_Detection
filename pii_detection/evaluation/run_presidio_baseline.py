"""Run Presidio (regex pattern + spaCy NER) over the annotated corpus and score it.

Executable entry point — ``python -m pii_detection.evaluation.run_presidio_baseline``
— the twin of :mod:`~pii_detection.evaluation.run_baseline` for Presidio. It prints
three tables — the pattern recognizers alone, the NER alone, and their union — so
the NER's quality (recall on names, false positives on structured data) is visible
on its own, comparable to the regex baseline.

Loads the real Italian spaCy model, so it is slow; run it on demand.
"""

from __future__ import annotations

import argparse

from pii_detection.detection.config import (
    default_config_dir,
    load_category_catalog,
    load_presidio_entities,
)
from pii_detection.detection.presidio_detector import (
    build_italian_analyzer,
    build_presidio_detectors,
)
from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.types import DetectorKind, PIICandidate
from pii_detection.evaluation.corpus import load_corpus_dir
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
    args = parser.parse_args()

    catalog = load_category_catalog(default_config_dir() / "categories.yaml")
    entities = load_presidio_entities(default_config_dir() / "presidio_entities.yaml", catalog)
    analyzer = build_italian_analyzer(use_gliner=args.gliner)
    pattern, ner = build_presidio_detectors(entities, analyzer)
    corpus = list(load_corpus_dir())

    ner_label = "Presidio NER (GLiNER)" if args.gliner else "Presidio NER (spaCy)"
    rows: list[tuple[str, PIIDetector]] = [
        ("Presidio pattern (regex/checksum)", pattern),
        (ner_label, ner),
        ("Presidio union (pattern + NER)", _UnionDetector([pattern, ner])),
    ]
    for label, detector in rows:
        print(f"\n=== {label} ===")
        print(format_report(evaluate(detector, corpus)))


if __name__ == "__main__":
    main()
