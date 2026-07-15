"""Run the regex detector over the annotated corpus and print the scores.

Executable entry point — ``python -m pii_detection.evaluation.run_baseline`` —
to reproduce the current baseline table on screen. Thin wiring only: config →
:class:`~pii_detection.detection.regex_detector.RegexDetector` → corpus →
:func:`~pii_detection.evaluation.scoring.evaluate` →
:func:`~pii_detection.evaluation.scoring.format_report`.
"""

from __future__ import annotations

from pii_detection.detection.config import load_detection_config
from pii_detection.detection.regex_detector import RegexDetector
from pii_detection.evaluation.corpus import load_corpus_dir
from pii_detection.evaluation.scoring import evaluate, format_report


def main() -> None:
    """Build the regex baseline, score it on the corpus, and print the report."""
    config = load_detection_config()
    detector = RegexDetector("regex.main", list(config.regex_rules))
    report = evaluate(detector, load_corpus_dir())
    print(format_report(report))


if __name__ == "__main__":
    main()
