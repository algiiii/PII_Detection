"""Multi-model AI benchmark: does the generative pass earn its cost? (block B4).

The thesis question in one table. For each candidate model it scores three things on
the same annotated corpus and prints them side by side:

- **quality** — precision / recall / F1 (span-based, via
  :func:`~pii_detection.evaluation.scoring.evaluate`);
- **latency** — seconds per document (``time.perf_counter`` around the scoring);
- **energy** — watt-hours per document, estimated with `codecarbon` (optional).

Three kinds of row make the AI's *net* contribution visible: a **baseline**
(``pattern + NER`` merged — the system today, no AI), the **AI alone** for each model
(``ai:<model>``), and the **union** (``pattern + NER + AI`` merged — the system with
that model, ``union:<model>``). The gap ``union − baseline`` is what the AI adds;
``ai`` on its own shows what the model finds by itself.

The default model list spans **three size tiers** (~4B, ~7–8B, ~12–14B): a benchmark
limited to 4B would measure the deployment compromise, not the AI's potential, which
is the real question — *how much does a larger model recover, and is it worth the
seconds and watt-hours on CPU?*

.. note::
   **Energy figures are comparative estimates, not measurements.** `codecarbon`
   falls back to a TDP-based estimate where RAPL is unavailable (containers, macOS),
   and the inference runs in the sibling Ollama container, not in this process — so
   the watt-hours are meaningful *relative to each other*, to be produced with the
   app and Ollama co-located, not as absolute consumption.

Run it in the container (the models need Ollama):
``python -m pii_detection.evaluation.run_ai_benchmark --limit 10``. On CPU the whole
corpus × five models is many hours (the 12B ones more), so use ``--limit`` for
exploratory runs.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.types import DetectionProvenance, DetectorKind, PIICandidate
from pii_detection.evaluation.corpus import AnnotatedDocument, load_corpus_dir
from pii_detection.evaluation.scoring import EvaluationReport, evaluate

#: Default models to benchmark — the three size tiers (see the module docstring).
DEFAULT_MODELS = ("phi4-mini", "qwen3:4b", "gemma3:4b", "qwen2.5:7b", "gemma3:12b")


@dataclass(frozen=True)
class BenchmarkRow:
    """One measured row of the benchmark table.

    :ivar label: row name, e.g. ``"union:phi4-mini"``.
    :ivar report: the quality metrics (overall + per category).
    :ivar seconds_per_doc: wall-clock seconds per document for this row.
    :ivar wh_per_doc: estimated watt-hours per document, or ``None`` when energy
        tracking was off or unavailable.
    """

    label: str
    report: EvaluationReport
    seconds_per_doc: float
    wh_per_doc: float | None


class MergedPipelineDetector:
    """Adapter: the full merged pipeline (pattern + NER + AI) as one detector.

    Runs the three sources and the :class:`~pii_detection.detection.pipeline.MergeEngine`,
    then exposes the merged result as plain
    :class:`~pii_detection.detection.types.PIICandidate` objects so the span scorer
    (:func:`~pii_detection.evaluation.scoring.evaluate`), which reads
    ``provenance.pii_type`` and ``span``, can score the system's real output. With
    ``ai=None`` it is the baseline (the system without the generative pass).

    :ivar detector_id: fixed identifier.
    :ivar detector_kind: nominal kind (the pipeline spans techniques).
    """

    detector_id: str = "pipeline.merged"
    detector_kind: DetectorKind = DetectorKind.REGEX

    def __init__(
        self,
        pattern: PIIDetector,
        ner: PIIDetector,
        ai: PIIDetector | None,
        *,
        merge: MergeEngine | None = None,
    ) -> None:
        """Store the sources and the merge engine.

        :param pattern: the pattern/regex detector.
        :param ner: the NER detector.
        :param ai: the AI detector, or ``None`` for the baseline.
        :param merge: merge engine to use; a default one is built when omitted.
        """
        self._pattern = pattern
        self._ner = ner
        self._ai = ai
        self._merge = merge if merge is not None else MergeEngine()

    def detect(self, text: str) -> list[PIICandidate]:
        """Detect with all sources, merge, and return the merged spans as candidates.

        :param text: the document text.
        :returns: one candidate per merged match, carrying the merged ``pii_type``.
        """
        matches = self._merge.merge(
            self._pattern.detect(text),
            self._ner.detect(text),
            self._ai.detect(text) if self._ai is not None else (),
            document_id="benchmark",
        )
        return [
            PIICandidate(
                match.span,
                match.text,
                DetectionProvenance(
                    self.detector_id, self.detector_kind, match.pii_type, match.confidence
                ),
            )
            for match in matches
        ]


def _make_tracker() -> object:
    """Build a `codecarbon` offline tracker (imported lazily; energy runs only)."""
    from codecarbon import OfflineEmissionsTracker

    return OfflineEmissionsTracker(
        country_iso_code=os.environ.get("PII_CC_COUNTRY", "ITA"),
        save_to_file=False,
        log_level="error",
    )


def _measure(
    label: str,
    detector: PIIDetector,
    corpus: Sequence[AnnotatedDocument],
    *,
    energy: bool,
) -> BenchmarkRow:
    """Score a detector over the corpus, timing it and (optionally) its energy.

    :param label: the row name.
    :param detector: the detector to score.
    :param corpus: the annotated documents.
    :param energy: track energy with `codecarbon` when ``True``.
    :returns: the measured :class:`BenchmarkRow` (per-document latency/energy).
    """
    tracker = _make_tracker() if energy else None
    if tracker is not None:
        tracker.start()  # type: ignore[attr-defined]
    start = time.perf_counter()
    report = evaluate(detector, corpus)
    elapsed = time.perf_counter() - start
    wh: float | None = None
    if tracker is not None:
        tracker.stop()  # type: ignore[attr-defined]
        data = tracker.final_emissions_data  # type: ignore[attr-defined]
        wh = data.energy_consumed * 1000.0 if data is not None else None
    n = max(1, len(corpus))
    return BenchmarkRow(label, report, elapsed / n, wh / n if wh is not None else None)


def run_benchmark(
    models: Sequence[str],
    corpus: Sequence[AnnotatedDocument],
    pattern: PIIDetector,
    ner: PIIDetector,
    ai_for: Callable[[str], PIIDetector],
    *,
    energy: bool = True,
) -> list[BenchmarkRow]:
    """Measure the baseline and, per model, the AI-alone and the merged-union rows.

    The base detectors are built once by the caller and reused across every row
    (DRY); ``ai_for`` builds a fresh AI detector per model. Injecting them keeps this
    function testable with fakes — no Presidio, Ollama or codecarbon needed.

    :param models: model names to benchmark, in the order to print them.
    :param corpus: the annotated documents to score on.
    :param pattern: the pattern/regex detector (shared across rows).
    :param ner: the NER detector (shared across rows).
    :param ai_for: builds the AI detector for a given model name.
    :param energy: track energy per row with `codecarbon`.
    :returns: the rows — baseline first, then ``ai:<m>`` and ``union:<m>`` per model.
    """
    rows = [
        _measure("presidio (pattern+ner)", MergedPipelineDetector(pattern, ner, None), corpus, energy=energy)
    ]
    for model in models:
        ai = ai_for(model)
        rows.append(_measure(f"ai:{model}", ai, corpus, energy=energy))
        rows.append(_measure(f"union:{model}", MergedPipelineDetector(pattern, ner, ai), corpus, energy=energy))
    return rows


def format_benchmark(rows: Sequence[BenchmarkRow]) -> str:
    """Render the rows as a fixed-width table: P / R / F1 / s/doc / Wh/doc.

    The rows are printed in the order given (baseline, then models by size), so the
    quality-vs-cost trade-off across the size tiers reads down the table.

    :param rows: the measured rows.
    :returns: a multi-line string ready to print.
    """
    header = f"{'row':<26}{'P':>6}{'R':>6}{'F1':>6}{'s/doc':>9}{'Wh/doc':>11}"
    lines = [header, "-" * len(header)]
    for row in rows:
        m = row.report.overall
        wh = f"{row.wh_per_doc:.4f}" if row.wh_per_doc is not None else "—"
        lines.append(
            f"{row.label:<26}{m.precision:>6.2f}{m.recall:>6.2f}{m.f1:>6.2f}"
            f"{row.seconds_per_doc:>9.3f}{wh:>11}"
        )
    return "\n".join(lines)


def _default_base(use_gliner: bool) -> tuple[PIIDetector, PIIDetector]:
    """Build the real Presidio detectors (lazy import: heavy stack)."""
    from pii_detection.detection.presidio_detector import build_default_detectors

    return build_default_detectors(use_gliner=use_gliner)


def _default_ai(model: str) -> PIIDetector:
    """Build the real AI detector for a model name (lazy import)."""
    from pii_detection.detection.ai_detector import build_ai_detector
    from pii_detection.llm.client import LLMClient

    return build_ai_detector(client=LLMClient(model=model))


def main(argv: list[str] | None = None) -> None:
    """CLI: build the detectors, run the benchmark, print the table.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Multi-model AI detection benchmark (B4).")
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS), help="Ollama model names to benchmark"
    )
    parser.add_argument("--corpus", type=Path, default=None, help="annotated corpus dir (default: packaged)")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N documents")
    parser.add_argument(
        "--gliner", action="store_true", help="use GLiNER for the NER (heavy; container only)"
    )
    parser.add_argument(
        "--no-energy", dest="energy", action="store_false", help="skip codecarbon energy tracking"
    )
    args = parser.parse_args(argv)

    corpus = load_corpus_dir(args.corpus)
    if args.limit is not None:
        corpus = corpus[: args.limit]
    pattern, ner = _default_base(args.gliner)
    rows = run_benchmark(args.models, corpus, pattern, ner, _default_ai, energy=args.energy)
    print(format_benchmark(rows))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_MODELS",
    "BenchmarkRow",
    "MergedPipelineDetector",
    "run_benchmark",
    "format_benchmark",
    "main",
]
