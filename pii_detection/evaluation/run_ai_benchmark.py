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
exploratory runs. Add ``--per-category`` to also print each row's per-``pii_type``
breakdown (via :func:`format_per_category`), which is what the per-detector and
whole-system tables of the thesis are filled from.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.types import DetectionProvenance, DetectorKind, PIICandidate
from pii_detection.evaluation.corpus import (
    AnnotatedDocument,
    load_annotated_corpus,
    sample_documents,
)
from pii_detection.evaluation.scoring import EvaluationReport, evaluate, format_report

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
    :ivar chunks_seen: chunks the generative detector submitted (0 for the baseline).
    :ivar chunks_failed: chunks it had to skip. A row measured while chunks were being
        dropped --- typically for lack of memory --- understates the model rather than
        describing it, so the count travels with the row instead of staying buried in
        the logs.
    """

    label: str
    report: EvaluationReport
    seconds_per_doc: float
    wh_per_doc: float | None
    chunks_seen: int = 0
    chunks_failed: int = 0


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
    ai: PIIDetector | None = None,
) -> BenchmarkRow:
    """Score a detector over the corpus, timing it and (optionally) its energy.

    :param label: the row name.
    :param detector: the detector to score.
    :param corpus: the annotated documents.
    :param energy: track energy with `codecarbon` when ``True``.
    :param ai: the generative detector behind this row, when there is one; its
        chunk counters are read after scoring so the row can report what it skipped.
    :returns: the measured :class:`BenchmarkRow` (per-document latency/energy).
    """
    seen_before = getattr(ai, "chunks_seen", 0)
    failed_before = getattr(ai, "chunks_failed", 0)
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
    return BenchmarkRow(
        label,
        report,
        elapsed / n,
        wh / n if wh is not None else None,
        getattr(ai, "chunks_seen", 0) - seen_before,
        getattr(ai, "chunks_failed", 0) - failed_before,
    )


def run_benchmark(
    models: Sequence[str],
    corpus: Sequence[AnnotatedDocument],
    pattern: PIIDetector | None,
    ner: PIIDetector | None,
    ai_for: Callable[[str], PIIDetector],
    *,
    energy: bool = True,
) -> list[BenchmarkRow]:
    """Measure the baseline and, per model, the AI-alone and the merged-union rows.

    The base detectors are built once by the caller and reused across every row
    (DRY); ``ai_for`` builds a fresh AI detector per model. Injecting them keeps this
    function testable with fakes — no Presidio, Ollama or codecarbon needed.

    Passing ``None`` for the base detectors measures the ``ai:`` rows **alone**,
    dropping the baseline and union rows that need them. The point is memory, not
    tidiness: an ``ai:`` row never consults the NER, yet loading the NER encoder
    keeps a couple of gigabytes resident beside the language model. Where the two
    together exceed what the machine can back with physical memory, the model runner
    is killed mid-scan and the row silently measures a fraction of the corpus — so
    dropping what that row does not use is what makes it measurable at all.

    :param models: model names to benchmark, in the order to print them.
    :param corpus: the annotated documents to score on.
    :param pattern: the pattern/regex detector, or ``None`` for AI-only rows.
    :param ner: the NER detector, or ``None`` for AI-only rows.
    :param ai_for: builds the AI detector for a given model name.
    :param energy: track energy per row with `codecarbon`.
    :returns: the rows — baseline first (unless AI-only), then ``ai:<m>`` and
        ``union:<m>`` per model (the latter only when the base detectors are given).
    """
    ai_only = pattern is None or ner is None
    rows: list[BenchmarkRow] = []
    if not ai_only:
        assert pattern is not None and ner is not None  # narrowed by ai_only
        rows.append(
            _measure(
                "presidio (pattern+ner)",
                MergedPipelineDetector(pattern, ner, None),
                corpus,
                energy=energy,
            )
        )
    for model in models:
        ai = ai_for(model)
        rows.append(_measure(f"ai:{model}", ai, corpus, energy=energy, ai=ai))
        if not ai_only:
            assert pattern is not None and ner is not None
            rows.append(
                _measure(
                    f"union:{model}",
                    MergedPipelineDetector(pattern, ner, ai),
                    corpus,
                    energy=energy,
                    ai=ai,
                )
            )
    return rows


def format_benchmark(rows: Sequence[BenchmarkRow]) -> str:
    """Render the rows as a fixed-width table: P / R / F1 / s/doc / Wh/doc.

    The rows are printed in the order given (baseline, then models by size), so the
    quality-vs-cost trade-off across the size tiers reads down the table.

    :param rows: the measured rows.
    :returns: a multi-line string ready to print.
    """
    header = (
        f"{'row':<26}{'P':>6}{'R':>6}{'F1':>6}{'s/doc':>9}{'Wh/doc':>11}{'skipped':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        m = row.report.overall
        wh = f"{row.wh_per_doc:.4f}" if row.wh_per_doc is not None else "—"
        skipped = f"{row.chunks_failed}/{row.chunks_seen}" if row.chunks_seen else "—"
        lines.append(
            f"{row.label:<26}{m.precision:>6.2f}{m.recall:>6.2f}{m.f1:>6.2f}"
            f"{row.seconds_per_doc:>9.3f}{wh:>11}{skipped:>9}"
        )
    damaged = [r for r in rows if r.chunks_seen and r.chunks_failed]
    if damaged:
        lines.append("")
        lines.append("WARNING: chunks were skipped — these rows understate their model,")
        lines.append("         they do not measure it. Re-run before quoting them:")
        for row in damaged:
            share = 100.0 * row.chunks_failed / row.chunks_seen
            lines.append(
                f"         {row.label}: {row.chunks_failed}/{row.chunks_seen} chunks ({share:.0f}%)"
            )
    return "\n".join(lines)


def format_per_category(rows: Sequence[BenchmarkRow]) -> str:
    """Render each row's per-category breakdown (P/R/F1/tp/fp/fn), row by row.

    The compact table from :func:`format_benchmark` shows only the micro-averaged
    line; this expands every row into the full per-``pii_type`` report of
    :func:`~pii_detection.evaluation.scoring.format_report`, which is what the
    per-detector and whole-system tables of the thesis need (a category left
    uncovered is a defect even when the overall figure is good).

    :param rows: the measured rows (baseline, then ``ai:<m>`` / ``union:<m>``).
    :returns: a multi-line string — one labelled report block per row.
    """
    blocks: list[str] = []
    for row in rows:
        blocks.append(f"=== {row.label} ===")
        blocks.append(format_report(row.report))
    return "\n\n".join(blocks)


def _default_base(use_gliner: bool) -> tuple[PIIDetector, PIIDetector]:
    """Build the real Presidio detectors (lazy import: heavy stack)."""
    from pii_detection.detection.presidio_detector import build_default_detectors

    return build_default_detectors(use_gliner=use_gliner)


def _default_ai(model: str) -> PIIDetector:
    """Build the real AI detector for a model name (lazy import).

    The client carries the same generated-token cap
    (:func:`~pii_detection.detection.ai_detector.resolve_num_predict`) the app
    uses, so a rambling model cannot dominate the measured latency/energy and the
    benchmark reflects the deployed configuration.
    """
    from pii_detection.detection.ai_detector import build_ai_detector, resolve_num_predict
    from pii_detection.llm.client import LLMClient

    return build_ai_detector(client=LLMClient(model=model, num_predict=resolve_num_predict()))


def main(argv: list[str] | None = None) -> None:
    """CLI: build the detectors, run the benchmark, print the table.

    :param argv: argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Multi-model AI detection benchmark (B4).")
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS), help="Ollama model names to benchmark"
    )
    parser.add_argument("--corpus", type=Path, default=None, help="annotated corpus: a dir of .txt or a sources.jsonl (default: packaged)")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N documents")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="score a random subset of N documents (preferred over --limit: an ordered "
        "corpus's head is not representative); reproducible via --seed",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed fixing the --sample draw")
    parser.add_argument(
        "--gliner", action="store_true", help="use GLiNER for the NER (heavy; container only)"
    )
    parser.add_argument(
        "--ai-only",
        action="store_true",
        help="measure only the ai:<model> rows, without building the traditional "
        "detectors — frees the memory the NER encoder would hold beside the language "
        "model, which an ai: row never uses (drops the baseline and union rows)",
    )
    parser.add_argument(
        "--no-energy", dest="energy", action="store_false", help="skip codecarbon energy tracking"
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="also print each row's per-category P/R/F1 (for the per-detector tables)",
    )
    args = parser.parse_args(argv)

    energy = args.energy
    if energy:
        try:
            import codecarbon  # noqa: F401  (availability probe only)
        except ImportError:
            print(
                "warning: codecarbon not installed — running without energy tracking "
                "(Wh/doc will show '—'). Install '.[eval]' to enable it.",
                file=sys.stderr,
            )
            energy = False

    corpus = load_annotated_corpus(args.corpus)
    if args.sample is not None:
        corpus = sample_documents(corpus, args.sample, args.seed)
    elif args.limit is not None:
        corpus = corpus[: args.limit]
    occurrences = sum(len(document.spans) for document in corpus)
    print(
        f"corpus: {len(corpus)} documents, {occurrences} annotated PII occurrences",
        file=sys.stderr,
    )
    if args.ai_only:
        pattern, ner = None, None
    else:
        pattern, ner = _default_base(args.gliner)
    rows = run_benchmark(args.models, corpus, pattern, ner, _default_ai, energy=energy)
    print(format_benchmark(rows))
    if args.per_category:
        print()
        print(format_per_category(rows))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_MODELS",
    "BenchmarkRow",
    "MergedPipelineDetector",
    "run_benchmark",
    "format_benchmark",
    "format_per_category",
    "main",
]
