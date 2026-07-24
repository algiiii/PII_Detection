"""Run the category-mapper benchmark: dictionary vs local LLM.

Usage (requires a running Ollama for the LLM row)::

    python -m pii_detection.ropa.eval

Prints micro precision/recall/F1 for the deterministic dictionary mapper and for
the LLM mapper (no fallback, so the row reflects the model alone), plus the wall
time per case — the numbers that back the "one model vs one per task" decision in
``doc/plans/ai-assessment.md``.
"""

from __future__ import annotations

import time

from pii_detection.detection.config import default_config_dir, load_category_catalog
from pii_detection.llm.client import LLMClient
from pii_detection.ropa.eval.mapper_eval import (
    MapperCase,
    MapperMetrics,
    default_cases_path,
    evaluate_mapper,
    load_cases,
)
from pii_detection.ropa.ingestion.category_mapper import (
    CategoryMapper,
    LLMCategoryMapper,
    build_dictionary_mapper,
)


def _format_row(name: str, metrics: MapperMetrics, seconds: float, n: int) -> str:
    """Format one result line with scores, counts and per-case latency."""
    return (
        f"{name:11}  P={metrics.precision:.2f}  R={metrics.recall:.2f}  F1={metrics.f1:.2f}"
        f"  (tp={metrics.tp} fp={metrics.fp} fn={metrics.fn})"
        f"  {seconds / n * 1000:6.0f} ms/case"
    )


def _timed(mapper: CategoryMapper, cases: list[MapperCase]) -> tuple[MapperMetrics, float]:
    """Evaluate a mapper and return its metrics with the elapsed wall time."""
    start = time.perf_counter()
    metrics = evaluate_mapper(mapper, cases)
    return metrics, time.perf_counter() - start


def main() -> None:
    """Load the cases and print the dictionary and LLM benchmark rows."""
    catalog = load_category_catalog(default_config_dir() / "categories.yaml")
    cases = load_cases(default_cases_path(), catalog)
    n = len(cases)

    dict_metrics, dict_time = _timed(build_dictionary_mapper(), cases)
    llm_metrics, llm_time = _timed(LLMCategoryMapper(LLMClient(), catalog), cases)

    print(f"cases: {n}")
    print(_format_row("dictionary", dict_metrics, dict_time, n))
    print(_format_row("llm", llm_metrics, llm_time, n))


if __name__ == "__main__":
    main()
