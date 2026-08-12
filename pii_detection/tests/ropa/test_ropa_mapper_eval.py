"""Tests for the category-mapper benchmark scoring (mapper faked, no LLM)."""

from __future__ import annotations

from pii_detection.detection.config import default_config_dir, load_category_catalog
from pii_detection.ropa.eval.mapper_eval import (
    MapperCase,
    default_cases_path,
    evaluate_mapper,
    load_cases,
)
from pii_detection.ropa.ingestion.category_mapper import MappedCategory


class _StubMapper:
    """A CategoryMapper returning canned pii_types per input text."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def map(self, raw_text: str) -> list[MappedCategory]:
        return [MappedCategory(raw_text, tuple(self._mapping.get(raw_text, [])))]


def test_shipped_cases_load_and_validate_against_catalog() -> None:
    catalog = load_category_catalog(default_config_dir() / "categories.yaml")
    cases = load_cases(default_cases_path(), catalog)
    assert len(cases) >= 10


def test_metric_counts_tp_fp_fn() -> None:
    cases = [
        MapperCase(text="a", expected=["iban"]),
        MapperCase(text="b", expected=["email", "phone"]),
    ]
    # "a": perfect; "b": one hit (email), one miss (phone), one spurious (iban).
    stub = _StubMapper({"a": ["iban"], "b": ["email", "iban"]})

    metrics = evaluate_mapper(stub, cases)

    assert (metrics.tp, metrics.fp, metrics.fn) == (2, 1, 1)
    assert metrics.precision == 2 / 3
    assert metrics.recall == 2 / 3


def test_perfect_mapper_scores_one() -> None:
    cases = [MapperCase(text="x", expected=["iban", "email"])]
    metrics = evaluate_mapper(_StubMapper({"x": ["iban", "email"]}), cases)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
