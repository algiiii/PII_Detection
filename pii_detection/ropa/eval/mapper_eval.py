"""Benchmark harness for category mappers — measure, don't eyeball (block B1).

Scores a :class:`~pii_detection.ropa.ingestion.category_mapper.CategoryMapper`
against a small annotated set of category free texts, as micro
precision/recall/F1 over the resolved ``pii_type`` sets. It lets us put numbers on
"dictionary vs LLM" and on any prompt/temperature/model change, instead of judging
by eye (see ``doc/plans/ai-assessment.md``).

For each case the metric compares the **union** of the ``pii_type`` ids the mapper
produces for the text against the expected set: true positives are the overlap,
false positives what the mapper added, false negatives what it missed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from pii_detection.detection.config import ConfigError, PIICategoryCatalog
from pii_detection.ropa.ingestion.category_mapper import CategoryMapper


class MapperCase(BaseModel):
    """One annotated case: a category free text and its expected ``pii_type`` set.

    :ivar text: the declared-category free text fed to the mapper.
    :ivar expected: the ``pii_type`` ids the text should resolve to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    expected: list[str]


class MapperCasesFile(BaseModel):
    """Top-level schema of ``category_cases.yaml``.

    :ivar cases: the annotated cases.
    """

    model_config = ConfigDict(extra="forbid")

    cases: list[MapperCase]


def default_cases_path() -> Path:
    """Locate the annotated dataset shipped with the package.

    :returns: absolute path to ``category_cases.yaml``.
    """
    return Path(__file__).resolve().parent / "category_cases.yaml"


def load_cases(path: Path, catalog: PIICategoryCatalog | None = None) -> list[MapperCase]:
    """Load and validate the annotated cases.

    :param path: path to the cases file.
    :param catalog: when given, every expected ``pii_type`` is checked against it,
        so a typo in the ground truth fails loudly instead of skewing the metric.
    :returns: the annotated cases, in file order.
    :raises ConfigError: on a missing/malformed file, or an expected ``pii_type``
        absent from the catalog.
    """
    if not path.is_file():
        raise ConfigError(f"cases file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc
    try:
        parsed = MapperCasesFile.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid cases in {path}:\n{exc}") from exc
    if catalog is not None:
        unknown = sorted({t for case in parsed.cases for t in case.expected if t not in catalog})
        if unknown:
            raise ConfigError(f"{path}: expected pii_type(s) not in catalog: {unknown}")
    return parsed.cases


@dataclass(frozen=True)
class MapperMetrics:
    """Micro-averaged evaluation counts and their derived scores.

    :ivar tp: true positives (correctly resolved ``pii_type`` ids).
    :ivar fp: false positives (ids the mapper added but were not expected).
    :ivar fn: false negatives (expected ids the mapper missed).
    """

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        """:returns: ``tp / (tp + fp)``, or ``0.0`` when nothing was predicted."""
        predicted = self.tp + self.fp
        return self.tp / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """:returns: ``tp / (tp + fn)``, or ``0.0`` when nothing was expected."""
        actual = self.tp + self.fn
        return self.tp / actual if actual else 0.0

    @property
    def f1(self) -> float:
        """:returns: the harmonic mean of :attr:`precision` and :attr:`recall`."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_mapper(mapper: CategoryMapper, cases: list[MapperCase]) -> MapperMetrics:
    """Score a mapper over the annotated cases.

    :param mapper: the mapper under test.
    :param cases: the annotated ground-truth cases.
    :returns: the micro-averaged :class:`MapperMetrics` across all cases.
    """
    tp = fp = fn = 0
    for case in cases:
        predicted = {pii_type for mapped in mapper.map(case.text) for pii_type in mapped.pii_types}
        expected = set(case.expected)
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    return MapperMetrics(tp, fp, fn)


__all__ = [
    "MapperCase",
    "MapperCasesFile",
    "default_cases_path",
    "load_cases",
    "MapperMetrics",
    "evaluate_mapper",
]
