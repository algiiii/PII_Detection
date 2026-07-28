"""Configuration schema and loading for the detection layer (block B4, Step 3).

Defined in ``doc/scaffolding-plan.md`` §"Step 3 — Schema e caricamento config".

This module makes the **extensibility requirement (§2.3.10)** real: PII
categories, regex rules and NER labels are declared in YAML and validated at
startup, so a non-developer operator adds a category by editing a file, without
touching Python.

Four declarative files feed the layer:

- ``categories.yaml`` — the canonical catalog of ``pii_type`` ids
  (:class:`PIICategoryCatalog`).
- ``regex_rules.yaml`` — rules for the ``RegexDetector`` (Step 4).
- ``ner_labels.yaml`` — label→``pii_type`` mapping consumed by the GLiNER
  recognizer inside Presidio.
- ``presidio_entities.yaml`` — Presidio ``entity_type``→``pii_type`` mapping for
  the Presidio detector.

The models here validate **structure and self-consistency** only (well-formed
YAML, value ranges, compilable patterns, and every referenced ``pii_type``
existing in the catalog). Turning a validated rule into a runtime detector
belongs to the later steps that consume this configuration.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed or inconsistent.

    Wraps the lower-level failures (missing file, YAML syntax error, Pydantic
    :class:`~pydantic.ValidationError`, orphan ``pii_type``) into a single
    startup-facing error with a readable message, so a config mistake fails
    loudly at load time rather than silently at detection time.
    """


_RE_FLAG_BY_NAME: dict[str, re.RegexFlag] = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "UNICODE": re.UNICODE,
    "VERBOSE": re.VERBOSE,
    "ASCII": re.ASCII,
}


class PIICategoryModel(BaseModel):
    """One entry of the PII category catalog, as declared in ``categories.yaml``.

    Immutable and closed to unknown keys (``extra="forbid"``) so a typo in the
    YAML fails at startup instead of being silently ignored.

    :ivar id: canonical, flat and stable identifier of the category, e.g.
        ``"iban"``; it is the value used as ``pii_type`` everywhere else.
    :ivar label: human-readable name for reports and the DPO-facing UI.
    :ivar frameworks: regulatory frameworks under which the category is
        relevant, e.g. ``("gdpr", "nlpd")``; free-form metadata, not validated
        against a closed list.
    :ivar special_category: ``True`` for special categories of data (e.g. health,
        GDPR art. 9 / nLPD art. 5), which downstream layers may treat with
        stricter rules.
    :ivar description: optional free-text note on the category.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    label: str
    frameworks: tuple[str, ...] = ()
    special_category: bool = False
    description: str | None = None


class CategoriesFile(BaseModel):
    """Top-level schema of ``categories.yaml``.

    :ivar categories: declared PII categories; the whole catalog.
    """

    model_config = ConfigDict(extra="forbid")

    categories: list[PIICategoryModel]


class RegexRuleModel(BaseModel):
    """One regex rule as declared in ``regex_rules.yaml``.

    Mirrors the runtime ``RegexRule`` of the design (``doc/planning.md``) but is
    the *config-layer* representation: validated here, converted to a runtime
    detector at Step 4.

    .. note::
        Checksum validation of structured identifiers (IBAN, credit card,
        AVS...) is intentionally deferred — see ``doc/sviluppi-futuri.md``.

    :ivar rule_id: identifier of the rule, unique within the file, e.g.
        ``"iban_v1"``.
    :ivar pii_type: category produced by the rule; must exist in the catalog.
    :ivar pattern: regex source; validated to be compilable at load time.
    :ivar flags: names of the :mod:`re` flags to compile the pattern with,
        e.g. ``("UNICODE", "IGNORECASE")``; defaults to ``("UNICODE",)``.
    :ivar base_confidence: confidence assigned to a match, in ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    pii_type: str
    pattern: str
    flags: tuple[str, ...] = ("UNICODE",)
    base_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("flags")
    @classmethod
    def _known_flags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject unknown :mod:`re` flag names.

        :param value: flag names read from YAML.
        :returns: the same tuple if all names are known.
        :raises ValueError: if any name is not a supported :mod:`re` flag.
        """
        unknown = [name for name in value if name not in _RE_FLAG_BY_NAME]
        if unknown:
            known = ", ".join(sorted(_RE_FLAG_BY_NAME))
            raise ValueError(f"unknown regex flag(s) {unknown}; supported: {known}")
        return value

    @property
    def re_flags(self) -> int:
        """Combine :attr:`flags` into a single :mod:`re` flag mask.

        :returns: the bitwise-or of the named flags (``0`` if none).
        """
        result = 0
        for name in self.flags:
            result |= _RE_FLAG_BY_NAME[name]
        return result

    @model_validator(mode="after")
    def _pattern_compiles(self) -> RegexRuleModel:
        """Ensure the pattern compiles with the resolved flags.

        :returns: the validated model.
        :raises ValueError: if :attr:`pattern` is not a valid regex.
        """
        try:
            re.compile(self.pattern, self.re_flags)
        except re.error as exc:
            raise ValueError(f"invalid regex for rule {self.rule_id!r}: {exc}") from exc
        return self


class RegexRulesFile(BaseModel):
    """Top-level schema of ``regex_rules.yaml``.

    :ivar rules: declared regex rules.
    """

    model_config = ConfigDict(extra="forbid")

    rules: list[RegexRuleModel]


class NerLabelModel(BaseModel):
    """One NER label→category mapping as declared in ``ner_labels.yaml``.

    :ivar label: textual label passed to the zero-shot NER model (GLiNER),
        e.g. ``"person"``; unique within the file.
    :ivar pii_type: category the label maps to; must exist in the catalog.
    :ivar threshold: minimum model score to keep a span for this label, in
        ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    pii_type: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class NerLabelsFile(BaseModel):
    """Top-level schema of ``ner_labels.yaml``.

    :ivar labels: declared label→category mappings.
    """

    model_config = ConfigDict(extra="forbid")

    labels: list[NerLabelModel]


class PresidioEntityModel(BaseModel):
    """One Presidio ``entity_type``- category mapping (``presidio_entities.yaml``).

    :ivar entity: Presidio ``entity_type`` produced by a recognizer, e.g.
        ``"IT_FISCAL_CODE"``; unique within the file.
    :ivar pii_type: category the entity maps to; must exist in the catalog.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity: str
    pii_type: str


class PresidioEntitiesFile(BaseModel):
    """Top-level schema of ``presidio_entities.yaml``.

    :ivar entities: declared entity→category mappings.
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[PresidioEntityModel]


class PIICategoryCatalog:
    """In-memory catalog of the declared PII categories, indexed by id.

    Built from ``categories.yaml``. It is the authority every other config file
    is validated against: a ``pii_type`` referenced by a rule or a label is
    legitimate only if it appears here.

    :ivar _by_id: internal id→category map (private).
    """

    def __init__(self, categories: Iterable[PIICategoryModel]) -> None:
        """Index the categories, rejecting duplicate ids.

        :param categories: category models loaded from the file.
        :raises ConfigError: if two categories share the same ``id``.
        """
        by_id: dict[str, PIICategoryModel] = {}
        for category in categories:
            if category.id in by_id:
                raise ConfigError(f"duplicate category id: {category.id!r}")
            by_id[category.id] = category
        self._by_id = by_id

    def __contains__(self, pii_type: object) -> bool:
        """:returns: ``True`` if ``pii_type`` is a declared category id."""
        return pii_type in self._by_id

    def __iter__(self) -> Iterator[PIICategoryModel]:
        """:returns: iterator over the categories, in declaration order."""
        return iter(self._by_id.values())

    def __len__(self) -> int:
        """:returns: number of declared categories."""
        return len(self._by_id)

    def get(self, pii_type: str) -> PIICategoryModel | None:
        """Look up a category without raising.

        :param pii_type: category id to resolve.
        :returns: the category, or ``None`` if it is not declared.
        """
        return self._by_id.get(pii_type)

    def require(self, pii_type: str) -> PIICategoryModel:
        """Look up a category, failing if it is unknown.

        :param pii_type: category id to resolve.
        :returns: the corresponding category.
        :raises ConfigError: if ``pii_type`` is not declared in the catalog.
        """
        try:
            return self._by_id[pii_type]
        except KeyError:
            raise ConfigError(f"unknown pii_type: {pii_type!r}") from None


@dataclass(frozen=True)
class DetectionConfig:
    """Fully validated configuration of the detection layer.

    Result of :func:`load_detection_config`: the three files parsed, and every
    ``pii_type`` cross-checked against the catalog. It is the single object the
    detector factory (Step 8) consumes.

    :ivar catalog: the PII category catalog.
    :ivar regex_rules: validated regex rules (referenced types exist in the
        catalog, ids unique).
    :ivar ner_labels: validated NER label mappings (referenced types exist in
        the catalog, labels unique).
    """

    catalog: PIICategoryCatalog
    regex_rules: tuple[RegexRuleModel, ...]
    ner_labels: tuple[NerLabelModel, ...]


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _parse_file(path: Path, model: type[_ModelT]) -> _ModelT:
    """Load a YAML file and validate it against a Pydantic model.

    :param path: file to read.
    :param model: schema to validate the parsed content against.
    :returns: the validated model instance.
    :raises ConfigError: if the file is missing, empty, not valid YAML, or does
        not satisfy the schema.
    """
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"empty config file: {path}")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{exc}") from exc


def _require_types_in_catalog(
    pii_types: Iterable[str], catalog: PIICategoryCatalog, path: Path
) -> None:
    """Fail if any referenced ``pii_type`` is absent from the catalog.

    :param pii_types: category ids referenced by a config file.
    :param catalog: the authoritative catalog.
    :param path: file the references come from (for the error message).
    :raises ConfigError: listing every orphan ``pii_type``.
    """
    orphans = sorted({t for t in pii_types if t not in catalog})
    if orphans:
        raise ConfigError(f"{path}: pii_type(s) not in category catalog: {orphans}")


def _require_unique(values: Iterable[str], path: Path, what: str) -> None:
    """Fail if a supposedly unique identifier repeats.

    :param values: identifiers that must be unique.
    :param path: file the identifiers come from (for the error message).
    :param what: human name of the identifier, e.g. ``"rule_id"``.
    :raises ConfigError: listing every duplicated value.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ConfigError(f"{path}: duplicate {what}: {sorted(duplicates)}")


def load_category_catalog(path: Path) -> PIICategoryCatalog:
    """Load and index ``categories.yaml``.

    :param path: path to the categories file.
    :returns: the built catalog.
    :raises ConfigError: on a missing/malformed file or duplicate category id.
    """
    parsed = _parse_file(path, CategoriesFile)
    return PIICategoryCatalog(parsed.categories)


def load_regex_rules(path: Path, catalog: PIICategoryCatalog) -> tuple[RegexRuleModel, ...]:
    """Load ``regex_rules.yaml`` and cross-validate it against the catalog.

    :param path: path to the regex rules file.
    :param catalog: catalog every ``pii_type`` must belong to.
    :returns: the validated rules, in declaration order.
    :raises ConfigError: on a missing/malformed file, a duplicate ``rule_id`` or
        a ``pii_type`` absent from the catalog.
    """
    parsed = _parse_file(path, RegexRulesFile)
    _require_unique((r.rule_id for r in parsed.rules), path, "rule_id")
    _require_types_in_catalog((r.pii_type for r in parsed.rules), catalog, path)
    return tuple(parsed.rules)


def load_ner_labels(path: Path, catalog: PIICategoryCatalog) -> tuple[NerLabelModel, ...]:
    """Load ``ner_labels.yaml`` and cross-validate it against the catalog.

    :param path: path to the NER labels file.
    :param catalog: catalog every ``pii_type`` must belong to.
    :returns: the validated label mappings, in declaration order.
    :raises ConfigError: on a missing/malformed file, a duplicate ``label`` or a
        ``pii_type`` absent from the catalog.
    """
    parsed = _parse_file(path, NerLabelsFile)
    _require_unique((label.label for label in parsed.labels), path, "label")
    _require_types_in_catalog((label.pii_type for label in parsed.labels), catalog, path)
    return tuple(parsed.labels)


def load_presidio_entities(
    path: Path, catalog: PIICategoryCatalog
) -> tuple[PresidioEntityModel, ...]:
    """Load ``presidio_entities.yaml`` and cross-validate it against the catalog.

    :param path: path to the Presidio entities file.
    :param catalog: catalog every ``pii_type`` must belong to.
    :returns: the validated entity mappings, in declaration order.
    :raises ConfigError: on a missing/malformed file, a duplicate ``entity`` or a
        ``pii_type`` absent from the catalog.
    """
    parsed = _parse_file(path, PresidioEntitiesFile)
    _require_unique((e.entity for e in parsed.entities), path, "entity")
    _require_types_in_catalog((e.pii_type for e in parsed.entities), catalog, path)
    return tuple(parsed.entities)


def default_config_dir() -> Path:
    """Locate the ``config`` directory shipped with the package.

    :returns: absolute path to ``pii_detection/config``.
    """
    return Path(__file__).resolve().parent.parent / "config"


def load_detection_config(config_dir: Path | None = None) -> DetectionConfig:
    """Load and cross-validate the whole detection configuration.

    Loads the three files, then guarantees every ``pii_type`` referenced by a
    rule or a label exists in the catalog. This is the startup gate that makes
    "a new category is one YAML line, zero code" (§2.3.10) safe.

    :param config_dir: directory containing the three YAML files; defaults to
        the packaged :func:`default_config_dir`.
    :returns: the fully validated :class:`DetectionConfig`.
    :raises ConfigError: on any missing/malformed file or inconsistency.
    """
    base = config_dir if config_dir is not None else default_config_dir()
    catalog = load_category_catalog(base / "categories.yaml")
    regex_rules = load_regex_rules(base / "regex_rules.yaml", catalog)
    ner_labels = load_ner_labels(base / "ner_labels.yaml", catalog)
    return DetectionConfig(catalog=catalog, regex_rules=regex_rules, ner_labels=ner_labels)


__all__ = [
    "ConfigError",
    "PIICategoryModel",
    "CategoriesFile",
    "RegexRuleModel",
    "RegexRulesFile",
    "NerLabelModel",
    "NerLabelsFile",
    "PresidioEntityModel",
    "PresidioEntitiesFile",
    "PIICategoryCatalog",
    "DetectionConfig",
    "load_category_catalog",
    "load_regex_rules",
    "load_ner_labels",
    "load_presidio_entities",
    "load_detection_config",
    "default_config_dir",
]
