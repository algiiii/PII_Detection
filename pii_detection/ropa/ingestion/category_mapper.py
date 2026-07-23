"""Resolve declared-category free text onto the ``pii_type`` catalog — block B1.

This is the interpretive step of the ingestion ("aggancio A"): a declared
category is written in prose (``"bank account details"``, ``"dati anagrafici,
coordinate bancarie"``) and must be split and resolved onto the canonical
``pii_type`` ids used by the detection layer (B4), so B7 can later compare what
the ROPA *declares* with what the engine *finds*.

The step is a **Strategy behind a Protocol** (:class:`CategoryMapper`): the
deterministic :class:`DictionaryCategoryMapper` implemented here and the future
LLM-backed mapper are interchangeable without touching the pipeline. Both operate
inside a **closed vocabulary**: every proposed ``pii_type`` is validated against
:class:`~pii_detection.detection.config.PIICategoryCatalog`, so a mapping can
never introduce an id the catalog does not declare. A phrase the mapper cannot
resolve yields an empty ``pii_types`` — the mapping stays ``PROPOSED`` for the
DPO to complete (human-in-the-loop), never silently dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from pii_detection.detection.config import (
    ConfigError,
    PIICategoryCatalog,
    default_config_dir,
    load_category_catalog,
)
from pii_detection.llm.client import LLMClient

#: Characters that separate the sub-categories inside one free-text cell.
_SEPARATORS = re.compile(r"[;,/]")


def _normalize(text: str) -> str:
    """Lowercase, strip and collapse inner whitespace for dictionary lookup.

    :param text: a raw phrase.
    :returns: its canonical lookup key, e.g. ``"Bank  Account "`` → ``"bank account"``.
    """
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class MappedCategory:
    """One sub-category split out of a declared-category cell, with its types.

    Immutable value object. ``text`` keeps the original phrase (for audit and for
    the DPO); ``pii_types`` is the resolved catalog ids, possibly empty when the
    phrase is not in the dictionary.

    :ivar text: the original sub-category phrase, e.g. ``"Bank account details"``.
    :ivar pii_types: catalog ids it resolves to, possibly empty.
    """

    text: str
    pii_types: tuple[str, ...]


@runtime_checkable
class CategoryMapper(Protocol):
    """Shape every category mapper must expose (Strategy contract).

    Structural typing: the ingestion depends only on ``map(raw_text) ->
    list[MappedCategory]``, so a dictionary mapper, an LLM mapper or a test fake
    are interchangeable without inheritance.
    """

    def map(self, raw_text: str) -> list[MappedCategory]:
        """Split a declared-category free text and resolve each part.

        :param raw_text: the free-text category cell, e.g.
            ``"coordinate bancarie, contatti"``.
        :returns: one :class:`MappedCategory` per sub-category, in order; a part
            that cannot be resolved carries an empty ``pii_types``.
        """
        ...


class CategoryMapFile(BaseModel):
    """Top-level schema of ``category_map.yaml``.

    :ivar mappings: phrase → list of ``pii_type`` ids the phrase resolves to.
    """

    model_config = ConfigDict(extra="forbid")

    mappings: dict[str, list[str]]


def load_category_map(path: Path, catalog: PIICategoryCatalog) -> dict[str, list[str]]:
    """Load and validate ``category_map.yaml`` against the catalog.

    :param path: path to the dictionary file.
    :param catalog: catalog every referenced ``pii_type`` must belong to.
    :returns: the phrase → ``pii_type`` ids table, as declared in the file.
    :raises ConfigError: on a missing/malformed file, or a ``pii_type`` that is
        not declared in the catalog.
    """
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc
    try:
        parsed = CategoryMapFile.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{exc}") from exc
    for phrase, pii_types in parsed.mappings.items():
        unknown = [t for t in pii_types if t not in catalog]
        if unknown:
            raise ConfigError(f"{path}: pii_type(s) not in catalog for {phrase!r}: {unknown}")
    return parsed.mappings


class DictionaryCategoryMapper:
    """Deterministic mapper backed by a phrase → ``pii_type`` dictionary.

    Splits the free text on ``,``/``;``/``/`` and looks up each part in the
    dictionary (case- and whitespace-insensitive). It is the default mapper: an
    LLM mapper will implement the same :class:`CategoryMapper` Protocol later.

    :ivar _table: normalized phrase → ``pii_type`` tuple (private).
    """

    def __init__(self, table: dict[str, list[str]], catalog: PIICategoryCatalog) -> None:
        """Build the mapper, validating every ``pii_type`` against the catalog.

        :param table: phrase → ``pii_type`` ids; phrases are normalized on
            insertion, so lookup is case- and whitespace-insensitive.
        :param catalog: catalog every ``pii_type`` must belong to.
        :raises ConfigError: if any ``pii_type`` is not declared in the catalog.
        """
        normalized: dict[str, tuple[str, ...]] = {}
        for phrase, pii_types in table.items():
            unknown = [t for t in pii_types if t not in catalog]
            if unknown:
                raise ConfigError(f"unknown pii_type(s) for {phrase!r}: {unknown}")
            normalized[_normalize(phrase)] = tuple(pii_types)
        self._table = normalized

    def map(self, raw_text: str) -> list[MappedCategory]:
        """Split ``raw_text`` on separators and resolve each part via the dictionary.

        :param raw_text: the free-text category cell.
        :returns: one :class:`MappedCategory` per non-empty part; an unknown part
            resolves to an empty ``pii_types`` tuple.
        """
        parts = [p.strip() for p in _SEPARATORS.split(raw_text) if p.strip()]
        return [MappedCategory(part, self._table.get(_normalize(part), ())) for part in parts]


_LLM_SYSTEM_PROMPT = (
    "You map declared data categories from a GDPR/nLPD processing register onto a "
    "fixed catalog of PII type ids. Be conservative and literal: never invent or infer "
    "sub-categories that are not explicitly written in the text, and assign only catalog "
    "ids that clearly correspond to what is written. When in doubt, return an empty list "
    "instead of guessing. Use only ids from the catalog given by the user. Answer with "
    "JSON only, no prose."
)

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class LLMCategoryMapper:
    """Category mapper backed by a local LLM, validated against the catalog.

    Same :class:`CategoryMapper` contract as :class:`DictionaryCategoryMapper`, so
    it is a drop-in replacement. The model is given the **closed** catalog and
    asked to split the free text and resolve each part; every returned id is
    validated against the catalog and unknown ones are dropped, so an
    hallucination can never reach the data model. On any failure (unreachable
    runtime, unparseable answer) it degrades to ``fallback`` — the AI is an
    enhancement, not a single point of failure.

    :ivar _client: the shared LLM client (private).
    :ivar _catalog: the catalog every returned id is validated against (private).
    :ivar _fallback: mapper used when the LLM fails, or ``None`` (private).
    """

    def __init__(
        self,
        client: LLMClient,
        catalog: PIICategoryCatalog,
        *,
        fallback: CategoryMapper | None = None,
    ) -> None:
        """Store the client, the validating catalog and the optional fallback.

        :param client: the shared LLM client.
        :param catalog: catalog every returned ``pii_type`` is validated against.
        :param fallback: mapper to use when the LLM call or its parsing fails;
            when ``None``, a failure yields the raw text with no ``pii_types``.
        """
        self._client = client
        self._catalog = catalog
        self._fallback = fallback

    def _prompt(self, raw_text: str) -> str:
        """Build the user prompt embedding the closed catalog and the free text."""
        catalog = "\n".join(f"- {c.id}: {c.label}" for c in self._catalog)
        return (
            f"Catalog of allowed pii_type ids:\n{catalog}\n\n"
            f"Declared category text: {raw_text!r}\n\n"
            "Split the text ONLY into the sub-categories that are verbatim substrings of "
            "it — do not add, infer, invent, or translate anything. For each sub-category, "
            "list only the catalog ids that clearly correspond; if none clearly applies, "
            "use an empty list. Reply as a JSON array of objects "
            '{"text": <verbatim substring of the category text>, "pii_types": [<id>, ...]}.'
        )

    def _parse(self, answer: str) -> list[MappedCategory]:
        """Parse the model's JSON answer, dropping ids absent from the catalog.

        :param answer: the raw text returned by the model.
        :returns: the resolved sub-categories.
        :raises ValueError: if no JSON array can be found in the answer.
        """
        match = _JSON_ARRAY.search(answer)
        if match is None:
            raise ValueError("no JSON array in LLM answer")
        result: list[MappedCategory] = []
        for item in json.loads(match.group(0)):
            pii_types = tuple(t for t in item.get("pii_types", []) if t in self._catalog)
            result.append(MappedCategory(str(item["text"]), pii_types))
        return result

    def map(self, raw_text: str) -> list[MappedCategory]:
        """Resolve a declared-category free text via the LLM, with a safe fallback.

        :param raw_text: the free-text category cell.
        :returns: one :class:`MappedCategory` per sub-category; on any LLM failure,
            the fallback's result, or the raw text with empty ``pii_types``.
        """
        try:
            return self._parse(self._client.complete(self._prompt(raw_text), system=_LLM_SYSTEM_PROMPT))
        except Exception:  # noqa: BLE001 — any LLM/parse failure degrades to the fallback
            if self._fallback is not None:
                return self._fallback.map(raw_text)
            return [MappedCategory(raw_text, ())]


def build_dictionary_mapper(config_dir: Path | None = None) -> DictionaryCategoryMapper:
    """Build a :class:`DictionaryCategoryMapper` from the packaged config.

    Loads ``categories.yaml`` (the catalog) and ``category_map.yaml`` (the
    dictionary) from the given directory.

    :param config_dir: directory holding the two YAML files; defaults to the
        packaged :func:`~pii_detection.detection.config.default_config_dir`.
    :returns: a mapper ready to resolve declared categories.
    :raises ConfigError: on any missing/malformed file or unknown ``pii_type``.
    """
    base = config_dir if config_dir is not None else default_config_dir()
    catalog = load_category_catalog(base / "categories.yaml")
    table = load_category_map(base / "category_map.yaml", catalog)
    return DictionaryCategoryMapper(table, catalog)


def build_llm_category_mapper(
    config_dir: Path | None = None, client: LLMClient | None = None
) -> LLMCategoryMapper:
    """Build an :class:`LLMCategoryMapper` with a deterministic dictionary fallback.

    :param config_dir: directory holding ``categories.yaml`` and
        ``category_map.yaml``; defaults to the packaged config.
    :param client: the LLM client to use; defaults to a fresh :class:`LLMClient`
        configured from the environment.
    :returns: an LLM mapper that falls back to the dictionary mapper on failure.
    :raises ConfigError: on any missing/malformed config file.
    """
    base = config_dir if config_dir is not None else default_config_dir()
    catalog = load_category_catalog(base / "categories.yaml")
    return LLMCategoryMapper(
        client if client is not None else LLMClient(),
        catalog,
        fallback=build_dictionary_mapper(config_dir),
    )


__all__ = [
    "MappedCategory",
    "CategoryMapper",
    "CategoryMapFile",
    "load_category_map",
    "DictionaryCategoryMapper",
    "build_dictionary_mapper",
    "LLMCategoryMapper",
    "build_llm_category_mapper",
]
