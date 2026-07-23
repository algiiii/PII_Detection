"""Tests for the ROPA category mapper (free text -> pii_type, deterministic)."""

from __future__ import annotations

import pytest

from pii_detection.detection.config import (
    ConfigError,
    PIICategoryCatalog,
    default_config_dir,
    load_category_catalog,
)
from pii_detection.ropa.ingestion.category_mapper import (
    CategoryMapper,
    DictionaryCategoryMapper,
    MappedCategory,
    build_dictionary_mapper,
)


def _catalog() -> PIICategoryCatalog:
    return load_category_catalog(default_config_dir() / "categories.yaml")


def test_dictionary_mapper_satisfies_protocol() -> None:
    assert isinstance(DictionaryCategoryMapper({}, _catalog()), CategoryMapper)


def test_resolves_known_phrase_keeping_original_text() -> None:
    mapper = DictionaryCategoryMapper({"bank account details": ["iban"]}, _catalog())
    assert mapper.map("Bank account details") == [
        MappedCategory("Bank account details", ("iban",))
    ]


def test_splits_on_separators_and_resolves_each_part() -> None:
    mapper = DictionaryCategoryMapper(
        {"coordinate bancarie": ["iban"], "contatti": ["email", "phone"]}, _catalog()
    )
    out = mapper.map("coordinate bancarie; contatti")
    assert [mc.text for mc in out] == ["coordinate bancarie", "contatti"]
    assert out[0].pii_types == ("iban",)
    assert set(out[1].pii_types) == {"email", "phone"}


def test_unknown_phrase_resolves_to_empty_types() -> None:
    mapper = DictionaryCategoryMapper({}, _catalog())
    assert mapper.map("mystery data") == [MappedCategory("mystery data", ())]


def test_rejects_pii_type_absent_from_catalog() -> None:
    with pytest.raises(ConfigError):
        DictionaryCategoryMapper({"whatever": ["not_a_type"]}, _catalog())


def test_shipped_config_builds_a_working_mapper() -> None:
    mapper = build_dictionary_mapper()
    assert mapper.map("Bank account details")[0].pii_types == ("iban",)
