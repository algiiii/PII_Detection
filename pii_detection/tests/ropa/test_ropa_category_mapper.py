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


def test_splits_on_the_conjunction_e() -> None:
    mapper = DictionaryCategoryMapper(
        {"nome": ["person_name"], "cognome": ["person_name"]}, _catalog()
    )
    out = mapper.map("nome e cognome")
    assert [mc.text for mc in out] == ["nome", "cognome"]
    assert all(mc.pii_types == ("person_name",) for mc in out)


def test_keyword_spotting_matches_an_inner_key() -> None:
    mapper = DictionaryCategoryMapper({"carta di credito": ["credit_card"]}, _catalog())
    assert mapper.map("estremi della carta di credito") == [
        MappedCategory("estremi della carta di credito", ("credit_card",))
    ]


def test_longer_key_wins_over_shorter_overlapping_one() -> None:
    mapper = DictionaryCategoryMapper(
        {"indirizzo": ["address"], "indirizzo email": ["email"]}, _catalog()
    )
    # The longer "indirizzo email" consumes the tokens, so "indirizzo" -> address
    # does not also fire: the part resolves to email only.
    assert mapper.map("indirizzo email aziendale") == [
        MappedCategory("indirizzo email aziendale", ("email",))
    ]


def test_rejects_pii_type_absent_from_catalog() -> None:
    with pytest.raises(ConfigError):
        DictionaryCategoryMapper({"whatever": ["not_a_type"]}, _catalog())


def test_shipped_config_builds_a_working_mapper() -> None:
    mapper = build_dictionary_mapper()
    assert mapper.map("Bank account details")[0].pii_types == ("iban",)
