"""Tests for the post-ingestion mapping pass (split + resolve declared categories)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_detection.detection.config import default_config_dir, load_category_catalog
from pii_detection.ropa.ingestion.category_mapper import DictionaryCategoryMapper
from pii_detection.ropa.ingestion.pipeline import map_categories
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import (
    DeclaredCategory,
    DeclaredMacroCategory,
    MappingState,
    ProcessingActivity,
)


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'ropa.db'}"


def _seed(db_url: str, raw_text: str) -> None:
    ROPARepository(db_url).save(
        [
            ProcessingActivity(
                id="payroll",
                name="Payroll management",
                purpose="Administer the employment relationship",
                macro_categories=[
                    DeclaredMacroCategory(
                        raw_text="Economic and financial information",
                        retention_text="5 years",
                        retention_months=60,
                        categories=[DeclaredCategory(raw_text=raw_text, pii_types=[])],
                    )
                ],
            )
        ]
    )


def _mapper() -> DictionaryCategoryMapper:
    catalog = load_category_catalog(default_config_dir() / "categories.yaml")
    return DictionaryCategoryMapper(
        {"coordinate bancarie": ["iban"], "contatti": ["email", "phone"]}, catalog
    )


def test_split_category_replaces_one_with_many(db_url: str) -> None:
    _seed(db_url, "whatever")
    repo = ROPARepository(db_url)
    activity_id = repo.split_category(1, [("names", ["person_name"]), ("addresses", ["address"])])
    assert activity_id == "payroll"

    macro = repo.get("payroll").macro_categories[0]  # type: ignore[union-attr]
    assert {(c.raw_text, tuple(c.pii_types)) for c in macro.categories} == {
        ("names", ("person_name",)),
        ("addresses", ("address",)),
    }
    assert all(c.mapping_state is MappingState.PROPOSED for c in macro.categories)


def test_map_categories_splits_and_resolves(db_url: str) -> None:
    _seed(db_url, "coordinate bancarie; contatti")
    repo = ROPARepository(db_url)

    assert map_categories(repo, _mapper()) == 1

    macro = repo.get("payroll").macro_categories[0]  # type: ignore[union-attr]
    resolved = {c.raw_text: set(c.pii_types) for c in macro.categories}
    assert resolved == {"coordinate bancarie": {"iban"}, "contatti": {"email", "phone"}}


def test_map_categories_is_idempotent(db_url: str) -> None:
    _seed(db_url, "coordinate bancarie; contatti")
    repo = ROPARepository(db_url)

    assert map_categories(repo, _mapper()) == 1
    assert map_categories(repo, _mapper()) == 0  # nothing left to map on a second run


def test_map_categories_leaves_unresolved_single_phrase(db_url: str) -> None:
    _seed(db_url, "mystery data")
    repo = ROPARepository(db_url)

    assert map_categories(repo, _mapper()) == 0  # single unresolved phrase: untouched
    macro = repo.get("payroll").macro_categories[0]  # type: ignore[union-attr]
    assert [c.raw_text for c in macro.categories] == ["mystery data"]
    assert macro.categories[0].pii_types == []
