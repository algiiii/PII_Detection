"""Tests for proposing an orphan ``pii_type`` as a declared category (B7 → B1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_detection.detection.config import default_config_dir, load_category_catalog
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


def _seed(db_url: str) -> ROPARepository:
    repo = ROPARepository(db_url)
    repo.save(
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
                        categories=[DeclaredCategory(raw_text="salary", pii_types=[])],
                    )
                ],
            )
        ]
    )
    return repo


def _proposed(repo: ROPARepository, pii_type: str) -> list[DeclaredCategory]:
    activity = repo.get("payroll")
    assert activity is not None
    return [c for m in activity.macro_categories for c in m.categories if pii_type in c.pii_types]


def test_propose_adds_a_proposed_category_with_the_catalog_label(db_url: str) -> None:
    repo = _seed(db_url)
    repo.propose_category("payroll", "iban")

    label = load_category_catalog(default_config_dir() / "categories.yaml").require("iban").label
    (proposed,) = _proposed(repo, "iban")
    assert proposed.pii_types == ["iban"]
    assert proposed.mapping_state is MappingState.PROPOSED  # a proposal, not confirmed
    assert proposed.raw_text == label


def test_propose_has_no_retention(db_url: str) -> None:
    """A proposal must not invent a retention the system does not know."""
    repo = _seed(db_url)
    repo.propose_category("payroll", "iban")
    activity = repo.get("payroll")
    assert activity is not None
    proposals_macro = next(
        m for m in activity.macro_categories if any("iban" in c.pii_types for c in m.categories)
    )
    assert proposals_macro.retention_months is None


def test_propose_is_idempotent(db_url: str) -> None:
    repo = _seed(db_url)
    repo.propose_category("payroll", "iban")
    repo.propose_category("payroll", "iban")
    assert len(_proposed(repo, "iban")) == 1  # proposed once, not twice


def test_propose_unknown_pii_type_raises(db_url: str) -> None:
    repo = _seed(db_url)
    with pytest.raises(ValueError):
        repo.propose_category("payroll", "not_a_type")


def test_propose_unknown_activity_raises(db_url: str) -> None:
    repo = ROPARepository(db_url)  # empty register
    with pytest.raises(KeyError):
        repo.propose_category("ghost", "iban")
