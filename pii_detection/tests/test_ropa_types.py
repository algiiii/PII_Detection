"""Tests for the ROPA data model (:mod:`pii_detection.ropa.types`)."""

from __future__ import annotations

import pytest

from pii_detection.ropa.types import (
    DeclaredDataCategory,
    MappingState,
    ProcessingActivity,
    Retention,
    ROPA,
)


class TestDeclaredDataCategory:
    """Shape of :class:`DeclaredDataCategory`."""

    def test_maps_one_to_many(self) -> None:
        category = DeclaredDataCategory(
            raw_text="dati anagrafici",
            pii_types=("person_name", "date_of_birth", "italian_id"),
        )
        assert category.pii_types == ("person_name", "date_of_birth", "italian_id")
        assert category.mapping_state is MappingState.PROPOSED

    def test_unmapped_is_allowed(self) -> None:
        assert DeclaredDataCategory(raw_text="varie").pii_types == ()


class TestRetention:
    """Validation of :class:`Retention`."""

    def test_computable_duration(self) -> None:
        retention = Retention(raw_text="5 anni dalla cessazione", duration_months=60)
        assert retention.duration_months == 60

    def test_criterion_without_duration(self) -> None:
        assert Retention(raw_text="finché richiesto").duration_months is None

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative duration_months"):
            Retention(raw_text="x", duration_months=-1)


class TestProcessingActivity:
    """Aggregation on :class:`ProcessingActivity`."""

    def _activity(self) -> ProcessingActivity:
        return ProcessingActivity(
            activity_id="hr-01",
            name="Gestione del personale",
            purpose="Amministrazione del rapporto di lavoro",
            legal_basis="Contratto",
            controller="ACME S.p.A.",
            data_categories=[
                DeclaredDataCategory(
                    raw_text="dati anagrafici",
                    pii_types=("person_name", "italian_id"),
                    mapping_state=MappingState.CONFIRMED,
                ),
                DeclaredDataCategory(
                    raw_text="coordinate bancarie",
                    pii_types=("iban",),
                    mapping_state=MappingState.PROPOSED,
                ),
            ],
        )

    def test_declared_pii_types_union(self) -> None:
        assert self._activity().declared_pii_types() == {"person_name", "italian_id", "iban"}

    def test_declared_pii_types_confirmed_only(self) -> None:
        assert self._activity().declared_pii_types(confirmed_only=True) == {
            "person_name",
            "italian_id",
        }


class TestROPA:
    """Container behaviour of :class:`ROPA`."""

    def _pair(self) -> list[ProcessingActivity]:
        return [
            ProcessingActivity(
                activity_id="a1",
                name="A1",
                purpose="p",
                legal_basis="l",
                controller="c",
                data_categories=[DeclaredDataCategory(raw_text="x", pii_types=("email",))],
            ),
            ProcessingActivity(
                activity_id="a2",
                name="A2",
                purpose="p",
                legal_basis="l",
                controller="c",
                data_categories=[DeclaredDataCategory(raw_text="y", pii_types=("iban",))],
            ),
        ]

    def test_duplicate_activity_id_rejected(self) -> None:
        a1, _ = self._pair()
        with pytest.raises(ValueError, match="duplicate activity_id"):
            ROPA(activities=[a1, a1])

    def test_lookup(self) -> None:
        ropa = ROPA(activities=self._pair())
        found = ropa.activity("a2")
        assert found is not None
        assert found.name == "A2"
        assert ropa.activity("missing") is None

    def test_declared_pii_types_aggregate(self) -> None:
        assert ROPA(activities=self._pair()).declared_pii_types() == {"email", "iban"}
