"""Simple tests for the ROPA SQLite repository (save/load round-trip)."""

from __future__ import annotations

from pathlib import Path

from pii_detection.ropa.persistence.repository import ROPARepository
from pii_detection.ropa.types import (
    ROPA,
    DeclaredDataCategory,
    MappingState,
    ProcessingActivity,
    Retention,
)


def _sample_ropa() -> ROPA:
    return ROPA(
        activities=[
            ProcessingActivity(
                activity_id="act-0000",
                name="Gestione del personale",
                purpose="p",
                legal_basis="Contratto",
                controller="ACME",
                dpo="dpo@acme.example",
                data_categories=[
                    DeclaredDataCategory(
                        raw_text="person_name;health_data",
                        pii_types=("person_name", "health_data"),
                        mapping_state=MappingState.CONFIRMED,
                    )
                ],
                retentions=[Retention(raw_text="10 anni", duration_months=120)],
                data_subjects=["Dipendenti", "Collaboratori"],
                recipients=["INPS"],
            ),
            ProcessingActivity(
                activity_id="act-0001",
                name="Marketing",
                purpose="p",
                legal_basis="Consenso",
                controller="ACME",
                data_categories=[
                    DeclaredDataCategory(raw_text="email", pii_types=("email",))
                ],
                retentions=[Retention(raw_text="fino a revoca", duration_months=None)],
            ),
        ]
    )


class TestROPARepository:
    """Round-trip behaviour of :class:`ROPARepository`."""

    def _repo(self, tmp_path: Path) -> ROPARepository:
        return ROPARepository(f"sqlite:///{tmp_path / 'ropa.db'}")

    def test_activity_count(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        repo.save_ropa(_sample_ropa())
        assert len(repo.load_ropa().activities) == 2

    def test_pii_types_preserved(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        original = _sample_ropa()
        repo.save_ropa(original)
        assert repo.load_ropa().declared_pii_types() == original.declared_pii_types()

    def test_multivalue_and_mapping_state(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        repo.save_ropa(_sample_ropa())
        hr = repo.load_ropa().activity("act-0000")
        assert hr is not None
        assert hr.data_subjects == ["Dipendenti", "Collaboratori"]
        assert hr.data_categories[0].mapping_state is MappingState.CONFIRMED

    def test_retention_criterion_preserved(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        repo.save_ropa(_sample_ropa())
        mkt = repo.load_ropa().activity("act-0001")
        assert mkt is not None
        assert mkt.retentions[0].duration_months is None
