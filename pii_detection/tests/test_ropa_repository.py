"""Simple tests for the ROPA SQLite repository (save/load round-trip)."""

from __future__ import annotations

from pathlib import Path

import pytest

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


class TestActivityCrud:
    """Create / update / delete of processing activities."""

    def _repo(self, tmp_path: Path) -> ROPARepository:
        return ROPARepository(f"sqlite:///{tmp_path / 'ropa.db'}")

    def test_create_and_update(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        activity_id = repo.create_activity(
            name="Payroll", purpose="p", legal_basis="Contract", controller="ACME"
        )
        repo.update_activity(activity_id, {"name": "Payroll v2", "recipients": ["INPS"]})
        activity = repo.load_ropa().activity(activity_id)
        assert activity is not None
        assert activity.name == "Payroll v2"
        assert activity.recipients == ["INPS"]

    def test_update_rejects_unknown_field(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        activity_id = repo.create_activity(
            name="Payroll", purpose="p", legal_basis="Contract", controller="ACME"
        )
        with pytest.raises(ValueError):
            repo.update_activity(activity_id, {"activity_id": "hacked"})

    def test_update_unknown_activity(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        with pytest.raises(KeyError):
            repo.update_activity("nope", {"name": "x"})

    def test_delete_cascades_to_children(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        activity_id = repo.create_activity(
            name="Payroll", purpose="p", legal_basis="Contract", controller="ACME"
        )
        repo.add_category(activity_id, "dati anagrafici", ("person_name",))
        repo.add_retention(activity_id, "5 anni", 60)

        repo.delete_activity(activity_id)

        assert repo.load_ropa().activity(activity_id) is None
        assert repo.list_category_rows(activity_id) == []
        assert repo.list_retention_rows(activity_id) == []


class TestCategoryCrud:
    """Add / update / delete of declared data categories."""

    def _repo_with_activity(self, tmp_path: Path) -> tuple[ROPARepository, str]:
        repo = ROPARepository(f"sqlite:///{tmp_path / 'ropa.db'}")
        activity_id = repo.create_activity(
            name="Payroll", purpose="p", legal_basis="Contract", controller="ACME"
        )
        return repo, activity_id

    def test_add_appears_in_list(self, tmp_path: Path) -> None:
        repo, activity_id = self._repo_with_activity(tmp_path)
        category_id = repo.add_category(activity_id, "email", ("email",))
        rows = repo.list_category_rows(activity_id)
        assert [r.id for r in rows] == [category_id]
        assert rows[0].pii_types == ["email"]
        assert rows[0].mapping_state == MappingState.PROPOSED.value

    def test_update_changes_types_and_state(self, tmp_path: Path) -> None:
        repo, activity_id = self._repo_with_activity(tmp_path)
        category_id = repo.add_category(activity_id, "email", ("email",))
        repo.update_category(
            category_id, "email + telefono", ("email", "phone"), MappingState.CONFIRMED
        )
        row = repo.list_category_rows(activity_id)[0]
        assert row.pii_types == ["email", "phone"]
        assert row.mapping_state == MappingState.CONFIRMED.value

    def test_delete_returns_parent(self, tmp_path: Path) -> None:
        repo, activity_id = self._repo_with_activity(tmp_path)
        category_id = repo.add_category(activity_id, "email", ("email",))
        assert repo.delete_category(category_id) == activity_id
        assert repo.list_category_rows(activity_id) == []

    def test_add_rejects_unknown_pii_type(self, tmp_path: Path) -> None:
        repo, activity_id = self._repo_with_activity(tmp_path)
        with pytest.raises(ValueError):
            repo.add_category(activity_id, "x", ("not_a_type",))


class TestRetentionCrud:
    """Add / update / delete of retention rules."""

    def _repo_with_activity(self, tmp_path: Path) -> tuple[ROPARepository, str]:
        repo = ROPARepository(f"sqlite:///{tmp_path / 'ropa.db'}")
        activity_id = repo.create_activity(
            name="Payroll", purpose="p", legal_basis="Contract", controller="ACME"
        )
        return repo, activity_id

    def test_add_update_delete(self, tmp_path: Path) -> None:
        repo, activity_id = self._repo_with_activity(tmp_path)
        retention_id = repo.add_retention(activity_id, "5 anni", 60)
        repo.update_retention(retention_id, "criterio", None)
        row = repo.list_retention_rows(activity_id)[0]
        assert row.duration_months is None
        assert repo.delete_retention(retention_id) == activity_id
        assert repo.list_retention_rows(activity_id) == []

    def test_negative_duration_rejected(self, tmp_path: Path) -> None:
        repo, activity_id = self._repo_with_activity(tmp_path)
        with pytest.raises(ValueError):
            repo.add_retention(activity_id, "bad", -1)
