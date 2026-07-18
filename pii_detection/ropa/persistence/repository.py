from typing import Any

from sqlmodel import Session, SQLModel, create_engine, select

from pii_detection.ropa.types import ROPA, ProcessingActivity, DeclaredDataCategory, Retention, MappingState
from pii_detection.ropa.persistence.models import ProcessingActivityRow, DeclaredDataCategoryRow, RetentionRow

from uuid import uuid4
from pii_detection.detection.config import load_category_catalog, default_config_dir

#: Fields of a :class:`ProcessingActivity` a caller may update through
#: :meth:`ROPARepository.update_activity`. The primary key ``activity_id`` is
#: intentionally excluded: it identifies the row and must stay stable.
EDITABLE_ACTIVITY_FIELDS: frozenset[str] = frozenset({
    "name",
    "purpose",
    "legal_basis",
    "controller",
    "dpo",
    "data_subjects",
    "recipients",
    "third_country_transfers",
    "security_measures",
    "information_systems",
})


class ROPARepository:
    def __init__(self, url: str = "sqlite:///ropa.db"):
        self.engine = create_engine(url)
        SQLModel.metadata.create_all(self.engine)

        # Carica catalogo
        self._catalog = load_category_catalog(default_config_dir() / "categories.yaml")

    # Per scrivere in DB con una sola transazione
    def save_ropa(self, ropa:ROPA) -> None:
        with Session(self.engine) as session:
            for a in ropa.activities:
                session.add(_activity_to_row(a))
                for c in a.data_categories:
                    session.add(_category_to_row(a.activity_id, c))
                for r in a.retentions:
                    session.add(_retention_to_row(a.activity_id, r))
            session.commit()

    def clear(self) -> None:
        """Delete every row from the register (activities and their children).

        Wipes the three tables (categories and retentions first, then the parent
        activities), leaving an empty but initialized database. Used by a
        ``--replace`` re-ingestion; it is destructive and drops any DPO edits.
        """
        with Session(self.engine) as session:
            for model in (DeclaredDataCategoryRow, RetentionRow, ProcessingActivityRow):
                for row in session.exec(select(model)).all():
                    session.delete(row)
            session.commit()

    # Legge e ricostruisce per types.py
    def load_ropa(self) -> ROPA:
        with Session(self.engine) as session:
            activities = []
            for row in session.exec(select(ProcessingActivityRow)).all():

                # Categories
                cats = session.exec(
                    select(DeclaredDataCategoryRow).where(DeclaredDataCategoryRow.activity_id == row.activity_id)
                ).all()

                # Retentions
                rets = session.exec(
                        select(RetentionRow)
                        .where(RetentionRow.activity_id == row.activity_id)
                ).all()
                
                activities.append(ProcessingActivity(
                    activity_id = row.activity_id, 
                    name = row.name, 
                    purpose=row.purpose, 
                    legal_basis = row.legal_basis,
                    controller = row.controller,
                    dpo = row.dpo,
                    data_categories = [DeclaredDataCategory(
                        raw_text = c.raw_text,
                        pii_types = tuple(c.pii_types),
                        mapping_state = MappingState(c.mapping_state),
                    ) for c in cats],
                    retentions = [Retention(
                        raw_text = r.raw_text,
                        duration_months = r.duration_months
                    ) for r in rets],

                    data_subjects = row.data_subjects,
                    recipients = row.recipients,
                    third_country_transfers = row.third_country_transfers,
                    security_measures = row.security_measures,
                    information_systems = row.information_systems,
                ))

        return ROPA(activities = activities)
    
    def get_activity_row(self, activity_id: str) -> ProcessingActivityRow | None:
        with Session(self.engine) as session:
            return session.get(ProcessingActivityRow, activity_id)
        
    def list_category_rows(self, activity_id: str) -> list[DeclaredDataCategoryRow]:
        with Session(self.engine) as session:
            return list(session.exec(
                select(DeclaredDataCategoryRow).where(
                    DeclaredDataCategoryRow.activity_id == activity_id)
            ).all())
        
    def list_activity_rows(self) -> list[ProcessingActivityRow]:
        with Session(self.engine) as session:
            return list(session.exec(
                select(ProcessingActivityRow)
            ).all())
        
    def list_retention_rows(self, activity_id: str) -> list[RetentionRow]:
        with Session(self.engine) as session:
            return list(session.exec(
                select(RetentionRow).where(RetentionRow.activity_id == activity_id)
            ).all())
    
    def _require_pii_types(self, pii_types: list[str]) -> None:
        unknown = [t for t in pii_types if t not in self._catalog]
        if unknown:
            raise ValueError(f"unknown pii_type(s): {unknown}")
        
    # MODIFICA

    def create_activity(
        self,
        *,
        name: str,
        purpose: str,
        legal_basis: str,
        controller: str,
        dpo: str | None = None,
        data_subjects: list[str] | None = None,
        recipients: list[str] | None = None,
        third_country_transfers: list[str] | None = None,
        security_measures: list[str] | None = None,
        information_systems: list[str] | None = None,
    ) -> str:
        """Create a new processing activity with empty children.

        :param name: name of the processing activity.
        :param purpose: declared purpose.
        :param legal_basis: legal basis.
        :param controller: data controller.
        :param dpo: data protection officer, if designated.
        :param data_subjects: categories of data subjects.
        :param recipients: recipients of the data.
        :param third_country_transfers: transfers outside the EU/EEA.
        :param security_measures: technical/organizational measures.
        :param information_systems: systems the data resides in.
        :returns: the autogenerated ``activity_id`` (a UUID hex string).
        """
        activity_id = uuid4().hex
        with Session(self.engine) as session:
            session.add(ProcessingActivityRow(
                activity_id = activity_id,
                name = name,
                purpose = purpose,
                legal_basis = legal_basis,
                controller = controller,
                dpo = dpo,
                data_subjects = data_subjects or [],
                recipients = recipients or [],
                third_country_transfers = third_country_transfers or [],
                security_measures = security_measures or [],
                information_systems = information_systems or [],
            ))
            session.commit()
        return activity_id
    
    def update_activity(self, activity_id: str, fields: dict[str, Any]) -> None:
        """Update the editable fields of one processing activity.

        :param activity_id: identifier of the activity to update.
        :param fields: field name → new value; only keys in
            :data:`EDITABLE_ACTIVITY_FIELDS` are accepted.
        :raises ValueError: if ``fields`` contains a non-editable key.
        :raises KeyError: if no activity has that ``activity_id``.
        """
        unknown = set(fields) - EDITABLE_ACTIVITY_FIELDS
        if unknown:
            raise ValueError(f"not editable: {sorted(unknown)}")

        with Session(self.engine) as session:
            row = session.get(ProcessingActivityRow, activity_id)
            if row is None:
                raise KeyError(activity_id)
            for key, value in fields.items():
                setattr(row, key, value)
            session.add(row)
            session.commit()

    def delete_activity(self, activity_id: str) -> None:
        """Delete an activity and its child rows (categories, retentions).

        SQLite does not enforce foreign keys by default, so the children are
        removed explicitly before the parent (manual cascade).

        :param activity_id: identifier of the activity to delete.
        :raises KeyError: if no activity has that ``activity_id``.
        """
        with Session(self.engine) as session:
            row = session.get(ProcessingActivityRow, activity_id)
            if row is None:
                raise KeyError(activity_id)

            # Delete the children first (categories, then retentions).
            for c in session.exec(
                select(DeclaredDataCategoryRow).where(
                    DeclaredDataCategoryRow.activity_id == activity_id
                )
            ).all():
                session.delete(c)

            for r in session.exec(
                select(RetentionRow).where(RetentionRow.activity_id == activity_id)
            ).all():
                session.delete(r)

            session.delete(row)
            session.commit()

    # ----- CATEGORY CRUD -----

    def add_category(
        self,
        activity_id: str,
        raw_text: str,
        pii_types: tuple[str, ...] = (),
        mapping_state: MappingState = MappingState.PROPOSED,
    ) -> int:
        """Add a declared data category to an activity.

        :param activity_id: activity the category belongs to.
        :param raw_text: original free-text category from the register.
        :param pii_types: catalog ids the category resolves to.
        :param mapping_state: whether the mapping is proposed or confirmed.
        :returns: the autogenerated id of the new category row.
        :raises KeyError: if the activity does not exist.
        :raises ValueError: if a ``pii_type`` is not in the catalog.
        """
        self._require_pii_types(list(pii_types))
        with Session(self.engine) as session:
            if session.get(ProcessingActivityRow, activity_id) is None:
                raise KeyError(activity_id)
            row = DeclaredDataCategoryRow(
                activity_id=activity_id,
                raw_text=raw_text,
                pii_types=list(pii_types),
                mapping_state=mapping_state.value,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            assert row.id is not None
            return row.id

    def update_category(
        self,
        category_id: int,
        raw_text: str,
        pii_types: tuple[str, ...],
        mapping_state: MappingState,
    ) -> None:
        """Replace the editable fields of a declared data category.

        Full replacement: the DPO's edit form submits all three fields at once,
        so there is no partial-update ambiguity.

        :param category_id: id of the category row to update.
        :param raw_text: new free-text category.
        :param pii_types: new catalog ids it resolves to.
        :param mapping_state: new confirmation state.
        :raises KeyError: if no category has that id.
        :raises ValueError: if a ``pii_type`` is not in the catalog.
        """
        self._require_pii_types(list(pii_types))
        with Session(self.engine) as session:
            row = session.get(DeclaredDataCategoryRow, category_id)
            if row is None:
                raise KeyError(category_id)
            row.raw_text = raw_text
            row.pii_types = list(pii_types)
            row.mapping_state = mapping_state.value
            session.add(row)
            session.commit()

    def delete_category(self, category_id: int) -> str:
        """Delete a declared data category.

        :param category_id: id of the category row to delete.
        :returns: the ``activity_id`` of the parent activity (so the caller can
            redirect back to it).
        :raises KeyError: if no category has that id.
        """
        with Session(self.engine) as session:
            row = session.get(DeclaredDataCategoryRow, category_id)
            if row is None:
                raise KeyError(category_id)
            activity_id = row.activity_id
            session.delete(row)
            session.commit()
        return activity_id

    # ----- RETENTION CRUD -----

    def add_retention(
        self, activity_id: str, raw_text: str, duration_months: int | None = None
    ) -> int:
        """Add a retention rule to an activity.

        :param activity_id: activity the retention belongs to.
        :param raw_text: original retention wording from the register.
        :param duration_months: length in months, or ``None`` for a criterion.
        :returns: the autogenerated id of the new retention row.
        :raises KeyError: if the activity does not exist.
        :raises ValueError: if ``duration_months`` is negative.
        """
        if duration_months is not None and duration_months < 0:
            raise ValueError(f"negative duration_months: {duration_months}")
        with Session(self.engine) as session:
            if session.get(ProcessingActivityRow, activity_id) is None:
                raise KeyError(activity_id)
            row = RetentionRow(
                activity_id=activity_id,
                raw_text=raw_text,
                duration_months=duration_months,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            assert row.id is not None
            return row.id

    def update_retention(
        self, retention_id: int, raw_text: str, duration_months: int | None
    ) -> None:
        """Replace the fields of a retention rule (full replacement).

        :param retention_id: id of the retention row to update.
        :param raw_text: new retention wording.
        :param duration_months: new length in months, or ``None`` for a criterion.
        :raises KeyError: if no retention has that id.
        :raises ValueError: if ``duration_months`` is negative.
        """
        if duration_months is not None and duration_months < 0:
            raise ValueError(f"negative duration_months: {duration_months}")
        with Session(self.engine) as session:
            row = session.get(RetentionRow, retention_id)
            if row is None:
                raise KeyError(retention_id)
            row.raw_text = raw_text
            row.duration_months = duration_months
            session.add(row)
            session.commit()

    def delete_retention(self, retention_id: int) -> str:
        """Delete a retention rule.

        :param retention_id: id of the retention row to delete.
        :returns: the ``activity_id`` of the parent activity.
        :raises KeyError: if no retention has that id.
        """
        with Session(self.engine) as session:
            row = session.get(RetentionRow, retention_id)
            if row is None:
                raise KeyError(retention_id)
            activity_id = row.activity_id
            session.delete(row)
            session.commit()
        return activity_id

    def catalog_ids(self) -> list[str]:
        """List the ``pii_type`` ids of the catalog (for building UI choices).

        :returns: the catalog category ids, in declaration order.
        """
        return [c.id for c in self._catalog]


# ----- SUPPORT METHODS -----

# Funzioni di traduzione effettiva
def _activity_to_row(a: ProcessingActivity) -> ProcessingActivityRow:
    return ProcessingActivityRow(
        activity_id=a.activity_id, name=a.name, purpose=a.purpose,
        legal_basis=a.legal_basis, controller=a.controller, dpo=a.dpo,
        data_subjects=a.data_subjects, recipients=a.recipients,
        third_country_transfers=a.third_country_transfers,
        security_measures=a.security_measures, information_systems=a.information_systems,
    )

def _category_to_row(activity_id: str, c: DeclaredDataCategory) -> DeclaredDataCategoryRow:
    return DeclaredDataCategoryRow(
        activity_id = activity_id, raw_text = c.raw_text, pii_types = list(c.pii_types), mapping_state=c.mapping_state.value,
    )

def _retention_to_row(activity_id: str, r:Retention) -> RetentionRow:
    return RetentionRow(activity_id=activity_id, raw_text=r.raw_text, duration_months=r.duration_months)

