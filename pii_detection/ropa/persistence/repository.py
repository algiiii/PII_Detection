from sqlmodel import Session, SQLModel, create_engine, select

from pii_detection.ropa.types import ROPA, ProcessingActivity, DeclaredDataCategory, Retention, MappingState
from pii_detection.ropa.persistence.models import ProcessingActivityRow, DeclaredDataCategoryRow, RetentionRow

from uuid import uuid4
from pii_detection.detection.config import load_category_catalog, default_config_dir

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

    def create_activity(self, *, name, purpose, legal_basis, controller, dpo = None, data_subjects = None, recipients = None, third_country_transfers = None, security_measures = None, information_systems = None) -> str:
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
    
    def update_acitivty(self, activity_id: str, fields: dict) -> None:
        allowed = 0 # FARSI RETURNARE LA LISTA DEI CAMPI POSSIBILI, NON HARD-CODATA

        if set(fields) - allowed:
            raise ValueError(f"not editable: {sorted(set(fields) - allowed)}")
        
        with Session(self.engine) as session:
            row = session.get(ProcessingActivityRow, activity_id)
            if row is None:
                raise KeyError(activity_id)
            for k,v in fields.items():
                setattr(row, k, v)
            session.add(row); session.commit()

    def delete_activity(self, activity_id: str) -> None:
        with Session(self.engine) as session:
            row = session.get(ProcessingActivityRow, activity_id)

            if row is None:
                raise KeyError(activity_id)
            
            # Prima vanno eliminate le figlie (Retention & declared data
            # Declared data
            for c in session.exec(select(DeclaredDataCategoryRow.where(DeclaredDataCategoryRow.activity_id == activity_id)).all()):
                session.delete(c)

            # Retention
            for r in session.exec(select(RetentionRow.where(RetentionRow.activity_id == activity_id)).all()):
                session.delete(r)
            
            session.delete(row); session.commit()


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

