from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

class ProcessingActivityRow(SQLModel, table=True):
    __tablename__ = "processing_activity"

    activity_id: str = Field(primary_key=True)
    name: str
    purpose: str
    legal_basis: str
    controller: str
    dpo: str | None = None

    # String lists -> JSON column
    data_subjects: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    recipients: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    third_country_transfers: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    security_measures: list[str] = field(default_factory=list, sa_column=Column(JSON))
    information_systems: list[str] = Field(default_factory=list, sa_column=Column(JSON))

class DeclaredDataCategoryRow(SQLModel, table=True):
    __tablename__ = "declared_data_category"

    id: int | None = Field(default=None, primary_key=True)
    activity_id: str = Field(foreing_key = "processing_activity.activity_id")
    raw_text: str
    pii_types: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    mapping_state: str

class RetentionRow(SQLModel, table=True):
    __tablename__ = "retention"
    id: int | None = Field(default=None, primary_key=True)
    activity_id: str = Field(foreign_key="processing_activity.activity_id")
    raw_text: str
    duration_months: int | None = None