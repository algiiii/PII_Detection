# Responsible for associating excel data with normalized table in py

from pii_detection.ropa.ingestion.excel_reader import RawTable
from pii_detection.ropa.types import ProcessingActivity, ROPA, DeclaredDataCategory, Retention, MappingState

def split_multi(value: str | None) -> list[str]:
    if value is None:
        return []
    return [piece.strip() for piece in value.split(";") if piece.strip()]

def normalize(table: RawTable) -> ROPA:
    activities = []
    for i, record in enumerate(table.records):
        category = DeclaredDataCategory(
            raw_text=record["data_categories"],
            pii_types=tuple(split_multi(record["data_categories"])),
            mapping_state=MappingState.CONFIRMED,
        )
        retention = Retention(
            raw_text=record["retention_raw"],
            duration_months=record["retention_months"],
        )
        activity = ProcessingActivity(
            activity_id=f"act-{i:04d}",
            name=record["name"],
            purpose=record["purpose"],
            legal_basis=record["legal_basis"],
            controller=record["controller"],
            dpo=record["dpo"],
            data_categories=[category],
            retentions=[retention],
            data_subjects=split_multi(record["data_subjects"]),
            recipients=split_multi(record["recipients"]),
            third_country_transfers=split_multi(record["third_country_transfers"]),
            security_measures=split_multi(record["security_measures"]),
            information_systems=split_multi(record["information_systems"]),
        )
        activities.append(activity)
    return ROPA(activities=activities)