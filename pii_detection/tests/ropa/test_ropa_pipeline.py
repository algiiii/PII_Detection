"""End-to-end pipeline test for block B1 on the real CNIL fixture.

Exercises the whole B1 chain wired together — read_sheet -> normalize (+ retention)
-> repository.save -> map_categories (shipped dictionary) -> DPO confirmation via
the review web app — to check the pieces compose, not just work in isolation.
This is the block-scoped integration test; a cross-block pipeline (B3 -> B4 -> ...)
waits until those blocks exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pii_detection.ropa.ingestion.category_mapper import build_dictionary_mapper
from pii_detection.ropa.ingestion.pipeline import ingest_file, map_categories
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.review.app import app
from pii_detection.ropa.types import MappingState

_FIXTURE = Path(__file__).parent / "record-processing-activities.ods"


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'ropa.db'}"


def test_b1_pipeline_ingest_map_and_confirm(db_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Ingest the whole workbook: only the real activity sheet survives.
    activities = ingest_file(_FIXTURE, db_url)
    assert len(activities) == 1

    # 2. Deterministic mapping pass over the persisted register.
    assert map_categories(ROPARepository(db_url), build_dictionary_mapper()) == 3

    # 3. State after mapping: retention parsed to months, categories resolved onto
    #    the catalog but still PROPOSED (awaiting the DPO).
    activity = ROPARepository(db_url).get("payroll-management")
    assert activity is not None
    assert all(m.retention_months == 60 for m in activity.macro_categories)
    resolved = {t for m in activity.macro_categories for c in m.categories for t in c.pii_types}
    assert {"iban", "swiss_avs", "person_name", "address"} <= resolved
    assert all(
        c.mapping_state is MappingState.PROPOSED
        for m in activity.macro_categories
        for c in m.categories
    )

    # 4. The DPO opens the register and confirms one macro category over HTTP.
    economic = next(m for m in activity.macro_categories if m.raw_text.startswith("Economic"))
    monkeypatch.setenv("ROPA_DB_URL", db_url)
    client = TestClient(app)
    assert client.get("/activity/payroll-management").status_code == 200
    resp = client.post(
        f"/macro/{economic.id}/confirm", data={"activity_id": "payroll-management"}
    )
    assert resp.status_code == 200  # followed the 303 to the detail page

    # 5. That macro's categories are now CONFIRMED; the others stay PROPOSED.
    activity = ROPARepository(db_url).get("payroll-management")
    assert activity is not None
    for macro in activity.macro_categories:
        expected = (
            MappingState.CONFIRMED
            if macro.raw_text.startswith("Economic")
            else MappingState.PROPOSED
        )
        assert all(c.mapping_state is expected for c in macro.categories)
