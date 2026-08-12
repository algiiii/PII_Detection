"""Smoke tests for the DPO web app (dashboard + association + mounted ROPA review)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIIMatch,
    TextSpan,
)
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import AssociationSource
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import (
    DeclaredCategory,
    DeclaredMacroCategory,
    MappingState,
    ProcessingActivity,
)
from pii_detection.web.app import app


def _match(start: int, end: int, pii_type: str) -> PIIMatch:
    provenance = DetectionProvenance("det.x", DetectorKind.REGEX, pii_type, 0.9)
    return PIIMatch(
        span=TextSpan(start, end),
        text="?",
        pii_type=pii_type,
        confidence=0.9,
        confirmation_level=ConfirmationLevel.SINGLE_SOURCE,
        sources=[provenance],
        document_id="ignored",
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    ropa_url = f"sqlite:///{tmp_path / 'ropa.db'}"
    pii_url = f"sqlite:///{tmp_path / 'pii.db'}"
    monkeypatch.setenv("ROPA_DB_URL", ropa_url)
    monkeypatch.setenv("PII_DB_URL", pii_url)

    ROPARepository(ropa_url).save(
        [
            ProcessingActivity(
                id="payroll",
                name="Gestione del personale",
                purpose="Amministrazione del rapporto di lavoro",
                macro_categories=[
                    DeclaredMacroCategory(
                        raw_text="Anagrafica",
                        retention_text="5 anni",
                        retention_months=60,
                        categories=[
                            DeclaredCategory(
                                raw_text="iban",
                                pii_types=["iban"],
                                mapping_state=MappingState.CONFIRMED,
                            )
                        ],
                    )
                ],
            )
        ]
    )
    PIIRepository(pii_url).record_scan(
        "doc", [_match(0, 10, "iban"), _match(20, 30, "health_data")], path="/docs/doc.pdf"
    )
    return TestClient(app)


def test_dashboard_lists_document(client: TestClient) -> None:
    assert "doc" in client.get("/").text


def test_document_detail_shows_pii_and_assign_form(client: TestClient) -> None:
    body = client.get("/document/doc").text
    assert "iban" in body and "health_data" in body
    assert "Gestione del personale" in body  # the assignment option
    assert "non ancora associato" in body  # no verdict before association


def test_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/document/nope").status_code == 404


def test_document_id_with_slashes_is_reachable(client: TestClient, tmp_path: Path) -> None:
    # A recursive folder scan uses relative-path ids (e.g. "HR/contratti/x.pdf");
    # the dashboard link and the detail route must handle the embedded slashes.
    registry = PIIRepository(f"sqlite:///{tmp_path / 'pii.db'}")
    registry.record_scan("HR/contratti/mario.pdf", [_match(0, 4, "iban")])

    assert "HR/contratti/mario.pdf" in client.get("/").text  # dashboard renders the link
    assert client.get("/document/HR/contratti/mario.pdf").status_code == 200


def test_assign_persists_and_produces_verdict(client: TestClient, tmp_path: Path) -> None:
    resp = client.post("/document/doc/assign", data={"activity_ids": ["payroll"]})
    assert resp.status_code == 200  # followed the 303 to the detail page
    assert "NON CONFORME" in resp.text  # health_data is orphan

    registry = PIIRepository(f"sqlite:///{tmp_path / 'pii.db'}")
    document = registry.get_document("doc")
    assert document is not None
    assert document.activity_ids == ["payroll"]
    coverage = {i.pii_type: i.processing_activity_id for i in registry.instances_for("doc")}
    assert coverage == {"iban": "payroll", "health_data": None}


def test_ropa_review_is_mounted(client: TestClient) -> None:
    resp = client.get("/ropa/")
    assert resp.status_code == 200
    assert "Gestione del personale" in resp.text


def test_rules_page_lists_activities(client: TestClient) -> None:
    body = client.get("/rules").text
    assert "Gestione del personale" in body  # ROPA activity offered for a rule


def test_create_rule_and_apply_associates_document(client: TestClient, tmp_path: Path) -> None:
    created = client.post("/rules", data={"prefix": "", "activity_ids": ["payroll"]})
    assert created.status_code == 200  # followed the 303 back to /rules
    assert client.post("/rules/apply").status_code == 200

    registry = PIIRepository(f"sqlite:///{tmp_path / 'pii.db'}")
    document = registry.get_document("doc")
    assert document is not None
    assert document.activity_ids == ["payroll"]
    assert document.association_source is AssociationSource.RULE
