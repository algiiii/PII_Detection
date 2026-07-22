"""Smoke tests for the ROPA review web app (browse + confirm via HTTP)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.review.app import app
from pii_detection.ropa.types import (
    DeclaredCategory,
    DeclaredMacroCategory,
    MappingState,
    ProcessingActivity,
)


def _seed(db_url: str) -> None:
    """Persist one activity with a single macro category and category."""
    ROPARepository(db_url).save(
        [
            ProcessingActivity(
                id="payroll",
                name="Payroll management",
                purpose="Administer the employment relationship",
                macro_categories=[
                    DeclaredMacroCategory(
                        raw_text="Civil status, identity",
                        retention_text="5 years",
                        retention_months=60,
                        categories=[DeclaredCategory(raw_text="name, address", pii_types=[])],
                    )
                ],
            )
        ]
    )


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'ropa.db'}"


@pytest.fixture
def client(db_url: str, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ROPA_DB_URL", db_url)
    _seed(db_url)
    return TestClient(app)


def test_index_lists_activity(client: TestClient) -> None:
    assert "Payroll management" in client.get("/").text


def test_detail_shows_three_level_tree(client: TestClient) -> None:
    body = client.get("/activity/payroll").text
    assert "Civil status, identity" in body  # macro category
    assert "name, address" in body  # declared category


def test_unknown_activity_is_404(client: TestClient) -> None:
    assert client.get("/activity/nope").status_code == 404


def test_update_category_sets_types_and_state(client: TestClient, db_url: str) -> None:
    resp = client.post(
        "/category/1",
        data={"activity_id": "payroll", "pii_types": ["person_name"], "mapping_state": "confirmed"},
    )
    assert resp.status_code == 200  # followed the 303 to the detail page

    activity = ROPARepository(db_url).get("payroll")
    assert activity is not None
    category = activity.macro_categories[0].categories[0]
    assert category.pii_types == ["person_name"]
    assert category.mapping_state is MappingState.CONFIRMED


def test_update_category_rejects_unknown_pii_type(client: TestClient) -> None:
    resp = client.post(
        "/category/1",
        data={"activity_id": "payroll", "pii_types": ["not_a_type"], "mapping_state": "proposed"},
    )
    assert resp.status_code == 400


def test_confirm_macro_confirms_children(client: TestClient, db_url: str) -> None:
    resp = client.post("/macro/1/confirm", data={"activity_id": "payroll"})
    assert resp.status_code == 200

    activity = ROPARepository(db_url).get("payroll")
    assert activity is not None
    states = [c.mapping_state for m in activity.macro_categories for c in m.categories]
    assert states == [MappingState.CONFIRMED]
