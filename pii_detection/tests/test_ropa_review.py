"""Smoke tests for the ROPA review web app (create / edit / delete via HTTP)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pii_detection.ropa.review.app import app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ROPA_DB_URL", f"sqlite:///{tmp_path / 'ropa.db'}")
    return TestClient(app)


def test_create_shows_in_list_then_delete(client: TestClient) -> None:
    resp = client.post(
        "/activity",
        data={
            "name": "Gestione personale",
            "purpose": "p",
            "legal_basis": "Contratto",
            "controller": "ACME",
        },
    )
    assert resp.status_code == 200
    activity_id = resp.url.path.rsplit("/", 1)[-1]

    assert "Gestione personale" in client.get("/").text

    assert client.post(f"/activity/{activity_id}/delete").status_code == 200
    assert "Gestione personale" not in client.get("/").text


def test_add_and_confirm_category(client: TestClient) -> None:
    resp = client.post(
        "/activity",
        data={"name": "Marketing", "purpose": "p", "legal_basis": "Consenso", "controller": "ACME"},
    )
    activity_id = resp.url.path.rsplit("/", 1)[-1]

    detail = client.post(
        f"/activity/{activity_id}/category",
        data={"raw_text": "email", "pii_types": ["email"], "mapping_state": "confirmed"},
    )
    assert detail.status_code == 200
    assert "email" in detail.text
