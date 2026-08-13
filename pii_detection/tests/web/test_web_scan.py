"""Tests for the web folder-scan UI (form, preview, background run, status)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import pii_detection.web.scan_jobs as scan_jobs
from pii_detection.detection.types import DetectorKind, PIICandidate
from pii_detection.registry.repository import PIIRepository
from pii_detection.web.app import app
from pii_detection.web.scan_jobs import get_job


class _FakeDetector:
    """Detector that finds nothing — lets the scan run without Presidio."""

    detector_id = "fake"
    detector_kind = DetectorKind.REGEX

    def detect(self, text: str) -> list[PIICandidate]:
        return []


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PII_DB_URL", f"sqlite:///{tmp_path / 'pii.db'}")
    monkeypatch.setenv("ROPA_DB_URL", f"sqlite:///{tmp_path / 'ropa.db'}")
    monkeypatch.setattr(
        scan_jobs, "_build_detectors", lambda use_gliner: (_FakeDetector(), _FakeDetector())
    )
    return TestClient(app)


def _make_tree(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text("ciao", encoding="utf-8")
    (folder / "sub" / "b.txt").write_text("ciao", encoding="utf-8")
    (folder / "skip.bin").write_text("x", encoding="utf-8")
    return folder


def test_scan_form_renders(client: TestClient) -> None:
    assert client.get("/scan").status_code == 200


def test_preview_lists_scannable_and_skipped(client: TestClient, tmp_path: Path) -> None:
    folder = _make_tree(tmp_path)
    body = client.get("/scan/preview", params={"path": str(folder)}).text
    assert "a.txt" in body and "sub/b.txt" in body  # scannable, relative ids
    assert "skip.bin" in body  # shown among skipped


def test_preview_bad_path_is_400(client: TestClient, tmp_path: Path) -> None:
    resp = client.get("/scan/preview", params={"path": str(tmp_path / "nope")})
    assert resp.status_code == 400


def test_run_starts_background_job_and_completes(client: TestClient, tmp_path: Path) -> None:
    folder = _make_tree(tmp_path)

    resp = client.post("/scan/run", data={"path": str(folder)})
    assert resp.status_code == 200  # followed the 303 to the status page
    job_id = resp.url.path.rsplit("/", 1)[-1]

    for _ in range(100):  # wait for the background thread (<=5s)
        job = get_job(job_id)
        if job is not None and job.state != "running":
            break
        time.sleep(0.05)

    job = get_job(job_id)
    assert job is not None
    assert job.state == "done"
    assert job.result is not None
    assert job.result.scanned == 2  # a.txt, sub/b.txt (skip.bin skipped)
    assert "completata" in client.get(f"/scan/status/{job_id}").text


def test_status_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/scan/status/nope").status_code == 404


def test_run_applies_folder_rules(client: TestClient, tmp_path: Path) -> None:
    folder = _make_tree(tmp_path)
    registry = PIIRepository()  # same PII_DB_URL as the client fixture
    registry.save_rule("", ["catch_all"])  # root prefix matches every document

    resp = client.post("/scan/run", data={"path": str(folder)})
    job_id = resp.url.path.rsplit("/", 1)[-1]
    for _ in range(100):
        job = get_job(job_id)
        if job is not None and job.state != "running":
            break
        time.sleep(0.05)

    job = get_job(job_id)
    assert job is not None and job.state == "done"
    assert job.rules_applied is not None
    assert job.rules_applied.associated == 2  # both scanned docs matched the root rule
    doc = registry.get_document("a.txt")
    assert doc is not None and doc.activity_ids == ["catch_all"]


def _wait_done(job_id: str) -> scan_jobs.ScanJob:
    for _ in range(100):
        job = get_job(job_id)
        if job is not None and job.state != "running":
            break
        time.sleep(0.05)
    job = get_job(job_id)
    assert job is not None
    return job


def test_scan_form_offers_browser_upload(client: TestClient) -> None:
    body = client.get("/scan").text
    assert "upload-form" in body  # the browser file/folder upload form
    assert "/scan/upload" in body  # the fetch target


def test_upload_single_file_is_scanned(client: TestClient) -> None:
    resp = client.post("/scan/upload", files=[("files", ("nota.txt", b"ciao", "text/plain"))])
    assert resp.status_code == 200  # followed the 303 to the status page
    job = _wait_done(resp.url.path.rsplit("/", 1)[-1])
    assert job.state == "done"
    assert job.result is not None and job.result.scanned == 1


def test_upload_folder_preserves_relative_ids(client: TestClient) -> None:
    resp = client.post(
        "/scan/upload",
        files=[
            ("files", ("docs/a.txt", b"ciao", "text/plain")),
            ("files", ("docs/sub/b.txt", b"ciao", "text/plain")),
        ],
    )
    assert resp.status_code == 200
    job = _wait_done(resp.url.path.rsplit("/", 1)[-1])
    assert job.state == "done"
    assert job.result is not None and job.result.scanned == 2
    ids = {document.document_id for document in PIIRepository().documents()}
    assert {"docs/a.txt", "docs/sub/b.txt"} <= ids


def test_upload_rejects_path_traversal(client: TestClient) -> None:
    resp = client.post("/scan/upload", files=[("files", ("../evil.txt", b"x", "text/plain"))])
    assert resp.status_code == 400  # no valid file written
