"""Tests for the web folder-scan UI (form, preview, background run, status)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import pii_detection.web.scan_jobs as scan_jobs
from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.types import (
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)
from pii_detection.registry.repository import PIIRepository
from pii_detection.web.app import app
from pii_detection.web.scan_jobs import ScanJob, get_job


class _FakeDetector:
    """Detector that finds nothing — lets the scan run without Presidio."""

    detector_id = "fake"
    detector_kind = DetectorKind.REGEX

    def detect(self, text: str) -> list[PIICandidate]:
        return []


class _FakeAIDetector:
    """Fake AI detector that discovers one PII covering the whole text."""

    detector_id = "ai.fake"
    detector_kind = DetectorKind.AI

    def detect(self, text: str) -> list[PIICandidate]:
        if not text:
            return []
        return [
            PIICandidate(
                TextSpan(0, len(text)),
                text,
                DetectionProvenance("ai.fake", DetectorKind.AI, "person_name", 0.6),
            )
        ]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PII_DB_URL", f"sqlite:///{tmp_path / 'pii.db'}")
    monkeypatch.setenv("ROPA_DB_URL", f"sqlite:///{tmp_path / 'ropa.db'}")
    # Never let the host environment turn on AI sampling under the tests.
    monkeypatch.delenv("PII_AI_SAMPLING_RATE", raising=False)

    def _fake_build(
        use_gliner: bool, ai_rate: int
    ) -> tuple[PIIDetector, PIIDetector, PIIDetector | None]:
        # The AI is built only when the scan uses it (ai_rate > 0).
        ai = _FakeAIDetector() if ai_rate > 0 else None
        return _FakeDetector(), _FakeDetector(), ai

    monkeypatch.setattr(scan_jobs, "_build_detectors", _fake_build)
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
    assert job.result is not None and job.result.scanned == 2  # a.txt, sub/b.txt (skip.bin skipped)
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
    assert resp.json()["detail"]


def test_upload_rejection_always_carries_a_reason(client: TestClient) -> None:
    """A rejected upload must say *why*: the page shows the server's ``detail``.

    Selecting a folder with more parts than the multipart parser accepts is
    refused before the route runs, so the reason is not "no valid file" — and
    the page must not claim it is.
    """
    too_many = [
        ("files", (f"docs/f{i}.txt", b"ciao", "text/plain"))
        for i in range(1001)  # Starlette's Request.form() caps files at 1000
    ]
    resp = client.post("/scan/upload", files=too_many)
    assert resp.status_code == 400
    assert "files" in resp.json()["detail"].lower()


# --- incremental scan from the browser ---------------------------------------


def _wait(job_id: str) -> ScanJob:
    for _ in range(100):  # wait for the background thread (<=5s)
        job = get_job(job_id)
        if job is not None and job.state != "running":
            break
        time.sleep(0.05)
    job = get_job(job_id)
    assert job is not None and job.result is not None, f"job {job_id} did not finish"
    return job


def _run_scan(client: TestClient, folder: Path, **data: str) -> ScanJob:
    resp = client.post("/scan/run", data={"path": str(folder), **data})
    return _wait(resp.url.path.rsplit("/", 1)[-1])


def test_second_run_skips_unchanged_files(client: TestClient, tmp_path: Path) -> None:
    folder = _make_tree(tmp_path)
    _run_scan(client, folder)

    job = _run_scan(client, folder)

    assert job.result is not None and job.result.scanned == 0
    assert len(job.result.unchanged) == 2
    assert job.result is not None and job.result.removed == []  # skipped is not gone


def test_full_run_reanalyses_everything(client: TestClient, tmp_path: Path) -> None:
    folder = _make_tree(tmp_path)
    _run_scan(client, folder)

    job = _run_scan(client, folder, full="true")

    assert job.result is not None and job.result.scanned == 2
    assert job.result is not None and job.result.unchanged == []


def test_preview_announces_how_much_is_already_done(
    client: TestClient, tmp_path: Path
) -> None:
    # The preview must state it before the run, not explain it afterwards.
    folder = _make_tree(tmp_path)
    body = client.get("/scan/preview", params={"path": str(folder)}).text
    assert "0</span> invariati" in body.replace('class="badge ok">', "")

    _run_scan(client, folder)
    body = client.get("/scan/preview", params={"path": str(folder)}).text
    assert "2</span> invariati" in body.replace('class="badge ok">', "")


def test_upload_is_always_a_full_scan(client: TestClient) -> None:
    # Uploaded files carry the upload time as their mtime, so no stamp of theirs
    # would mean anything: the job must not pretend it can skip them.
    files = [("files", ("nota.txt", b"IBAN IT60X0542811101000000123456", "text/plain"))]
    resp = client.post("/scan/upload", files=files)
    job = _wait(resp.url.path.rsplit("/", 1)[-1])
    assert job.incremental is False
    assert job.result is not None and job.result.scanned == 1


# --- AI second opinion -------------------------------------------------------


def test_scan_form_has_ai_rate_selector(client: TestClient) -> None:
    body = client.get("/scan").text
    assert 'name="ai_rate"' in body  # the AI sampling menu (server form)


def test_run_with_ai_rate_one_runs_it_on_every_document(
    client: TestClient, tmp_path: Path
) -> None:
    folder = _make_tree(tmp_path)
    job = _run_scan(client, folder, ai_rate="1")  # 1 = every document
    assert job.ai_rate == 1
    assert job.phase == "ai"  # went through the second (AI) phase
    assert job.result is not None and job.result.ai_documents == 2  # both scannable docs
    # Phase 2 enriched the registry: the AI discovery is on the document.
    instances = PIIRepository().instances_for("a.txt")
    assert any("ai.fake" in i.sources for i in instances)


def test_scan_form_preselects_env_default_rate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PII_AI_SAMPLING_RATE", "50")
    body = client.get("/scan").text
    assert 'value="50" selected' in body  # the env default pre-selects the menu option


def test_run_without_ai_leaves_rate_zero(client: TestClient, tmp_path: Path) -> None:
    folder = _make_tree(tmp_path)
    job = _run_scan(client, folder)  # no ai_rate -> default 0
    assert job.ai_rate == 0
    assert job.phase == "traditional"  # no AI phase
    assert job.result is not None and job.result.ai_documents == 0


def test_document_ai_trigger_starts_job_and_discovers(client: TestClient, tmp_path: Path) -> None:
    folder = _make_tree(tmp_path)
    _run_scan(client, folder)  # records a.txt, sub/b.txt (fakes find no PII)

    resp = client.post("/document/a.txt/scan-ai")
    assert resp.status_code == 200  # followed the 303 to the status page
    job = _wait(resp.url.path.rsplit("/", 1)[-1])
    assert job.state == "done"
    assert job.document_id == "a.txt"

    body = client.get("/document/a.txt").text
    assert "person_name" in body  # the AI discovery is now on the document page


def test_document_ai_trigger_unknown_document_is_404(client: TestClient) -> None:
    assert client.post("/document/nope.txt/scan-ai").status_code == 404


def test_document_ai_trigger_missing_source_redirects_with_message(
    client: TestClient, tmp_path: Path
) -> None:
    # A browser upload's source is removed after its scan: re-scanning must not start a
    # doomed job on a dangling path — it redirects back with an explanatory message.
    folder = _make_tree(tmp_path)
    _run_scan(client, folder)
    (folder / "a.txt").unlink()  # simulate the upload cleanup

    resp = client.post("/document/a.txt/scan-ai", follow_redirects=False)
    assert resp.status_code == 303
    assert "scan_ai_error" in resp.headers["location"]
    # The button is gone and the reason is shown on the document page.
    body = client.get("/document/a.txt").text
    assert "Analizza con l'AI" not in body
    assert "Analisi AI on-demand non disponibile" in body


# --- active-scan indicator ---------------------------------------------------


def test_dashboard_shows_active_scan_indicator(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    running = ScanJob(id="j1", folder="/data/share", use_gliner=False, done=3, total=10)
    monkeypatch.setattr(scan_jobs, "_JOBS", {"j1": running})

    body = client.get("/").text  # the shared header shows it on any page

    assert "1 scansione in corso" in body
    assert "3/10 documenti" in body
    assert "/scan/status/j1" in body  # links to the live status page


def test_no_indicator_when_no_scan_is_running(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    done = ScanJob(id="j2", folder="/data/share", use_gliner=False, state="done")
    monkeypatch.setattr(scan_jobs, "_JOBS", {"j2": done})
    assert "in corso" not in client.get("/").text
