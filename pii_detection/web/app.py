"""DPO web dashboard: compliance verdict + document–activity association (block B8).

A single FastAPI + Jinja2 web app, served on one port. This module is the
dashboard: it lists the documents in the detected-PII registry, shows each
document's detected PII (never their values — minimization), lets the DPO
associate a document with the processing activities it belongs to (B6), and
renders the compliance verdict (B7). The ROPA review app (block B1) is mounted
under ``/ropa``, so a single server exposes both interfaces.

Configuration is read from the environment (``PII_DB_URL``, ``ROPA_DB_URL``), so
the same image runs locally and inside a container. Writes follow Post/Redirect/Get
(a ``303`` after each ``POST``), like the review app.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pii_detection.compliance.checker import check_document
from pii_detection.compliance.overview import retention_overview
from pii_detection.extraction import supported_suffixes
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.scan_folder import count_unchanged, plan_folder
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.review.app import app as ropa_app
from pii_detection.web.scan_jobs import get_job, start_scan_job

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

app = FastAPI(title="PII compliance")


def get_registry() -> PIIRepository:
    """:returns: a detected-PII registry bound to ``PII_DB_URL``."""
    return PIIRepository(os.environ.get("PII_DB_URL", "sqlite:///data/pii.db"))


def get_ropa() -> ROPARepository:
    """:returns: a ROPA repository bound to ``ROPA_DB_URL``."""
    return ROPARepository(os.environ.get("ROPA_DB_URL", "sqlite:///data/ropa.db"))


@app.get("/", response_class=HTMLResponse)
def documents(request: Request) -> HTMLResponse:
    """List every document in the registry with a short compliance summary.

    :param request: the incoming request (required by the template engine).
    :returns: the rendered document list.
    """
    registry = get_registry()
    rows: list[dict[str, object]] = []
    for document in registry.documents():
        instances = registry.instances_for(document.document_id)
        rows.append(
            {
                "id": document.document_id,
                "path": document.path,
                "pii_count": len(instances),
                "activity_ids": document.activity_ids,
            }
        )
    return _TEMPLATES.TemplateResponse(request, "documents.html", {"rows": rows})


@app.get("/document/{document_id:path}", response_class=HTMLResponse)
def document_detail(request: Request, document_id: str) -> HTMLResponse:
    """Show a document's detected PII, its association and the compliance verdict.

    The verdict is computed read-only (no coverage is written on a ``GET``). It
    counts both confirmed and proposed category mappings, so it is meaningful even
    before the DPO has confirmed everything in the ROPA review.

    :param request: the incoming request.
    :param document_id: identifier of the document to display.
    :returns: the rendered document detail.
    :raises HTTPException: 404 if no document has that id.
    """
    registry = get_registry()
    document = registry.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"unknown document: {document_id}")
    ropa = get_ropa()
    report = None
    if document.activity_ids:
        report = check_document(
            document_id,
            ropa=ropa,
            registry=registry,
            include_proposed=True,
            persist_coverage=False,
        )
    return _TEMPLATES.TemplateResponse(
        request,
        "document.html",
        {
            "document": document,
            "instances": registry.instances_for(document_id),
            "activities": ropa.load(),
            "report": report,
        },
    )


@app.post("/document/{document_id:path}/assign")
def assign_document(
    request: Request, document_id: str, activity_ids: list[str] = Form([])
) -> RedirectResponse:
    """Associate a document with the selected activities and refresh the verdict (B6).

    :param request: the incoming request, for building the redirect URL.
    :param document_id: identifier of an already-recorded document.
    :param activity_ids: the activity ids ticked on the form; an empty selection
        leaves the association unchanged.
    :returns: a 303 redirect back to the document detail.
    :raises HTTPException: 404 if the document is unknown.
    """
    registry = get_registry()
    if registry.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown document: {document_id}")
    if activity_ids:
        registry.assign_activities(document_id, activity_ids)
        check_document(
            document_id,
            ropa=get_ropa(),
            registry=registry,
            include_proposed=True,
            persist_coverage=True,
        )
    url = request.url_for("document_detail", document_id=document_id)
    return RedirectResponse(url=str(url), status_code=303)


@app.get("/retention", response_class=HTMLResponse)
def retention_page(request: Request) -> HTMLResponse:
    """List every document kept past its declared retention, worst first (B7).

    The corpus-wide counterpart of the per-document verdict: the DPO's real
    question about a file share is "what is overdue?", which cannot be answered by
    opening documents one at a time. Read-only, like the document page.

    :param request: the incoming request.
    :returns: the rendered retention overview.
    """
    rows = retention_overview(ropa=get_ropa(), registry=get_registry())
    return _TEMPLATES.TemplateResponse(
        request,
        "retention.html",
        {"rows": rows, "breaches": sum(1 for row in rows if row.flags)},
    )


@app.get("/scan", response_class=HTMLResponse)
def scan_form(request: Request) -> HTMLResponse:
    """Show the scan form: a server-side path and a browser file/folder upload.

    :param request: the incoming request.
    :returns: the rendered form, with the supported file suffixes for the
        client-side upload filter.
    """
    return _TEMPLATES.TemplateResponse(
        request, "scan.html", {"supported": sorted(supported_suffixes())}
    )


@app.get("/scan/preview", response_class=HTMLResponse)
def scan_preview(
    request: Request, path: str, gliner: bool = False, full: bool = False
) -> HTMLResponse:
    """Preview the files a recursive scan of ``path`` would cover, before running.

    Enumerates only (no detection), so it is fast: lists the scannable files
    grouped by format, the skipped (unsupported) ones, and — since the scan is
    incremental by default — how many are already up to date and would not be read
    again. The preview has to state that *before* the run, not explain it after.

    :param request: the incoming request.
    :param path: server-side directory to preview.
    :param gliner: carried through to the run form.
    :param full: preview a forced full re-analysis (nothing counts as unchanged).
    :returns: the rendered preview with a confirm button.
    :raises HTTPException: 400 if ``path`` is not a directory.
    """
    try:
        plan = plan_folder(path)
    except NotADirectoryError:
        raise HTTPException(status_code=400, detail=f"not a directory: {path}") from None
    by_format = Counter(Path(doc_id).suffix.lower() for _path, doc_id in plan.scannable)
    unchanged = 0 if full else count_unchanged(plan, get_registry())
    return _TEMPLATES.TemplateResponse(
        request,
        "scan_preview.html",
        {
            "path": path,
            "gliner": gliner,
            "full": full,
            "plan": plan,
            "by_format": sorted(by_format.items()),
            "unchanged": unchanged,
            "to_scan": len(plan.scannable) - unchanged,
        },
    )


@app.post("/scan/run")
def scan_run(
    request: Request,
    path: str = Form(...),
    gliner: bool = Form(False),
    full: bool = Form(False),
) -> RedirectResponse:
    """Start a background scan of ``path`` and redirect to its status page.

    :param request: the incoming request, for building the redirect URL.
    :param path: server-side directory to scan.
    :param gliner: use GLiNER for the NER.
    :param full: re-analyse every file, including those unchanged since last time.
    :returns: a 303 redirect to the job status page.
    :raises HTTPException: 400 if ``path`` is not a directory.
    """
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {path}")
    job_id = start_scan_job(path, use_gliner=gliner, incremental=not full)
    url = request.url_for("scan_status", job_id=job_id)
    return RedirectResponse(url=str(url), status_code=303)


def _safe_relative_path(filename: str | None) -> PurePosixPath | None:
    """Validate an uploaded file's relative path, rejecting traversal.

    :param filename: the multipart part filename (a client-provided relative path).
    :returns: the sanitized relative path, or ``None`` if it is empty, absolute, or
        escapes the target directory (contains a ``..`` segment).
    """
    if not filename:
        return None
    candidate = PurePosixPath(filename)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        return None
    parts = [part for part in candidate.parts if part not in ("", ".")]
    return PurePosixPath(*parts) if parts else None


@app.post("/scan/upload")
async def scan_upload(
    request: Request, files: list[UploadFile] = File(...), gliner: bool = Form(False)
) -> RedirectResponse:
    """Scan files uploaded from the browser — a single file or a whole folder.

    The browser's native picker runs on the client; the selected files are uploaded
    here, each with its relative path as the part filename (the client sends
    ``webkitRelativePath`` for a folder, the plain name for a single file). They are
    written under a temporary directory, preserving the tree, and scanned by the same
    background job as a server-side folder — with ``prune`` off (a partial upload must
    not remove the rest of the registry) and the temp dir cleaned up when the job ends.

    :param request: the incoming request, for building the redirect URL.
    :param files: the uploaded files; each ``filename`` is a relative path.
    :param gliner: use GLiNER for the NER.
    :returns: a 303 redirect to the job status page.
    :raises HTTPException: 400 if no valid file is uploaded.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="scan_upload_"))
    written = 0
    for upload in files:
        relative = _safe_relative_path(upload.filename)
        if relative is None:
            continue
        destination = temp_dir.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(await upload.read())
        written += 1
    if written == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="nessun file valido caricato")
    # Always a full analysis: the materialized files carry the upload time as their
    # modification time, not the original one, so no file stamp here would mean
    # anything and every document would look modified regardless.
    job_id = start_scan_job(
        str(temp_dir),
        use_gliner=gliner,
        prune=False,
        incremental=False,
        cleanup_dir=str(temp_dir),
    )
    url = request.url_for("scan_status", job_id=job_id)
    return RedirectResponse(url=str(url), status_code=303)


@app.get("/scan/status/{job_id}", response_class=HTMLResponse)
def scan_status(request: Request, job_id: str) -> HTMLResponse:
    """Show a scan job's progress or its final summary.

    While the job runs the page auto-refreshes; when done it shows the summary
    (scanned/skipped/errors/removed + inventory).

    :param request: the incoming request.
    :param job_id: id of the job to display.
    :returns: the rendered status page.
    :raises HTTPException: 404 if the job is unknown.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return _TEMPLATES.TemplateResponse(request, "scan_status.html", {"job": job})


@app.get("/rules", response_class=HTMLResponse)
def rules_page(request: Request) -> HTMLResponse:
    """List the folder→activity rules and the form to add one (block B6).

    Shows, for each rule, how many currently-recorded documents it matches, and —
    after an ``/rules/apply`` — a one-line summary carried over the redirect.

    :param request: the incoming request.
    :returns: the rendered rules page.
    """
    registry = get_registry()
    activities = get_ropa().load()
    names = {activity.id: activity.name for activity in activities}
    document_ids = [document.document_id for document in registry.documents()]
    rows = [
        {
            "prefix": rule.prefix,
            "activity_names": [names.get(a, a) for a in rule.activity_ids],
            "matched": sum(1 for document_id in document_ids if rule.matches(document_id)),
        }
        for rule in registry.folder_rules()
    ]
    applied = None
    if "associated" in request.query_params:
        applied = {
            "associated": request.query_params.get("associated"),
            "skipped_manual": request.query_params.get("skipped_manual"),
            "cleared": request.query_params.get("cleared"),
            "unmatched": request.query_params.get("unmatched"),
        }
    return _TEMPLATES.TemplateResponse(
        request, "rules.html", {"rows": rows, "activities": activities, "applied": applied}
    )


@app.post("/rules")
def create_rule(
    request: Request, prefix: str = Form(""), activity_ids: list[str] = Form([])
) -> RedirectResponse:
    """Create or update a folder→activity rule (upsert by prefix, block B6).

    :param request: the incoming request, for building the redirect URL.
    :param prefix: the folder prefix (normalized by the repository); an empty
        prefix is the root and matches every document.
    :param activity_ids: the activities ticked on the form; an empty selection is
        a no-op.
    :returns: a 303 redirect back to the rules page.
    """
    if activity_ids:
        get_registry().save_rule(prefix, activity_ids)
    return RedirectResponse(url=str(request.url_for("rules_page")), status_code=303)


@app.post("/rules/delete")
def delete_rule(request: Request, prefix: str = Form(...)) -> RedirectResponse:
    """Delete a folder→activity rule by prefix (block B6).

    :param request: the incoming request, for building the redirect URL.
    :param prefix: the prefix of the rule to delete.
    :returns: a 303 redirect back to the rules page, with the reconciliation
        summary (deleting a rule clears the associations it had derived).
    """
    applied = get_registry().delete_rule(prefix)
    url = request.url_for("rules_page").include_query_params(
        associated=applied.associated,
        skipped_manual=applied.skipped_manual,
        cleared=applied.cleared,
        unmatched=applied.unmatched,
    )
    return RedirectResponse(url=str(url), status_code=303)


@app.post("/rules/apply")
def apply_rules(request: Request) -> RedirectResponse:
    """Apply the folder rules to the whole registry now (block B6).

    Manual associations are preserved (manual wins); the counts are carried back
    to the rules page as query parameters for a one-line summary.

    :param request: the incoming request, for building the redirect URL.
    :returns: a 303 redirect back to the rules page, with the apply summary.
    """
    applied = get_registry().apply_folder_rules()
    url = request.url_for("rules_page").include_query_params(
        associated=applied.associated,
        skipped_manual=applied.skipped_manual,
        cleared=applied.cleared,
        unmatched=applied.unmatched,
    )
    return RedirectResponse(url=str(url), status_code=303)


# The ROPA review (block B1) is reachable under /ropa on the same port.
app.mount("/ropa", ropa_app)
