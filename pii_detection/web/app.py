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
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pii_detection.compliance.checker import check_document
from pii_detection.registry.repository import PIIRepository
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.review.app import app as ropa_app

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


@app.get("/document/{document_id}", response_class=HTMLResponse)
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


@app.post("/document/{document_id}/assign")
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


# The ROPA review (block B1) is reachable under /ropa on the same port.
app.mount("/ropa", ropa_app)
