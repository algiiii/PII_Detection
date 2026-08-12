"""DPO review web app for the ROPA — browse and confirm (block B1).

A small FastAPI + Jinja2 application that lets the DPO browse the ingested
register as a three-level tree (activity → macro category → declared category)
and confirm the mapping of each declared category onto the ``pii_type`` catalog.
The register is authored from a CNIL spreadsheet, which the DPO can **import** from
the browser (file upload) and prune by **deleting** a single activity; field-by-field
authoring stays out of scope — the file is the source of truth.

It is the "interface on a port": the database stays a file behind the app,
reached only through :class:`~pii_detection.ropa.repository.ROPARepository`. The
database URL is read from the ``ROPA_DB_URL`` environment variable and defaults
to ``sqlite:///ropa.db``. Every write follows the Post/Redirect/Get pattern (a
``303`` redirect after each ``POST``), so a browser refresh never re-submits a
form.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError

from pii_detection.ropa.ingestion.category_mapper import build_mapper
from pii_detection.ropa.ingestion.pipeline import ingest_file, map_categories
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import MappingState

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

app = FastAPI(title="ROPA review")


def get_repository() -> ROPARepository:
    """Open the repository pointed to by ``ROPA_DB_URL``.

    :returns: a repository bound to the configured database URL.
    """
    return ROPARepository(os.environ.get("ROPA_DB_URL", "sqlite:///ropa.db"))


def _redirect_index(request: Request, **params: object) -> RedirectResponse:
    """Build a 303 redirect to the index carrying banner query parameters.

    :param request: the incoming request, so the URL is correct under the ``/ropa``
        mount prefix.
    :param params: query parameters for the index banner (``imported`` / ``error``).
    :returns: the redirect response.
    """
    url = request.url_for("index").include_query_params(**params)
    return RedirectResponse(url=str(url), status_code=303)


# ----- READ -----


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """List every processing activity in the register.

    :param request: the incoming request (required by the template engine).
    :returns: the rendered activity list, with an optional banner after a redirect
        (``?imported=N`` on a successful import, ``?error=…`` on failure).
    """
    activities = get_repository().load()
    return _TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "activities": activities,
            "imported": request.query_params.get("imported"),
            "error": request.query_params.get("error"),
        },
    )


@app.get("/activity/{activity_id}", response_class=HTMLResponse)
def activity_detail(request: Request, activity_id: str) -> HTMLResponse:
    """Show one activity as a three-level tree with the confirmation forms.

    :param request: the incoming request.
    :param activity_id: identifier of the activity to display.
    :returns: the rendered activity detail.
    :raises HTTPException: 404 if no activity has that id.
    """
    repo = get_repository()
    activity = repo.get(activity_id)
    if activity is None:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}")
    return _TEMPLATES.TemplateResponse(
        request,
        "activity.html",
        {"a": activity, "catalog_ids": repo.catalog_ids()},
    )


# ----- CONFIRM / EDIT -----


@app.post("/category/{category_id}")
def update_category_submit(
    request: Request,
    category_id: int,
    activity_id: str = Form(...),
    pii_types: list[str] = Form([]),
    mapping_state: str = Form(MappingState.PROPOSED.value),
) -> RedirectResponse:
    """Update a declared category's ``pii_types`` and mapping state.

    :param request: the incoming request, used to build the redirect URL so it is
        correct whether the app runs standalone or mounted under a path prefix.
    :param category_id: id of the declared category to update.
    :param activity_id: parent activity, carried by the form for the redirect.
    :param pii_types: the catalog ids ticked on the form.
    :param mapping_state: the new mapping state (``proposed``/``confirmed``).
    :returns: a 303 redirect to the activity's detail page.
    :raises HTTPException: 404 if the category is unknown, 400 on a bad value.
    """
    try:
        get_repository().update_category(category_id, pii_types, MappingState(mapping_state))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown category: {category_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    url = request.url_for("activity_detail", activity_id=activity_id)
    return RedirectResponse(url=str(url), status_code=303)


@app.post("/macro/{macro_id}/confirm")
def confirm_macro_submit(
    request: Request, macro_id: int, activity_id: str = Form(...)
) -> RedirectResponse:
    """Confirm every declared category under a macro category in one click.

    :param request: the incoming request, used to build the redirect URL so it is
        correct whether the app runs standalone or mounted under a path prefix.
    :param macro_id: id of the macro category whose children to confirm.
    :param activity_id: parent activity, carried by the form for the redirect.
    :returns: a 303 redirect to the activity's detail page.
    :raises HTTPException: 404 if the macro category is unknown.
    """
    try:
        get_repository().confirm_macro(macro_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown macro category: {macro_id}") from None
    url = request.url_for("activity_detail", activity_id=activity_id)
    return RedirectResponse(url=str(url), status_code=303)


# ----- IMPORT / DELETE -----


@app.post("/import")
async def import_ropa(
    request: Request,
    file: UploadFile = File(...),
    mapper: str = Form("dictionary"),
    replace: bool = Form(False),
) -> RedirectResponse:
    """Import a ROPA workbook uploaded from the browser (block B1).

    The browser's native file picker runs on the client; the file is uploaded here
    as bytes, so the container never touches the client filesystem. The bytes are
    written to a temporary file and fed to the existing ingestion pipeline
    (:func:`~pii_detection.ropa.ingestion.pipeline.ingest_file`), then the selected
    mapper resolves the declared categories
    (:func:`~pii_detection.ropa.ingestion.pipeline.map_categories`).

    :param request: the incoming request, for building the redirect URL.
    :param file: the uploaded ``.ods``/``.xlsx`` register.
    :param mapper: ``"none"`` to skip, or a mapper name
        (``dictionary``/``llm``/``hybrid``) to resolve categories after ingesting;
        defaults to ``dictionary``.
    :param replace: if ``True``, wipe the register before importing (destructive);
        otherwise add to it, which fails if an activity id already exists.
    :returns: a 303 redirect to the index, with an ``?imported=N`` or ``?error=…``
        banner.
    :raises HTTPException: 400 if the file is not an ``.ods``/``.xlsx``.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".ods", ".xlsx"}:
        raise HTTPException(status_code=400, detail="carica un file .ods o .xlsx")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    db_url = str(get_repository().engine.url)
    try:
        activities = ingest_file(tmp_path, db_url, replace=replace)
        if mapper != "none":
            map_categories(get_repository(), build_mapper(mapper))
    except IntegrityError:
        return _redirect_index(
            request, error="Esistono già trattamenti con questi id: usa «sostituisci»."
        )
    finally:
        os.unlink(tmp_path)
    return _redirect_index(request, imported=len(activities))


@app.post("/activity/{activity_id}/delete")
def delete_activity_submit(request: Request, activity_id: str) -> RedirectResponse:
    """Delete a single processing activity from the register (block B1).

    :param request: the incoming request, for building the redirect URL.
    :param activity_id: identifier of the activity to delete; an unknown id is a
        no-op.
    :returns: a 303 redirect back to the index.
    """
    get_repository().delete_activity(activity_id)
    return RedirectResponse(url=str(request.url_for("index")), status_code=303)
