"""DPO review web app for the ROPA — browse and confirm (block B1).

A small FastAPI + Jinja2 application that lets the DPO browse the ingested
register as a three-level tree (activity → macro category → declared category)
and confirm the mapping of each declared category onto the ``pii_type`` catalog.
The structure itself comes from the ingestion of the CNIL register; the app only
reviews it, so there is no create/delete of activities here.

It is the "interface on a port": the database stays a file behind the app,
reached only through :class:`~pii_detection.ropa.repository.ROPARepository`. The
database URL is read from the ``ROPA_DB_URL`` environment variable and defaults
to ``sqlite:///ropa.db``. Every write follows the Post/Redirect/Get pattern (a
``303`` redirect after each ``POST``), so a browser refresh never re-submits a
form.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import MappingState

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

app = FastAPI(title="ROPA review")


def get_repository() -> ROPARepository:
    """Open the repository pointed to by ``ROPA_DB_URL``.

    :returns: a repository bound to the configured database URL.
    """
    return ROPARepository(os.environ.get("ROPA_DB_URL", "sqlite:///ropa.db"))


# ----- READ -----


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """List every processing activity in the register.

    :param request: the incoming request (required by the template engine).
    :returns: the rendered activity list.
    """
    activities = get_repository().load()
    return _TEMPLATES.TemplateResponse(request, "index.html", {"activities": activities})


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
