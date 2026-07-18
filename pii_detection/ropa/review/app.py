"""DPO review web app for the ROPA — browse and edit (block B1).

A small FastAPI + Jinja2 application that lets the DPO browse the ingested
register **and** correct it: create, edit and delete processing activities,
their declared data categories and their retention rules. It is the "interface
on a port": the DB stays a file behind the app, accessed only through
:class:`~pii_detection.ropa.persistence.repository.ROPARepository`.

The database URL is read from the ``ROPA_DB_URL`` environment variable and
defaults to ``sqlite:///ropa.db``. Every write follows the Post/Redirect/Get
pattern (a ``303`` redirect after each ``POST``), so a browser refresh never
re-submits a form.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pii_detection.ropa.persistence.repository import ROPARepository
from pii_detection.ropa.types import MappingState

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

app = FastAPI(title="ROPA review")


def get_repository() -> ROPARepository:
    """Open the repository pointed to by ``ROPA_DB_URL``.

    :returns: a repository bound to the configured database URL.
    """
    return ROPARepository(os.environ.get("ROPA_DB_URL", "sqlite:///ropa.db"))


def _parse_list(text: str) -> list[str]:
    """Parse a textarea into a list of values.

    Splits on newlines and semicolons, strips each item and drops empty ones.

    :param text: raw textarea content.
    :returns: the non-empty, stripped values.
    """
    items: list[str] = []
    for line in text.replace(";", "\n").splitlines():
        piece = line.strip()
        if piece:
            items.append(piece)
    return items


def _parse_months(value: str) -> int | None:
    """Parse the retention duration field: empty means "criterion" (``None``).

    :param value: raw form value.
    :returns: the duration in months, or ``None`` if the field is blank.
    :raises HTTPException: 400 if the value is not a valid integer.
    """
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid duration_months: {value!r}") from None


# ----- READ -----


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """List every processing activity in the register.

    :param request: the incoming request (required by the template engine).
    :returns: the rendered activity list.
    """
    ropa = get_repository().load_ropa()
    return _TEMPLATES.TemplateResponse(
        request, "index.html", {"activities": ropa.activities}
    )


@app.get("/activity/{activity_id}", response_class=HTMLResponse)
def activity_detail(request: Request, activity_id: str) -> HTMLResponse:
    """Show the full detail of one processing activity, with edit forms.

    The child rows (categories, retentions) are passed as persistence rows, not
    domain objects, because the forms need their database ``id``.

    :param request: the incoming request.
    :param activity_id: identifier of the activity to display.
    :returns: the rendered activity detail.
    :raises HTTPException: 404 if no activity has that id.
    """
    repo = get_repository()
    activity = repo.load_ropa().activity(activity_id)
    if activity is None:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}")
    return _TEMPLATES.TemplateResponse(
        request,
        "activity.html",
        {
            "a": activity,
            "categories": repo.list_category_rows(activity_id),
            "retentions": repo.list_retention_rows(activity_id),
            "catalog_ids": repo.catalog_ids(),
        },
    )


# ----- ACTIVITY: create / edit / delete -----


@app.get("/new", response_class=HTMLResponse)
def new_activity_form(request: Request) -> HTMLResponse:
    """Show the empty form to create a new activity.

    :param request: the incoming request.
    :returns: the rendered create form.
    """
    return _TEMPLATES.TemplateResponse(request, "activity_form.html", {"a": None})


@app.get("/activity/{activity_id}/edit", response_class=HTMLResponse)
def edit_activity_form(request: Request, activity_id: str) -> HTMLResponse:
    """Show the pre-filled form to edit an activity's scalar and list fields.

    :param request: the incoming request.
    :param activity_id: identifier of the activity to edit.
    :returns: the rendered edit form.
    :raises HTTPException: 404 if no activity has that id.
    """
    activity = get_repository().load_ropa().activity(activity_id)
    if activity is None:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}")
    return _TEMPLATES.TemplateResponse(request, "activity_form.html", {"a": activity})


@app.post("/activity")
def create_activity_submit(
    name: str = Form(...),
    purpose: str = Form(...),
    legal_basis: str = Form(...),
    controller: str = Form(...),
    dpo: str = Form(""),
    data_subjects: str = Form(""),
    recipients: str = Form(""),
    third_country_transfers: str = Form(""),
    security_measures: str = Form(""),
    information_systems: str = Form(""),
) -> RedirectResponse:
    """Create a new activity from the form and redirect to its detail page.

    :returns: a 303 redirect to the new activity's detail page.
    """
    activity_id = get_repository().create_activity(
        name=name,
        purpose=purpose,
        legal_basis=legal_basis,
        controller=controller,
        dpo=dpo or None,
        data_subjects=_parse_list(data_subjects),
        recipients=_parse_list(recipients),
        third_country_transfers=_parse_list(third_country_transfers),
        security_measures=_parse_list(security_measures),
        information_systems=_parse_list(information_systems),
    )
    return RedirectResponse(url=f"/activity/{activity_id}", status_code=303)


@app.post("/activity/{activity_id}")
def update_activity_submit(
    activity_id: str,
    name: str = Form(...),
    purpose: str = Form(...),
    legal_basis: str = Form(...),
    controller: str = Form(...),
    dpo: str = Form(""),
    data_subjects: str = Form(""),
    recipients: str = Form(""),
    third_country_transfers: str = Form(""),
    security_measures: str = Form(""),
    information_systems: str = Form(""),
) -> RedirectResponse:
    """Update an activity from the form and redirect to its detail page.

    :param activity_id: identifier of the activity to update.
    :returns: a 303 redirect to the activity's detail page.
    :raises HTTPException: 404 if no activity has that id.
    """
    try:
        get_repository().update_activity(
            activity_id,
            {
                "name": name,
                "purpose": purpose,
                "legal_basis": legal_basis,
                "controller": controller,
                "dpo": dpo or None,
                "data_subjects": _parse_list(data_subjects),
                "recipients": _parse_list(recipients),
                "third_country_transfers": _parse_list(third_country_transfers),
                "security_measures": _parse_list(security_measures),
                "information_systems": _parse_list(information_systems),
            },
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}") from None
    return RedirectResponse(url=f"/activity/{activity_id}", status_code=303)


@app.post("/activity/{activity_id}/delete")
def delete_activity_submit(activity_id: str) -> RedirectResponse:
    """Delete an activity (and its children) and redirect to the register.

    :param activity_id: identifier of the activity to delete.
    :returns: a 303 redirect to the activity list.
    :raises HTTPException: 404 if no activity has that id.
    """
    try:
        get_repository().delete_activity(activity_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}") from None
    return RedirectResponse(url="/", status_code=303)


# ----- CATEGORY: add / update / delete -----


@app.post("/activity/{activity_id}/category")
def add_category_submit(
    activity_id: str,
    raw_text: str = Form(...),
    pii_types: list[str] = Form([]),
    mapping_state: str = Form(MappingState.PROPOSED.value),
) -> RedirectResponse:
    """Add a declared data category to an activity.

    :param activity_id: activity the category belongs to.
    :returns: a 303 redirect to the activity's detail page.
    :raises HTTPException: 404 if the activity is unknown, 400 on a bad value.
    """
    try:
        get_repository().add_category(
            activity_id, raw_text, tuple(pii_types), MappingState(mapping_state)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return RedirectResponse(url=f"/activity/{activity_id}", status_code=303)


@app.post("/category/{category_id}")
def update_category_submit(
    category_id: int,
    activity_id: str = Form(...),
    raw_text: str = Form(...),
    pii_types: list[str] = Form([]),
    mapping_state: str = Form(MappingState.PROPOSED.value),
) -> RedirectResponse:
    """Update a declared data category and redirect to the parent activity.

    :param category_id: id of the category row to update.
    :param activity_id: parent activity, carried by the form for the redirect.
    :returns: a 303 redirect to the activity's detail page.
    :raises HTTPException: 404 if the category is unknown, 400 on a bad value.
    """
    try:
        get_repository().update_category(
            category_id, raw_text, tuple(pii_types), MappingState(mapping_state)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown category: {category_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return RedirectResponse(url=f"/activity/{activity_id}", status_code=303)


@app.post("/category/{category_id}/delete")
def delete_category_submit(category_id: int) -> RedirectResponse:
    """Delete a declared data category and redirect to the parent activity.

    :param category_id: id of the category row to delete.
    :returns: a 303 redirect to the parent activity's detail page.
    :raises HTTPException: 404 if the category is unknown.
    """
    try:
        activity_id = get_repository().delete_category(category_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown category: {category_id}") from None
    return RedirectResponse(url=f"/activity/{activity_id}", status_code=303)


# ----- RETENTION: add / update / delete -----


@app.post("/activity/{activity_id}/retention")
def add_retention_submit(
    activity_id: str,
    raw_text: str = Form(...),
    duration_months: str = Form(""),
) -> RedirectResponse:
    """Add a retention rule to an activity.

    :param activity_id: activity the retention belongs to.
    :returns: a 303 redirect to the activity's detail page.
    :raises HTTPException: 404 if the activity is unknown, 400 on a bad value.
    """
    try:
        get_repository().add_retention(activity_id, raw_text, _parse_months(duration_months))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return RedirectResponse(url=f"/activity/{activity_id}", status_code=303)


@app.post("/retention/{retention_id}")
def update_retention_submit(
    retention_id: int,
    activity_id: str = Form(...),
    raw_text: str = Form(...),
    duration_months: str = Form(""),
) -> RedirectResponse:
    """Update a retention rule and redirect to the parent activity.

    :param retention_id: id of the retention row to update.
    :param activity_id: parent activity, carried by the form for the redirect.
    :returns: a 303 redirect to the activity's detail page.
    :raises HTTPException: 404 if the retention is unknown, 400 on a bad value.
    """
    try:
        get_repository().update_retention(retention_id, raw_text, _parse_months(duration_months))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown retention: {retention_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return RedirectResponse(url=f"/activity/{activity_id}", status_code=303)


@app.post("/retention/{retention_id}/delete")
def delete_retention_submit(retention_id: int) -> RedirectResponse:
    """Delete a retention rule and redirect to the parent activity.

    :param retention_id: id of the retention row to delete.
    :returns: a 303 redirect to the parent activity's detail page.
    :raises HTTPException: 404 if the retention is unknown.
    """
    try:
        activity_id = get_repository().delete_retention(retention_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown retention: {retention_id}") from None
    return RedirectResponse(url=f"/activity/{activity_id}", status_code=303)
