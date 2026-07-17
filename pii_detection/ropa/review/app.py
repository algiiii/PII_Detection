"""DPO review web app for the ROPA — read-only visualization (block B1).

A small FastAPI + Jinja2 application that lets the DPO browse the ingested
register. It is the "interface on a port": the DB stays a file behind the app,
accessed only through :class:`~pii_detection.ropa.persistence.repository.
ROPARepository`.

The database URL is read from the ``ROPA_DB_URL`` environment variable and
defaults to ``sqlite:///ropa.db``. This step is view-only; editing (add / remove
/ modify) comes next.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pii_detection.ropa.persistence.repository import ROPARepository

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

app = FastAPI(title="ROPA review")


def get_repository() -> ROPARepository:
    """Open the repository pointed to by ``ROPA_DB_URL``.

    :returns: a repository bound to the configured database URL.
    """
    return ROPARepository(os.environ.get("ROPA_DB_URL", "sqlite:///ropa.db"))


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
    """Show the full detail of one processing activity.

    :param request: the incoming request.
    :param activity_id: identifier of the activity to display.
    :returns: the rendered activity detail.
    :raises HTTPException: 404 if no activity has that id.
    """
    activity = get_repository().load_ropa().activity(activity_id)
    if activity is None:
        raise HTTPException(status_code=404, detail=f"unknown activity: {activity_id}")
    return _TEMPLATES.TemplateResponse(request, "activity.html", {"a": activity})
