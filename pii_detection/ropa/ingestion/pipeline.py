"""End-to-end ROPA ingestion: spreadsheet sheet → normalized activity → database.

Ties the ingestion pieces together:
:func:`~pii_detection.ropa.ingestion.sheet_reader.read_sheet` reads one sheet
(ODS or Excel) into a grid, :func:`~pii_detection.ropa.ingestion.normalizer.normalize`
maps it onto the domain model, and
:class:`~pii_detection.ropa.repository.ROPARepository` persists it.

It is the standalone B1 pipeline; the database URL is configurable so the same
command runs locally and inside a container (see ``__main__``).
"""

from pathlib import Path

from pii_detection.ropa.ingestion.sheet_reader import read_sheet, sheet_names
from pii_detection.ropa.ingestion.normalizer import normalize
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import ProcessingActivity


def ingest_file(
    path: str | Path,
    db_url: str,
    *,
    replace: bool = False,
) -> list[ProcessingActivity]:
    """Read every processing-activity sheet of a workbook and save them.

    Iterates all tabs of the workbook and normalizes each; a tab that is not a
    processing-activity record (a tutorial, a list, the blank template) makes
    :func:`~pii_detection.ropa.ingestion.normalizer.normalize` raise
    :class:`ValueError` and is skipped. So a single CNIL workbook yields one
    :class:`~pii_detection.ropa.types.ProcessingActivity` per real activity sheet.

    :param path: path to the ``.ods`` or ``.xlsx`` register.
    :param db_url: SQLAlchemy database URL to save into (e.g.
        ``"sqlite:///ropa.db"``).
    :param replace: if ``True``, wipe the existing register before saving
        (destructive); if ``False``, add to it, which fails on an activity ``id``
        that already exists.
    :returns: the normalized processing activities that were persisted.
    """
    activities: list[ProcessingActivity] = []
    for name in sheet_names(path):
        try:
            activities.append(normalize(read_sheet(path, name)))
        except ValueError:
            continue  # not a processing-activity sheet — skip it
    repository = ROPARepository(db_url)
    if replace:
        repository.clear()
    repository.save(activities)
    return activities
