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

from pii_detection.ropa.ingestion.sheet_reader import read_sheet
from pii_detection.ropa.ingestion.normalizer import normalize
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import ProcessingActivity


def ingest_file(
    path: str | Path,
    db_url: str,
    sheet_name: str = "4_-_Example_",
    *,
    replace: bool = False,
) -> list[ProcessingActivity]:
    """Read one spreadsheet ROPA sheet, normalize it and save it to the database.

    :param path: path to the ``.ods`` or ``.xlsx`` register.
    :param db_url: SQLAlchemy database URL to save into (e.g.
        ``"sqlite:///ropa.db"``).
    :param sheet_name: name of the sheet to ingest.
    :param replace: if ``True``, wipe the existing register before saving
        (destructive); if ``False``, add to it, which fails on an activity ``id``
        that already exists.
    :returns: the normalized processing activities that were persisted.
    """
    activity = normalize(read_sheet(path, sheet_name))
    repository = ROPARepository(db_url)
    if replace:
        repository.clear()
    repository.save([activity])
    return [activity]
