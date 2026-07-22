"""End-to-end ROPA ingestion: Excel file → normalized ROPA → database.

Ties the ingestion pieces together:
:func:`~pii_detection.ropa.ingestion.excel_reader.read_records` reads the
spreadsheet, :func:`~pii_detection.ropa.ingestion.normalizer.normalize` maps it
onto the domain model, and
:class:`~pii_detection.ropa.repository.ROPARepository` persists it.

It is the standalone B1 pipeline; the database URL is configurable so the same
command runs locally and inside a container (see ``__main__``).
"""

from pathlib import Path

from pii_detection.ropa.ingestion.excel_reader import read_records
from pii_detection.ropa.ingestion.normalizer import normalize
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import ProcessingActivity


def ingest_file(xlsx_path: str | Path, db_url: str, *, replace: bool = False) -> list[ProcessingActivity]:
    """Read an Excel ROPA, normalize it and save it to the database.

    :param xlsx_path: path to the ``.xlsx`` register.
    :param db_url: SQLAlchemy database URL to save into (e.g.
        ``"sqlite:///ropa.db"``).
    :param replace: if ``True``, wipe the existing register before saving
        (destructive); if ``False``, add to it, which fails on an ``activity_id``
        that already exists.
    :returns: the normalized processing activities that were persisted
    """
    activities = normalize(read_records(xlsx_path))
    repository = ROPARepository(db_url)
    if replace:
        repository.clear()
    repository.save(activities)
    return activities
