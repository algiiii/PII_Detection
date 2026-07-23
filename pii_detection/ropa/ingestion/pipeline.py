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

from pii_detection.ropa.ingestion.category_mapper import CategoryMapper
from pii_detection.ropa.ingestion.sheet_reader import read_sheet, sheet_names
from pii_detection.ropa.ingestion.normalizer import normalize
from pii_detection.ropa.repository import ROPARepository
from pii_detection.ropa.types import MappingState, ProcessingActivity


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


def map_categories(repository: ROPARepository, mapper: CategoryMapper) -> int:
    """Resolve the still-unmapped declared categories, splitting each in place.

    Separate pass, run after :func:`ingest_file`: for every declared category
    that is still ``PROPOSED`` with no ``pii_types``, run the mapper on its free
    text and, when that yields a real split or a resolution, replace it with the
    resulting sub-categories via
    :meth:`~pii_detection.ropa.repository.ROPARepository.split_category`.

    The pass is idempotent: a single phrase the mapper cannot resolve is left
    untouched (splitting it would only reproduce itself), so re-running maps only
    what is new. Confirmed or already-resolved categories are never touched.

    :param repository: the register to read from and write back to.
    :param mapper: the strategy resolving free text onto ``pii_type`` ids.
    :returns: the number of declared categories that were split.
    """
    split_count = 0
    for activity in repository.load():
        for macro in activity.macro_categories:
            for category in macro.categories:
                if category.id is None or category.pii_types:
                    continue
                if category.mapping_state is not MappingState.PROPOSED:
                    continue
                mapped = mapper.map(category.raw_text)
                if not mapped or (len(mapped) == 1 and not mapped[0].pii_types):
                    continue  # nothing to resolve or split — leave it for the DPO
                repository.split_category(
                    category.id, [(mc.text, list(mc.pii_types)) for mc in mapped]
                )
                split_count += 1
    return split_count
