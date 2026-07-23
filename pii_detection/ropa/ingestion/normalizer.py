"""Map a CNIL spreadsheet sheet onto the ROPA domain model — block B1.

The interpretive step of the ingestion: it takes the raw cell grid produced by
:func:`~pii_detection.ropa.ingestion.sheet_reader.read_sheet` and builds one
:class:`~pii_detection.ropa.types.ProcessingActivity` from it.

Only the "Categories of personal data" section of the sheet is modelled; the
activity's identity (name, purpose) is read from two labelled cells, and every
other block of the CNIL form is intentionally ignored. Rows are located by their
label (the first cell), not by a fixed index, so the mapping survives layout
changes.

The single categories (level 3) are kept as raw text with an empty ``pii_types``
and :attr:`~pii_detection.ropa.types.MappingState.PROPOSED`: splitting them and
resolving them onto the ``pii_type`` catalog is the job of the AI category mapper
and the DPO's confirmation, not of this deterministic step.
"""

from pii_detection.ropa.ingestion.retention import parse_retention
from pii_detection.ropa.types import (
    DeclaredCategory,
    DeclaredMacroCategory,
    MappingState,
    ProcessingActivity,
)

_NAME_LABEL = "Name of the processing operation"
_PURPOSE_LABEL = "Main purpose"
_CATEGORIES_HEADER = "Categories of personal data"


def _find_value(rows: list[list[str]], label: str) -> str:
    """Return the second cell of the first row whose first cell equals ``label``.

    :param rows: the sheet grid.
    :param label: the row label to look up, matched on the first cell.
    :returns: the value cell, stripped, or ``""`` if the label is absent.
    """
    for row in rows:
        if row and row[0].strip() == label:
            return row[1].strip() if len(row) > 1 else ""
    return ""


def _slugify(text: str) -> str:
    """Build a stable, lowercase, hyphenated id from a free-text name.

    :param text: the source text, typically the activity name.
    :returns: a slug such as ``"payroll-management"``; ``"activity"`` when empty.
    """
    slug = "-".join(text.lower().split())
    return slug or "activity"


def normalize(rows: list[list[str]]) -> ProcessingActivity:
    """Build a processing activity from one CNIL sheet grid.

    Reads the activity identity from its labelled cells and the declared data
    categories from the "Categories of personal data" section, mapping each
    section row onto a macro category (with its retention) holding a single raw
    child category.

    :param rows: the sheet grid, as returned by
        :func:`~pii_detection.ropa.ingestion.sheet_reader.read_sheet`.
    :returns: the normalized :class:`~pii_detection.ropa.types.ProcessingActivity`.
    :raises ValueError: if the sheet has no "Categories of personal data" section.
    """
    name = _find_value(rows, _NAME_LABEL)
    purpose = _find_value(rows, _PURPOSE_LABEL)

    start = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == _CATEGORIES_HEADER:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"section not found: {_CATEGORIES_HEADER!r}")

    macro_categories: list[DeclaredMacroCategory] = []
    for row in rows[start:]:
        if not row or not row[0].strip():
            break
        retention_text = row[2].strip() if len(row) > 2 else ""
        macro_categories.append(
            DeclaredMacroCategory(
                raw_text=row[0].strip(),
                retention_text=retention_text,
                retention_months=parse_retention(retention_text),
                categories=[
                    DeclaredCategory(
                        raw_text=row[1].strip() if len(row) > 1 else "",
                        pii_types=[],
                        mapping_state=MappingState.PROPOSED,
                    )
                ],
            )
        )

    return ProcessingActivity(
        id=_slugify(name),
        name=name,
        purpose=purpose,
        macro_categories=macro_categories,
    )


__all__ = ["normalize"]
