"""Text extraction layer (block B3): read documents into ``NormalizedDocument``.

The minimal, born-digital reader behind the detection pipeline. See
:mod:`pii_detection.extraction.extractor` for the implementation, and
:mod:`pii_detection.extraction.dates` for the reference date a document is dated
from, which the retention check (B7) works on.
"""

from pii_detection.extraction.dates import DateSource, ReferenceDate, reference_date
from pii_detection.extraction.extractor import (
    UnsupportedFormatError,
    extract_document,
    normalize_text,
    supported_suffixes,
)

__all__ = [
    "extract_document",
    "normalize_text",
    "supported_suffixes",
    "UnsupportedFormatError",
    "DateSource",
    "ReferenceDate",
    "reference_date",
]
