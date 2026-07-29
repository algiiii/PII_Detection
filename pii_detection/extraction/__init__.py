"""Text extraction layer (block B3): read documents into ``NormalizedDocument``.

The minimal, born-digital reader behind the detection pipeline. See
:mod:`pii_detection.extraction.extractor` for the implementation.
"""

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
]
