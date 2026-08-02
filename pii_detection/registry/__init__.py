"""Detected-PII registry (block B5): persist the PII found in documents.

Delta-based hybrid changelog --- a mutable current state (:class:`PIIInstance`)
plus an append-only log (:class:`PIIChange`) --- and **minimization-first**:
references only, never the PII values. Both steps are implemented: population and
the re-scan delta (:func:`diff_scan` → ``CONFIRMED``/``MOVED``/``NEW``/``REMOVED``).
"""

from pii_detection.registry.diff import ScanDiff, diff_scan
from pii_detection.registry.ingest import ingest_document
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import ChangeType, Document, PIIChange, PIIInstance, Scan

__all__ = [
    "ingest_document",
    "PIIRepository",
    "ScanDiff",
    "diff_scan",
    "ChangeType",
    "Document",
    "Scan",
    "PIIInstance",
    "PIIChange",
]
