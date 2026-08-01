"""Detected-PII registry (block B5): persist the PII found in documents.

Delta-based hybrid changelog --- a mutable current state (:class:`PIIInstance`)
plus an append-only log (:class:`PIIChange`) --- and **minimization-first**:
references only, never the PII values. Step 1 (population) is implemented; the
re-scan delta (``CONFIRMED``/``MOVED``/``REMOVED``) is Step 2.
"""

from pii_detection.registry.ingest import ingest_document
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import ChangeType, Document, PIIChange, PIIInstance, Scan

__all__ = [
    "ingest_document",
    "PIIRepository",
    "ChangeType",
    "Document",
    "Scan",
    "PIIInstance",
    "PIIChange",
]
