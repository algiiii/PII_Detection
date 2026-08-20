"""Detected-PII registry (block B5): persist the PII found in documents.

Holds the **current state** of the PII detected per document
(:class:`PIIInstance`) — each scan fully replaces a document's instances — and is
**minimization-first**: references only, never the PII values.
"""

from pii_detection.registry.ingest import ingest_document
from pii_detection.registry.repository import PIIRepository
from pii_detection.registry.types import Document, PIIInstance, Scan

__all__ = [
    "ingest_document",
    "PIIRepository",
    "Document",
    "Scan",
    "PIIInstance",
]
