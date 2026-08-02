"""Compliance block: document–activity association (B6) and the check (B7).

Connects the two halves of the system — the declared side (ROPA, B1) and the
detected side (registry, B5) — into the first end-to-end compliance verdict:

- :mod:`~pii_detection.compliance.assign` — associate a document with the
  processing activities it belongs to (explicit, DPO-driven; B6);
- :mod:`~pii_detection.compliance.checker` — compare declared vs detected and
  produce a :class:`~pii_detection.compliance.types.ComplianceReport` (B7).
"""

from pii_detection.compliance.assign import (
    ActivityAssigner,
    ExplicitAssigner,
    persist_assignment,
)
from pii_detection.compliance.checker import CheckResult, build_report, check_document
from pii_detection.compliance.types import ComplianceReport, RetentionFlag, format_report

__all__ = [
    "ActivityAssigner",
    "ExplicitAssigner",
    "persist_assignment",
    "CheckResult",
    "build_report",
    "check_document",
    "ComplianceReport",
    "RetentionFlag",
    "format_report",
]
