"""Data model of the ROPA (Record of Processing Activities) — block B1.

Design and rationale in ``doc/sections/5_architettura.tex`` (§``sec:modello-ropa``).
The flat CNIL sheet is normalized around :class:`ProcessingActivity`, gathered by
the :class:`ROPA` container. The model is **technology-agnostic**: it is the
in-memory structure a persistence layer will store, but the storage technology is
deliberately deferred.

Two design choices make the structure fit the compliance-checking purpose (B7)
rather than merely mirroring the source document:

1. **Shared vocabulary (bridge to B4).** :class:`DeclaredDataCategory` resolves a
   declared category onto the same ``pii_type`` catalog used by the detection
   layer, one-to-many, so comparing what the ROPA *declares* with what the engine
   *finds* becomes a set operation. The AI proposes the mapping, the DPO confirms
   it (:class:`MappingState`).
2. **Machine-computable retention.** :class:`Retention` stores a structured
   ``duration_months`` next to the original wording, so the "past retention?"
   check is computable rather than a prose read.

This first version keeps the remaining CNIL fields (data subjects, recipients,
transfers, security measures, information systems) as plain strings; they become
richer entities only once the ingestion actually needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MappingState(str, Enum):
    """Confirmation state of a declared-category → ``pii_type`` mapping.

    :cvar PROPOSED: mapping produced by the selective AI, not yet validated.
    :cvar CONFIRMED: mapping validated by the DPO; safe to use for compliance.
    """

    PROPOSED = "proposed"
    CONFIRMED = "confirmed"


# ----------------- Parte nuova -------------------
@dataclass
class DeclaredCategory:
    """Atomic category directly found in ROPA or singularly splitted or expanded by rules / AI
    """
    raw_text: str
    pii_types: tuple[str, ...] = ()
    mapping_state: MappingState = MappingState.PROPOSED # viene affidato ad automatismi e AI, ma il DPO controlla sempre

@dataclass
class DeclaredMacroCategory:
    """Container for different DeclaredCategory s with common retention policy
    """
    raw_text: str
    retention_text: str
    retention_date: int | None # DA METTERE DATETIME

@dataclass
class ProcessingActivity:
    """General container of different MacroCategories
    """
    id: str
    name: str
    purpose: str

__all__ = [
    "MappingState",
    "DeclaredCategory",
    "DeclaredMacroCategory",
    "ProcessingActivity",
]