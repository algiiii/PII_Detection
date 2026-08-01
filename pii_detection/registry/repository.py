"""Storage access for the detected-PII registry — block B5, persistence.

:class:`PIIRepository` is the single way in and out of the registry database.
Because the domain classes of :mod:`pii_detection.registry.types` are themselves
SQLModel tables (Active Record), there is no row/domain translation here.

The database URL is read from the ``PII_DB_URL`` environment variable, so the
registry lives in **its own** database, separate from the ROPA one; the same
image runs locally and in a container by changing only the configuration.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from sqlmodel import Session, SQLModel, create_engine, select

from pii_detection.detection.types import PIIMatch
from pii_detection.registry.types import ChangeType, Document, PIIChange, PIIInstance, Scan


class PIIRepository:
    """Read and write the detected-PII registry against a configured database.

    :ivar engine: SQLAlchemy engine bound to the resolved database URL.
    """

    def __init__(self, url: str | None = None) -> None:
        """Open the registry database, creating the schema if it does not exist.

        :param url: SQLAlchemy database URL; when ``None`` it is read from the
            ``PII_DB_URL`` environment variable, defaulting to the local file
            ``sqlite:///pii.db``.
        """
        url = url or os.environ.get("PII_DB_URL", "sqlite:///pii.db")
        self.engine = create_engine(url)
        SQLModel.metadata.create_all(self.engine)

    def record_scan(
        self,
        document_id: str,
        matches: Sequence[PIIMatch],
        *,
        path: str | None = None,
        replace: bool = False,
    ) -> Scan:
        """Persist one scan of a document and its detected PII (Step 1).

        Creates the :class:`~pii_detection.registry.types.Document` if new, a new
        :class:`~pii_detection.registry.types.Scan`, and for every match a
        :class:`~pii_detection.registry.types.PIIInstance` (**value dropped** —
        minimization) plus a :attr:`~pii_detection.registry.types.ChangeType.NEW`
        :class:`~pii_detection.registry.types.PIIChange`.

        :param document_id: identifier of the scanned document (its file stem).
        :param matches: the unified PII to record; their ``text`` is never stored.
        :param path: original document path, kept for reference.
        :param replace: if ``True``, drop the document's existing instances first,
            so a re-scan does not duplicate them (until the Step-2 delta lands).
        :returns: the created scan.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            if session.get(Document, document_id) is None:
                session.add(Document(document_id=document_id, path=path))
            if replace:
                existing = session.exec(
                    select(PIIInstance).where(PIIInstance.document_id == document_id)
                ).all()
                for instance in existing:
                    session.delete(instance)  # cascade removes its changes

            scan = Scan(document_id=document_id)
            session.add(scan)
            session.flush()  # assign scan.id

            for match in matches:
                instance = PIIInstance(
                    document_id=document_id,
                    pii_type=match.pii_type,
                    start=match.span.start,
                    end=match.span.end,
                    confidence=match.confidence,
                    confirmation_level=match.confirmation_level.value,
                    sources=[provenance.detector_id for provenance in match.sources],
                    last_scan_id=scan.id,
                )
                session.add(instance)
                session.flush()  # assign instance.id
                session.add(
                    PIIChange(
                        pii_instance_id=instance.id,
                        change_type=ChangeType.NEW,
                        scan_id=scan.id,
                        previous_scan_id=None,
                    )
                )

            session.commit()
            return scan

    def instances_for(self, document_id: str) -> list[PIIInstance]:
        """List the PII instances currently recorded for a document.

        :param document_id: identifier of the document.
        :returns: its instances, with their change history eagerly loaded.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            return list(
                session.exec(
                    select(PIIInstance).where(PIIInstance.document_id == document_id)
                ).all()
            )

    def clear(self) -> None:
        """Delete every document and its instances/changes from the registry.

        Destructive: leaves an empty but initialized database. Deleting a document
        cascades to its instances and their changes; the scans are then removed.
        """
        with Session(self.engine) as session:
            for document in session.exec(select(Document)).all():
                session.delete(document)
            for scan in session.exec(select(Scan)).all():
                session.delete(scan)
            session.commit()


__all__ = ["PIIRepository"]
