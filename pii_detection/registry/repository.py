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
from collections.abc import Mapping, Sequence
from datetime import datetime

from sqlmodel import Session, SQLModel, col, create_engine, select

from pii_detection.detection.types import PIIMatch
from pii_detection.registry.diff import diff_scan
from pii_detection.registry.folder_rules import ApplyRulesResult, match_activities
from pii_detection.registry.types import (
    AssociationSource,
    ChangeType,
    Document,
    FolderRule,
    PIIChange,
    PIIInstance,
    Scan,
)


def _instance_from_match(document_id: str, match: PIIMatch, scan_id: int | None) -> PIIInstance:
    """Build a minimized instance from a match (the value is deliberately dropped)."""
    return PIIInstance(
        document_id=document_id,
        pii_type=match.pii_type,
        start=match.span.start,
        end=match.span.end,
        confidence=match.confidence,
        confirmation_level=match.confirmation_level.value,
        sources=[provenance.detector_id for provenance in match.sources],
        last_scan_id=scan_id,
    )


class PIIRepository:
    """Read and write the detected-PII registry against a configured database.

    :ivar engine: SQLAlchemy engine bound to the resolved database URL.
    """

    def __init__(self, url: str | None = None) -> None:
        """Open the registry database, creating the schema if it does not exist.

        :param url: SQLAlchemy database URL; when ``None`` it is read from the
            ``PII_DB_URL`` environment variable, defaulting to the canonical local
            store ``sqlite:///data/pii.db`` (the ``data/`` directory, shared with
            the container via the repo bind-mount).
        """
        url = url or os.environ.get("PII_DB_URL", "sqlite:///data/pii.db")
        self.engine = create_engine(url)
        # Create only the registry tables: the SQLModel metadata is shared across
        # the whole process, so an unfiltered create_all() would also materialize
        # the ROPA tables (processing_activity/macro_category/declared_category) in
        # this database. The two registers live in separate databases and share no
        # physical foreign key.
        metadata = SQLModel.metadata
        metadata.create_all(
            self.engine,
            tables=[
                metadata.tables[name]
                for name in (
                    "registry_document",
                    "registry_scan",
                    "registry_pii_instance",
                    "registry_pii_change",
                    "registry_folder_rule",
                )
            ],
        )

    def record_scan(
        self,
        document_id: str,
        matches: Sequence[PIIMatch],
        *,
        path: str | None = None,
        source_modified_at: datetime | None = None,
        replace: bool = False,
    ) -> Scan:
        """Persist one scan of a document, computing the delta against its state.

        Creates the :class:`~pii_detection.registry.types.Document` if new and a
        new :class:`~pii_detection.registry.types.Scan`, then compares the matches
        with the document's current instances (:func:`diff_scan`) and applies the
        outcome: ``CONFIRMED`` (touch the last scan), ``MOVED`` (update position),
        ``NEW`` (create instance) and ``REMOVED`` (mark the instance gone). Every
        change links to the previous scan of the same document (``None`` on the
        first scan). Instances never store the PII value (minimization).

        :param document_id: identifier of the scanned document (its file stem).
        :param matches: the unified PII of this scan; their ``text`` is never stored.
        :param path: original document path, kept for reference.
        :param source_modified_at: last-modified time of the source file, stored on
            the document as the reference date for the retention check (B7); updated
            on every scan when provided.
        :param replace: if ``True``, wipe the document's history first and record
            this scan as a fresh bootstrap (all ``NEW``).
        :returns: the created scan.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            document = session.get(Document, document_id)
            if document is None:
                session.add(
                    Document(
                        document_id=document_id,
                        path=path,
                        source_modified_at=source_modified_at,
                    )
                )
            elif source_modified_at is not None:
                document.source_modified_at = source_modified_at

            all_instances = session.exec(
                select(PIIInstance).where(PIIInstance.document_id == document_id)
            ).all()
            if replace:
                for instance in all_instances:
                    session.delete(instance)  # cascade removes its changes
                all_instances = []

            previous = session.exec(
                select(Scan)
                .where(Scan.document_id == document_id)
                .order_by(col(Scan.id).desc())
            ).first()
            previous_scan_id = previous.id if previous is not None else None

            scan = Scan(document_id=document_id)
            session.add(scan)
            session.flush()  # assign scan.id

            current = [instance for instance in all_instances if not instance.removed]
            delta = diff_scan(current, matches)

            for instance, _match in delta.confirmed:
                instance.last_scan_id = scan.id
                self._log(session, instance, ChangeType.CONFIRMED, scan.id, previous_scan_id)

            for instance, match in delta.moved:
                instance.start = match.span.start
                instance.end = match.span.end
                instance.confidence = match.confidence
                instance.confirmation_level = match.confirmation_level.value
                instance.sources = [p.detector_id for p in match.sources]
                instance.last_scan_id = scan.id
                self._log(session, instance, ChangeType.MOVED, scan.id, previous_scan_id)

            for match in delta.new:
                instance = _instance_from_match(document_id, match, scan.id)
                session.add(instance)
                session.flush()  # assign instance.id
                self._log(session, instance, ChangeType.NEW, scan.id, previous_scan_id)

            for instance in delta.removed:
                instance.removed = True
                instance.last_scan_id = scan.id
                self._log(session, instance, ChangeType.REMOVED, scan.id, previous_scan_id)

            session.commit()
            return scan

    @staticmethod
    def _log(
        session: Session,
        instance: PIIInstance,
        change_type: ChangeType,
        scan_id: int | None,
        previous_scan_id: int | None,
    ) -> None:
        """Append a change-log entry for an instance."""
        session.add(
            PIIChange(
                pii_instance_id=instance.id,
                change_type=change_type,
                scan_id=scan_id,
                previous_scan_id=previous_scan_id,
            )
        )

    def instances_for(
        self, document_id: str, *, include_removed: bool = False
    ) -> list[PIIInstance]:
        """List the PII instances recorded for a document.

        :param document_id: identifier of the document.
        :param include_removed: also return instances marked removed; by default
            only the current state (present PII) is returned.
        :returns: the instances, with their change history eagerly loaded.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            instances = session.exec(
                select(PIIInstance).where(PIIInstance.document_id == document_id)
            ).all()
            if include_removed:
                return list(instances)
            return [instance for instance in instances if not instance.removed]

    def documents(self) -> list[Document]:
        """List every recorded document, for the dashboard overview (block B8).

        :returns: all documents in the registry, each with its activity assignment
            and reference date; the PII instances are not eagerly loaded here.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            return list(session.exec(select(Document)).all())

    def get_document(self, document_id: str) -> Document | None:
        """Load a document by id, with its activity assignment and reference date.

        :param document_id: identifier of the document.
        :returns: the :class:`~pii_detection.registry.types.Document`, or ``None``
            if it was never recorded.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            return session.get(Document, document_id)

    def assign_activities(self, document_id: str, activity_ids: Sequence[str], *, source: AssociationSource = AssociationSource.MANUAL) -> None:
        """Assign a document to the processing activities it belongs to (block B6).

        Explicit, DPO-driven association: the ids are stored as-is on the document
        (:attr:`~pii_detection.registry.types.Document.activity_ids`). They
        reference :class:`~pii_detection.ropa.types.ProcessingActivity` in the ROPA
        database; there is no cross-database foreign key, so existence is not
        checked here — the compliance check (B7) reports an id it cannot resolve.

        :param document_id: identifier of an already-recorded document.
        :param activity_ids: the activities to associate; must be non-empty and
            contain no blank id.
        :param source: where the association comes from —
            :attr:`~pii_detection.registry.types.AssociationSource.MANUAL` (explicit
            DPO input, the default) or ``RULE`` (derived by
            :meth:`apply_folder_rules`). Stored so rule application can skip
            documents a human associated (manual wins).
        :raises KeyError: if the document was never recorded.
        :raises ValueError: if ``activity_ids`` is empty or holds a blank id.
        """
        ids = list(activity_ids)
        if not ids or any(not str(activity_id).strip() for activity_id in ids):
            raise ValueError("activity_ids must be non-empty and contain no blank id")
        with Session(self.engine) as session:
            document = session.get(Document, document_id)
            if document is None:
                raise KeyError(document_id)
            document.activity_ids = ids
            document.association_source = source
            session.commit()

    def folder_rules(self) -> list[FolderRule]:
        with Session(self.engine, expire_on_commit = False) as session:
            statement = select(FolderRule).order_by(col(FolderRule.prefix))
            return list(session.exec(statement).all())

    def save_rule(self, prefix: str, activity_ids: Sequence[str]) -> FolderRule:

        ids = list(activity_ids)
        if not ids or any(not str(activity_id).strip() for activity_id in ids):
            raise ValueError("activity_ids must be non-empty and containt no blank id")
        normalized = prefix.strip().strip("/")
        with Session(self.engine, expire_on_commit=False) as session:
            rule = session.get(FolderRule, normalized)
            if rule is None:
                rule = FolderRule(prefix=normalized, activity_ids=ids)
                session.add(rule)
            else:
                rule.activity_ids = ids
            session.commit()
            session.refresh(rule)
            return rule

    def delete_rule(self, prefix: str) -> None:
        normalized = prefix.strip().strip("/")
        with Session(self.engine) as session:
            rule = session.get(FolderRule, normalized)
            if rule is not None:
                session.delete(rule)
                session.commit()

    def apply_folder_rules(self) -> ApplyRulesResult:
        rules = self.folder_rules()
        result = ApplyRulesResult()
        with Session(self.engine) as session:
            for document in session.exec(select(Document)).all():
                if document.association_source == AssociationSource.MANUAL:
                    result.skipped_manual += 1
                    continue
                activity_ids = match_activities(document.document_id, rules)
                if activity_ids:
                    document.activity_ids = activity_ids
                    document.association_source = AssociationSource.RULE
                    result.associated += 1
                else:
                    result.unmatched += 1
            session.commit()
        return result
            

    def apply_coverage(self, document_id: str, coverage: Mapping[int, str | None]) -> None:
        """Write the per-instance compliance outcome (block B7).

        Sets each instance's
        :attr:`~pii_detection.registry.types.PIIInstance.processing_activity_id` to
        the activity that justifies it, or ``None`` when the instance is orphan.
        The mapping is keyed by instance id; ids not belonging to the document are
        ignored.

        :param document_id: identifier of the document whose instances to update.
        :param coverage: instance id → justifying activity id (or ``None`` if orphan).
        """
        with Session(self.engine) as session:
            instances = session.exec(
                select(PIIInstance).where(PIIInstance.document_id == document_id)
            ).all()
            for instance in instances:
                if instance.id is not None and instance.id in coverage:
                    instance.processing_activity_id = coverage[instance.id]
            session.commit()

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
