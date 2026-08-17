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

from sqlmodel import Session, SQLModel, col, create_engine, select

from pii_detection.detection.types import PIIMatch
from pii_detection.extraction.dates import ReferenceDate
from pii_detection.registry.diff import diff_scan
from pii_detection.registry.folder_rules import ApplyRulesResult, match_activities
from pii_detection.registry.freshness import FileStamp
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
        reference_date: ReferenceDate | None = None,
        stamp: FileStamp | None = None,
        detector_signature: str | None = None,
        replace: bool = False,
    ) -> Scan:
        """Persist one scan of a document, computing the delta against its state.

        Creates the :class:`~pii_detection.registry.types.Document` if new and a
        new :class:`~pii_detection.registry.types.Scan`, then compares the matches
        with the document's current instances (:func:`diff_scan`) and applies the
        outcome: ``CONFIRMED`` (same position; refresh confidence/level/sources),
        ``MOVED`` (update position too), ``NEW`` (create instance) and ``REMOVED``
        (mark the instance gone). Every
        change links to the previous scan of the same document (``None`` on the
        first scan). Instances never store the PII value (minimization).

        :param document_id: identifier of the scanned document (its file stem).
        :param matches: the unified PII of this scan; their ``text`` is never stored.
        :param path: original document path, kept for reference.
        :param reference_date: the date the document is assumed to date from, with
            its provenance, for the retention check (B7); refreshed on every scan
            that provides one.
        :param stamp: the file's observed state (modification time and size), kept
            so a later scan can tell whether the file changed; refreshed likewise.
        :param detector_signature: fingerprint of the engine that produced these
            matches, so a later scan can tell the engine itself changed.
        :param replace: if ``True``, wipe the document's history first and record
            this scan as a fresh bootstrap (all ``NEW``).
        :returns: the created scan.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            document = session.get(Document, document_id)
            if document is None:
                document = Document(document_id=document_id, path=path)
                session.add(document)
            if reference_date is not None:
                document.reference_date = reference_date.value
                document.reference_date_source = reference_date.source.value
            if stamp is not None:
                document.source_mtime = stamp.modified_at
                document.source_size = stamp.size
            if detector_signature is not None:
                document.detector_signature = detector_signature

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
            document.last_scanned_at = scan.created_at

            current = [instance for instance in all_instances if not instance.removed]
            delta = diff_scan(current, matches)

            for instance, match in delta.confirmed:
                # Same identity (pii_type + position), but the certainty may have
                # changed: a re-scan where the AI now confirms an existing instance
                # keeps the span (hence CONFIRMED, not MOVED) yet must refresh
                # confidence/level/sources, or the new agreement stays invisible.
                instance.confidence = match.confidence
                instance.confirmation_level = match.confirmation_level.value
                instance.sources = [p.detector_id for p in match.sources]
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

    def delete_rule(self, prefix: str) -> ApplyRulesResult:
        """Delete a folder rule and reconcile the associations it had derived (B6).

        Removing the rule row is not enough: documents it had associated still
        carry its ``RULE``-sourced :attr:`~pii_detection.registry.types.Document.activity_ids`.
        After deleting, the rule-derived associations are therefore recomputed
        against the *remaining* rules (:meth:`apply_folder_rules`), so a document
        no longer covered by any rule is cleared instead of keeping a dangling
        association. Manual associations are preserved.

        :param prefix: the prefix of the rule to delete (normalized like
            :meth:`save_rule`); a no-op if no such rule exists.
        :returns: the reconciliation summary
            (:class:`~pii_detection.registry.folder_rules.ApplyRulesResult`).
        """
        normalized = prefix.strip().strip("/")
        with Session(self.engine) as session:
            rule = session.get(FolderRule, normalized)
            if rule is not None:
                session.delete(rule)
                session.commit()
        return self.apply_folder_rules()

    def apply_folder_rules(self) -> ApplyRulesResult:
        """Reconcile every document's rule-derived association with the rules (B6).

        For each document, except those associated **by hand**
        (:attr:`~pii_detection.registry.types.AssociationSource.MANUAL`, manual
        wins): the activities it inherits are recomputed from the current rules
        (:func:`~pii_detection.registry.folder_rules.match_activities`). A document
        that matches at least one rule is (re)associated with source ``RULE``; a
        document that matches none but still carries a stale ``RULE`` association
        is **cleared** (empty ids, source back to ``None``), so deleting or
        shrinking a rule removes the associations it had produced. Documents that
        match nothing and had no rule association are left untouched.

        Run as a separate step after a scan (keeping the scan pure) and on demand
        from the web UI; being a full reconciliation makes it idempotent.

        :returns: the counts of the reconciliation
            (:class:`~pii_detection.registry.folder_rules.ApplyRulesResult`).
        """
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
                elif document.association_source == AssociationSource.RULE:
                    document.activity_ids = []
                    document.association_source = None
                    result.cleared += 1
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
