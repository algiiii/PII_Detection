"""Storage access for the ROPA — block B1, persistence.

:class:`ROPARepository` is the single way in and out of the database. Because the
domain classes of :mod:`pii_detection.ropa.types` are themselves SQLModel tables
(Active Record), there is no row/domain translation here: the repository only
owns the engine and the two whole-register operations the rest of the system
needs — the ingestion pipeline writes with :meth:`ROPARepository.save`, the
review app reads with :meth:`ROPARepository.load`.

The database URL is read from the ``ROPA_DB_URL`` environment variable so the
same image runs locally and inside a container by changing only the config.
"""

from __future__ import annotations

import os

from sqlmodel import Session, SQLModel, create_engine, select

from pii_detection.detection.config import (
    default_config_dir,
    load_category_catalog,
)
from pii_detection.ropa.types import (
    DeclaredCategory,
    DeclaredMacroCategory,
    MappingState,
    ProcessingActivity,
)


class ROPARepository:
    """Read and write the whole register against a configured database.

    :ivar engine: SQLAlchemy engine bound to the resolved database URL.
    :ivar catalog: the canonical ``pii_type`` catalog, used to reject unknown
        types on a category update.
    """

    def __init__(self, url: str | None = None) -> None:
        """Open the database, creating the schema if it does not exist yet.

        :param url: SQLAlchemy database URL; when ``None`` it is read from the
            ``ROPA_DB_URL`` environment variable, defaulting to the canonical local
            store ``sqlite:///data/ropa.db`` (the ``data/`` directory, shared with
            the container via the repo bind-mount).
        """
        url = url or os.environ.get("ROPA_DB_URL", "sqlite:///data/ropa.db")
        self.engine = create_engine(url)
        # Create only the ROPA tables: the SQLModel metadata is shared across the
        # whole process, so an unfiltered create_all() would also materialize the
        # detected-PII registry tables (registry_*) in this ROPA database. The two
        # registers live in separate databases and share no physical foreign key.
        metadata = SQLModel.metadata
        metadata.create_all(
            self.engine,
            tables=[
                metadata.tables[name]
                for name in ("processing_activity", "macro_category", "declared_category")
            ],
        )
        self.catalog = load_category_catalog(default_config_dir() / "categories.yaml")

    def save(self, to_be_saved: list[ProcessingActivity]) -> None:
        """Persist processing activities with their macro categories and categories.

        The nested macro categories and declared categories are written by the
        cascade on the relationships (``all, delete-orphan``); only the top-level
        activities need to be added.

        :param to_be_saved: the activities to store, each with its full subtree.
        """
        with Session(self.engine) as s:
            s.add_all(to_be_saved)
            s.commit()

    def load(self) -> list[ProcessingActivity]:
        """Read the whole register back as domain objects.

        The macro categories and their declared categories come eagerly loaded
        (``lazy="selectin"`` on the relationships), so the returned tree is safe
        to navigate after the session is closed.

        :returns: every stored activity, each with its full subtree.
        """
        with Session(self.engine, expire_on_commit=False) as s:
            return list(s.exec(select(ProcessingActivity)).all())
        
    def clear(self) -> None:
        """Delete every processing activity and its subtree from the register.

        Wipes the register in place, leaving an empty but initialized database.
        Each activity is deleted through the session so the relationship cascade
        (``all, delete-orphan``) also removes its macro categories and declared
        categories. Destructive: used by a ``--replace`` re-ingestion, it drops
        any edit made through the review app.
        """
        with Session(self.engine) as s:
            for activity in s.exec(select(ProcessingActivity)).all():
                s.delete(activity)
            s.commit()

    def get(self, activity_id: str) -> ProcessingActivity | None:
        """Load a single activity by id, with its full subtree.

        :param activity_id: identifier of the activity to fetch.
        :returns: the activity with its macro categories and declared
            categories eagerly loaded, or ``None`` if the id is unknown.
        """
        with Session(self.engine, expire_on_commit=False) as s:
            return s.get(ProcessingActivity, activity_id)

    def delete_activity(self, activity_id: str) -> None:
        """Delete a single processing activity and its subtree (block B1).

        Removes one activity through the session so the relationship cascade
        (``all, delete-orphan``) also drops its macro categories and declared
        categories; a ``clear()`` in the small, per-activity form. Deleting an id
        that does not exist is a no-op.

        :param activity_id: identifier of the activity to delete.
        """
        with Session(self.engine) as s:
            activity = s.get(ProcessingActivity, activity_id)
            if activity is not None:
                s.delete(activity)
                s.commit()

    def catalog_ids(self) -> list[str]:
        """List the declared ``pii_type`` ids, in catalog order.

        :returns: the catalog ids, for the review checkboxes.
        """
        return [category.id for category in self.catalog]

    def update_category(
        self, category_id: int, pii_types: list[str], mapping_state: MappingState
    ) -> str:
        """Set a declared category's ``pii_types`` and mapping state.

        :param category_id: id of the declared category to update.
        :param pii_types: the catalog ids to associate; each must be declared in
            the catalog.
        :param mapping_state: the new mapping state.
        :returns: the id of the parent activity, for the caller's redirect.
        :raises KeyError: if no category (or parent) has that id.
        :raises ValueError: if any ``pii_type`` is not in the catalog.
        """
        unknown = [t for t in pii_types if t not in self.catalog]
        if unknown:
            raise ValueError(f"unknown pii_types: {unknown}")
        with Session(self.engine) as s:
            category = s.get(DeclaredCategory, category_id)
            if category is None:
                raise KeyError(category_id)
            macro = s.get(DeclaredMacroCategory, category.macro_category_id)
            if macro is None or macro.activity_id is None:
                raise KeyError(category_id)
            activity_id = macro.activity_id
            category.pii_types = pii_types
            category.mapping_state = mapping_state
            s.add(category)
            s.commit()
            return activity_id

    def split_category(self, category_id: int, parts: list[tuple[str, list[str]]]) -> str:
        """Replace a declared category with the sub-categories a mapper produced.

        Used by the post-ingestion mapping pass: the raw declared category is
        removed and one new :class:`~pii_detection.ropa.types.DeclaredCategory`
        (``PROPOSED``) is added per part, under the same macro category. Takes
        plain ``(raw_text, pii_types)`` tuples so the persistence layer stays
        independent of the mapping layer.

        :param category_id: id of the declared category to replace.
        :param parts: the sub-categories to insert, each a ``(raw_text,
            pii_types)`` pair; ``pii_types`` may be empty (not resolved).
        :returns: the id of the parent activity, for the caller's redirect.
        :raises KeyError: if no category (or parent) has that id.
        :raises ValueError: if any ``pii_type`` is not in the catalog.
        """
        unknown = [t for _, pii_types in parts for t in pii_types if t not in self.catalog]
        if unknown:
            raise ValueError(f"unknown pii_types: {unknown}")
        with Session(self.engine) as s:
            category = s.get(DeclaredCategory, category_id)
            if category is None:
                raise KeyError(category_id)
            macro = s.get(DeclaredMacroCategory, category.macro_category_id)
            if macro is None or macro.activity_id is None:
                raise KeyError(category_id)
            activity_id = macro.activity_id
            s.delete(category)
            for raw_text, pii_types in parts:
                s.add(
                    DeclaredCategory(
                        macro_category_id=macro.id,
                        raw_text=raw_text,
                        pii_types=pii_types,
                        mapping_state=MappingState.PROPOSED,
                    )
                )
            s.commit()
            return activity_id

    def confirm_macro(self, macro_id: int) -> str:
        """Confirm every declared category under a macro category.

        :param macro_id: id of the macro category whose children to confirm.
        :returns: the id of the parent activity, for the caller's redirect.
        :raises KeyError: if no macro category has that id.
        """
        with Session(self.engine) as s:
            macro = s.get(DeclaredMacroCategory, macro_id)
            if macro is None or macro.activity_id is None:
                raise KeyError(macro_id)
            activity_id = macro.activity_id
            for category in macro.categories:
                category.mapping_state = MappingState.CONFIRMED
            s.commit()
            return activity_id

    #: Wording of the macro category that gathers detection-driven proposals under an
    #: activity, so a proposed type never invents a retention it does not have.
    _PROPOSAL_MACRO_TEXT = "Categorie proposte dal rilevamento"

    def propose_category(self, activity_id: str, pii_type: str) -> None:
        """Propose an orphan ``pii_type`` as a declared category of an activity (B7→B1).

        When the AI (or any detector) finds a ``pii_type`` that no activity declares,
        the DPO can turn that orphan into a **proposal**: a
        :class:`~pii_detection.ropa.types.DeclaredCategory` in state
        :attr:`~pii_detection.ropa.types.MappingState.PROPOSED` (never ``CONFIRMED``),
        added under a dedicated proposals macro category with **no retention** — so the
        proposal carries *what* was found and *where*, never a retention nor a legal
        basis the system does not know. The category wording is the catalog label of the
        type (deterministic; the catalog already names it, so no LLM is asked to invent a
        name). Writing to the register stays an explicit DPO action; nothing here is
        automatic. Proposing a type already proposed on the activity is a no-op
        (idempotent).

        :param activity_id: the activity to attach the proposal to.
        :param pii_type: the orphan catalog id to propose.
        :raises KeyError: if the activity does not exist.
        :raises ValueError: if ``pii_type`` is not in the catalog.
        """
        category_model = self.catalog.get(pii_type)
        if category_model is None:
            raise ValueError(f"unknown pii_type: {pii_type!r}")
        with Session(self.engine) as s:
            activity = s.get(ProcessingActivity, activity_id)
            if activity is None:
                raise KeyError(activity_id)
            macro = s.exec(
                select(DeclaredMacroCategory).where(
                    DeclaredMacroCategory.activity_id == activity_id,
                    DeclaredMacroCategory.raw_text == self._PROPOSAL_MACRO_TEXT,
                )
            ).first()
            if macro is None:
                macro = DeclaredMacroCategory(
                    activity_id=activity_id,
                    raw_text=self._PROPOSAL_MACRO_TEXT,
                    retention_text="",
                    retention_months=None,
                )
                s.add(macro)
                s.flush()  # assign macro.id
            already = any(pii_type in category.pii_types for category in macro.categories)
            if not already:
                s.add(
                    DeclaredCategory(
                        macro_category_id=macro.id,
                        raw_text=category_model.label,
                        pii_types=[pii_type],
                        mapping_state=MappingState.PROPOSED,
                    )
                )
                s.commit()
