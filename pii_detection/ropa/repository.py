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

from pii_detection.ropa.types import ProcessingActivity


class ROPARepository:
    """Read and write the whole register against a configured database.

    :ivar engine: SQLAlchemy engine bound to the resolved database URL.
    """

    def __init__(self, url: str | None = None) -> None:
        """Open the database, creating the schema if it does not exist yet.

        :param url: SQLAlchemy database URL; when ``None`` it is read from the
            ``ROPA_DB_URL`` environment variable, defaulting to the local file
            ``sqlite:///ropa.db``.
        """
        url = url or os.environ.get("ROPA_DB_URL", "sqlite:///ropa.db")
        self.engine = create_engine(url)
        SQLModel.metadata.create_all(self.engine)

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
