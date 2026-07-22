from __future__ import annotations

import os
from pii_detection.ropa.types import ProcessingActivity
from sqlmodel import SQLModel, Session, create_engine, select

class ROPARepository:
    def __init__(self, url: str | None = None) -> None:
        url = url or os.environ.get("ROPA_DB_URL", "sqlite:///ropa.db")
        self.engine = create_engine(url)
        SQLModel.metadata.create_all(self.engine)

    def save(self, to_be_saved: list[ProcessingActivity]) -> None:
        with Session(self.engine) as s:
            s.add_all(to_be_saved)
            s.commit()

    def load(self) -> list[ProcessingActivity]:
        with Session(self.engine, expire_on_commit=False) as s:
            return list(s.exec(select(ProcessingActivity)).all())
