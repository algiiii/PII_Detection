# file/classe responsabile per mappare AI -> struttura fissa

from __future__ import annotations # per X | None
import json # parse della risposta del modello
import os # per leggere ROPA_LLM_MODEL dal .env
from typing import Any, Protocol, runtime_checkable

from ollama import Client # Client verso server di Ollama
from pii_detection.detection.config import PIICategoryCatalog, default_config_dir, load_category_catalog

DEFAULT_MODEL = "phi4-mini" # Costante di modulo (impostabile volendo come parametro)

@runtime_checkable
class CategoryMapper:
    def map(self, raw_text: str) -> tuple[str, ...]: ...