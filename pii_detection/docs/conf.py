"""Configurazione Sphinx per la reference API del motore di detection (B4).

Genera un sito HTML navigabile a partire dai docstring reST del package
``pii_detection``. Build (dalla radice del repo):
``sphinx-build -b html pii_detection/docs pii_detection/docs/_build/html``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Risale a pii_detection/docs -> pii_detection -> radice del repo, così il
# package è importabile anche fuori dal venv editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "PII Detection (blocco B4)"
author = "Gabriele Algisi"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",  # estrae i docstring dal codice
    "sphinx.ext.viewcode",  # link "source" a fianco di ogni oggetto
    "sphinx.ext.intersphinx",  # link alla stdlib Python
    "sphinx_autodoc_typehints",  # type hint resi nella descrizione
]

language = "it"

# --- autodoc -------------------------------------------------------------
autodoc_member_order = "bysource"  # stesso ordine del codice, non alfabetico
autoclass_content = "both"  # unisce docstring di classe e __init__
autodoc_typehints = "description"  # i tipi finiscono nel corpo, non nella firma
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# --- HTML ----------------------------------------------------------------
html_theme = "furo"  # tema con sidebar navigabile e ricerca client-side
html_title = "PII Detection — reference API"
