"""Sphinx configuration for the API reference of the detection engine (B4).

Generates a navigable HTML site from the reST docstrings of the
``pii_detection`` package. Build (from the repo root):
``sphinx-build -b html pii_detection/docs pii_detection/docs/_build/html``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Walk up pii_detection/docs -> pii_detection -> repo root, so the package is
# importable even outside the editable venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "PII Detection (blocco B4)"
author = "Gabriele Algisi"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",  # extract docstrings from the code
    "sphinx.ext.viewcode",  # "source" link next to every object
    "sphinx.ext.intersphinx",  # link to the Python stdlib
    "sphinx_autodoc_typehints",  # type hints rendered in the description
]

language = "en"

# --- autodoc -------------------------------------------------------------
autodoc_member_order = "bysource"  # same order as the code, not alphabetical
autoclass_content = "both"  # merge class and __init__ docstrings
autodoc_typehints = "description"  # types go in the body, not the signature
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# --- HTML ----------------------------------------------------------------
html_theme = "furo"  # theme with navigable sidebar and client-side search
html_title = "PII Detection — API reference"
