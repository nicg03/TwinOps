# Configuration file for the Sphinx documentation builder.
#
# TwinOps documentation

import os
import sys

# Allow Sphinx to import the twinops package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "TwinOps"
copyright = "2025, TwinOps"
author = "TwinOps"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# Use sphinx_rtd_theme if installed (pip install sphinx-rtd-theme), else default
try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"
html_static_path = ["_static"]
html_title = "TwinOps"

# -- Extension configuration -------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = ["torch", "gplearn"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
