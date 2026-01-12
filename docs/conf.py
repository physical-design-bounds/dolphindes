"""Sphinx configuration for the Dolphindes documentation."""

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Dolphindes"
copyright = "2025, DolphinDes contributors"
author = "DolphinDes contributors"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.mathjax",  # Render math via MathJax
    "myst_parser",  # Parse Markdown files
    "nbsphinx",  # Parse Jupyter Notebooks
    "sphinx_automodapi.automodapi",  # Automated API documentation
    "sphinx.ext.graphviz",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"

# Optional: Add your github repo link to the header like NumPy does
html_theme_options = {
    "github_url": "https://github.com/physical-design-bounds/dolphindes",
    "show_nav_level": 2,
    "navigation_depth": 4,
}

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

project = "dolphindes"

# Keep both Google and NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Prefer docstring types over annotations for rendering (optional)
autodoc_typehints = "description"

# # Disambiguate short type names used in docstrings
# napoleon_type_aliases = {
#     # pick canonical, re-exported targets to avoid duplicates
#     "TM_FDFD": "dolphindes.maxwell.TM_FDFD",
#     "Projectors": "dolphindes.util.Projectors",
#     "SparseSharedProjQCQP": "dolphindes.cvxopt.SparseSharedProjQCQP",
#     "DenseSharedProjQCQP": "dolphindes.cvxopt.DenseSharedProjQCQP",
#     # add other common short names if needed
#     # "Photonics_TM_FDFD": "dolphindes.photonics.Photonics_TM_FDFD",
#     # "CartesianFDFDGeometry": "dolphindes.photonics.CartesianFDFDGeometry",
# }

add_module_names = False
nbsphinx_execute = "never"
