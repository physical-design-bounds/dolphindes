# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Dolphindes"
copyright = "2025, Dolphindes Contributors"
author = "Dolphindes Contributors"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

import os
import sys

# Ensure the package is importable
sys.path.insert(0, os.path.abspath(".."))

project = "dolphindes"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Keep both Google and NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Prefer docstring types over annotations for rendering (optional)
autodoc_typehints = "description"

# Disambiguate short type names used in docstrings
napoleon_type_aliases = {
    # pick canonical, re-exported targets to avoid duplicates
    "TM_FDFD": "dolphindes.maxwell.TM_FDFD",
    "Projectors": "dolphindes.util.Projectors",
    "SparseSharedProjQCQP": "dolphindes.cvxopt.SparseSharedProjQCQP",
    "DenseSharedProjQCQP": "dolphindes.cvxopt.DenseSharedProjQCQP",
    # add other common short names if needed
    # "Photonics_TM_FDFD": "dolphindes.photonics.Photonics_TM_FDFD",
    # "CartesianFDFDGeometry": "dolphindes.photonics.CartesianFDFDGeometry",
}

html_theme = "sphinx_rtd_theme"

add_module_names = False
