# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


# Project source path
sys.path.insert(0, os.path.abspath(".."))  # docs/ is sibling to sardalign/


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sardalign"
copyright = "2025, Anil Keshwani"
author = "Anil Keshwani"
release = "v0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # For auto-extracting docstrings
    "sphinx.ext.napoleon",  # Supports Google/NumPy docstring styles
    "sphinx_autodoc_typehints",  # For type hints
    "myst_parser",  # Enable Markdown support
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",  # Markdown support via MyST
}


# -- MyST configuration ---------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # Support ::: for code blocks
    "deflist",  # Definition lists
    "substitution",  # Variable substitutions
    "linkify",  # Auto-convert URLs to links
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # out-of-the-box: "alabaster"
html_static_path = ["_static"]
