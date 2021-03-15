# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'PyCalib'
copyright = '2021, Miquel Perello-Nieto'
author = 'Miquel Perello-Nieto'

# The full version, including alpha/beta/rc tags
release = '0.0.4.dev0'

github_org = 'perellonieto'
github_repo = 'pycalib'
github_docs_repo = 'pycalib'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx_gallery.gen_gallery"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Generate examples with figures
sphinx_gallery_conf = {
    'filename_pattern': '/xmpl_',
    'examples_dirs': os.path.join('..', '..', 'examples'),
    'gallery_dirs': 'examples',
    'backreferences_dir': 'generated',  # `doc_module`
    'doc_module': 'pycalib',  # Generate mini galleries for the API documentation.
    'reference_url': {'pycalib': None},  # Put links to docs in the examples code.
    'binder': {
        'org': github_org,
        'repo': github_docs_repo,
        'branch': 'gh-pages',
        'binderhub_url': 'https://mybinder.org',
        'dependencies': [os.path.join('..', '..', 'requirements.txt'),
                         os.path.join('..', '..', 'requirements-dev.txt')]
    }
}
