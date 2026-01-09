# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Mock imports for dependencies that might not be available on ReadTheDocs
# We need BOTH sys.modules mocks AND autodoc_mock_imports for complete coverage

import types
from unittest.mock import MagicMock

class MockModule(MagicMock):
    """Mock module that acts as both a module and a class."""
    
    # Make it usable as a base class
    @classmethod
    def __subclasshook__(cls, C):
        return True
    
    def __init__(self, name='', *args, **kwargs):
        super().__init__()
        self.__name__ = name
        self.__path__ = []
        self.__file__ = f'<mock {name}>'
        self.__all__ = []
    
    def __call__(self, *args, **kwargs):
        return MockModule()
    
    def __getitem__(self, key):
        return MockModule()
    
    def __setitem__(self, key, value):
        pass

# List of modules to mock
mock_modules = [
    # PyTorch (all submodules needed)
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.nn.init', 
    'torch.nn.utils', 'torch.optim', 'torch.cuda', 'torch.autograd',
    'torch.fft', 'torch.linalg', 'torch.distributions',
    # JAX
    'jax', 'jax.numpy', 'jax.random', 'jax.lax', 'jax.nn', 
    'jax.config', 'jax.experimental', 'jaxlib',
    # Numba
    'numba',
    # Other ML libraries
    'optax', 'cupy', 'sklearn', 'numpyro', 'optuna',
]

# Pre-install mocks in sys.modules so they're available during imports
for mod_name in mock_modules:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MockModule(mod_name)

# Also set autodoc_mock_imports for Sphinx autodoc
autodoc_mock_imports = mock_modules

# -- Project information -----------------------------------------------------

project = 'HPFRACC'
copyright = '2025, Davian R. Chin'
author = 'Davian R. Chin'

# The full version, including alpha/beta/rc tags
release = '3.1.0'
version = '3.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'logo_only': False,
    'display_version': True,
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# -- Options for autodoc ----------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = 'description'

# Don't show type hints in the signature
autodoc_typehints_format = 'short'

# Include both class docstring and __init__ docstring
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# -- Options for intersphinx -------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for Napoleon ---------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# -- Options for MathJax ----------------------------------------------------

# MathJax configuration
mathjax_config = {
    'TeX': {
        'equationNumbers': {'autoNumber': 'AMS'},
        'extensions': ['AMSmath.js', 'AMSsymbols.js'],
    }
}

# -- Options for autosummary -------------------------------------------------

# Generate stub files for autosummary
autosummary_generate = True

# -- Options for MyST-Parser -------------------------------------------------

# MyST-Parser extensions
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_image',
    'html_admonition',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# -- Options for PDF output -------------------------------------------------

# PDF options
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'figure_align': 'htbp',
}

# -- Options for EPUB output -------------------------------------------------

# EPUB options
epub_show_urls = 'footnote'

# -- Extension configuration -------------------------------------------------

# Add any extension configuration here
