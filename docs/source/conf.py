# -*- coding: utf-8 -*-
"""
Sphinx configuration for the flux-data-footprint documentation.
"""

import re
from pathlib import Path

# read the version from the package without importing it
init_file = Path(__file__).resolve().parents[2] / 'fluxfootprint' / '__init__.py'
release = re.search(r"__version__ = '(.+)'", init_file.read_text()).group(1)
version = release

project = 'flux-data-footprint'
author = 'John Volk'
copyright = '2026, John Volk'

# autodoc and napoleon (Google style docstrings) for API pages later
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
