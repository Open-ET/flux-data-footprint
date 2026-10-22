# -*- coding: utf-8 -*-
"""
Sphinx configuration for the flux-data-footprint documentation.
"""

import re
import sys
from pathlib import Path

# repository folder, so autodoc can import fluxfootprint without installing it
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

# read the version from the package without importing it
init_file = repo / 'fluxfootprint' / '__init__.py'
release = re.search(r"__version__ = '(.+)'", init_file.read_text()).group(1)
version = release

project = 'flux-data-footprint'
author = 'John Volk'
copyright = '2026, John Volk'

# autodoc and napoleon (Google style docstrings) for the API page
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

# packages fluxfootprint imports when loaded, mocked so Read the Docs only
# needs sphinx and the theme installed
autodoc_mock_imports = ['numpy', 'pandas', 'affine', 'requests']
autodoc_member_order = 'bysource'

# show function names without the module path, they are imported from
# the package itself
add_module_names = False

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
