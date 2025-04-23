# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

#Sphinx can find the Python module:
import os
import sys

# Adjust the path so it points to the folder containing 'ringdetection.py'
# If 'docs' and 'ring_detection_dummy' are siblings, you go one level up:
sys.path.insert(0, os.path.abspath('../RingDetectionToolKit'))
#sys.path.insert(0, os.path.abspath('RELATIVE_PATH_TO_CODE'))


project = 'RingDetection'
copyright = '2025, Alessandro Fiorentino'
author = 'Alessandro Fiorentino'
release = '1.0.a0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Auto-generate docs from docstrings
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx.ext.napoleon',      # Support for Google/Numpy docstrings
    'sphinx.ext.mathjax',       # Render math
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'private-members': True
}

autodoc_mock_imports = ["tensorflow"]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme options
html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
