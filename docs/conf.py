# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
print(sys.path)


# -- Project information -----------------------------------------------------

project = 'METcalcpy'
copyright = '2022, NCAR'
author = 'UCAR/NCAR, NOAA, CSU/CIRA, and CU/CIRES'
author_list = 'Win-Gildenmeister, M., T. Burek, H. Fisher, C. Kalb, D. Adriaansen, D. Fillmore, and T. Jensen'
version = 'develop'
verinfo = version
release = f'{version}'
release_year = '2022'
release_date = f'{release_year}-09-14'
copyright = f'{release_year}, {author}'

# if set, adds "Last updated on " followed by
# the date in the specified format
html_last_updated_fmt = '%c'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'sphinx_gallery.gen_gallery']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'utils/README_util.rst','auto_examples']

# Suppress certain warning messages
suppress_warnings = ['ref.citation']

# -- Sphinx control -----------------------------------------------------------
#sphinx_gallery_conf = {
#      'examples_dirs': [os.path.join('..', 'examples')],
#      'gallery_dirs': ['examples']
#}
    

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes", ]
html_js_files = ['pop_ver.js']
html_css_files = ['theme_override.css']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = os.path.join('_static','met_calcpy_logo_2019_09.png')

# Control html_sidebars
# Include global TOC instead of local TOC by default
#html_sidebars = { '**': ['globaltoc.html','relations.html','sourcelink.html','searchbox.html']}

# -- Intersphinx control -----------------------------------------------------

numfig = True

numfig_format = {
        'figure': 'Figure %s',
        }
    
# -- Export variables --------------------------------------------------------

rst_epilog = """                                                                                                                                    
.. |copyright|    replace:: {copyrightstr}                                                                                                          
.. |author_list|  replace:: {author_liststr}                                                                                                        
.. |release_date| replace:: {release_datestr}                                                                                                       
.. |release_year| replace:: {release_yearstr}                                                                                                       
""".format(copyrightstr    = copyright,
           author_liststr  = author_list,
           release_datestr = release_date,
           release_yearstr = release_year)

