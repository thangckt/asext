import sys
from pathlib import Path

#####ANCHOR Make root package importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "API Documentation"
extensions = [
    "myst_parser",
    "sphinx_ext_mystmd",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google style docstrings
    "sphinx.ext.autosummary",
]
numfig = True

#####ANCHOR Sphinx extensions
### autodoc for API documentation
autodoc_default_options = {
    "member-order": "bysource",
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
}

autodoc_typehints = "both"
# autosummary_generate = True

### napoleon for Google style docstrings
napoleon_google_docstring = True
# napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = False


#####ANCHOR  MyST config
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
