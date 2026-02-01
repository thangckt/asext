import sys
from pathlib import Path

#####ANCHOR Make root package importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "API Documentation"
extensions = [
    "myst_parser",
    "sphinx_ext_mystmd",
    "autodoc2",
    # "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google style docstrings
]
numfig = True

#####ANCHOR  autodoc2 config
autodoc2_packages = [
    {
        "path": "../../asext",
        "module": "asext",
        "exclude_files": ["_version.py"],
    }
]

autodoc2_render_plugin = "myst"
autodoc2_hidden_objects = [
    "private",
    "dunder",
    "inherited",
]
autodoc2_skip_module_regexes = [
    "_version",
]


#####ANCHOR  MyST config
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]


#####ANCHOR napoleon for Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Quality-of-life (recommended)
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = False
# napoleon_use_param = True
# napoleon_use_rtype = True
