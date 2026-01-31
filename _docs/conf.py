import sys
from pathlib import Path

#####ANCHOR Make root package importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "API Documentation"
extensions = [
    "myst_parser",
    "autodoc2",
    # "sphinx.ext.autodoc",
    "sphinx_ext_mystmd",
]
numfig = True

#####ANCHOR  autodoc2 config
autodoc2_packages = [
    {
        "path": "../asext",
        # "module": "asext",
    }
]

autodoc2_render_plugin = "myst"
# autodoc2_hidden_objects = ["private"]

#####ANCHOR  MyST config
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
