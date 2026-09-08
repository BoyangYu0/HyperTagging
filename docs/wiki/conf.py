"""Offline Sphinx configuration; no hypertagging imports or remote inventories."""
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "docs" / "_ext"))

project = "HyperTagging"
author = "HyperTagging contributors"
copyright = "HyperTagging contributors"
version = "0.0.0"
release = version
extensions = [
    "wiki", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode", "sphinx.ext.mathjax", "sphinx.ext.autosectionlabel",
]
source_suffix = ".rst"
master_doc = "index"
root_doc = master_doc
exclude_patterns = ["_build/**", "**/__pycache__/**"]
language = "en"
nitpicky = False  # External annotation names are source text, not installed APIs.
autodoc_mock_imports = [
    "basf2", "ROOT", "modularAnalysis", "variables", "stdCharged", "stdPhotons",
    "torch", "numpy", "pandas", "awkward", "pyarrow", "uproot", "pdg",
    "onnx", "onnxruntime", "scipy", "matplotlib", "networkx",
]
autodoc_typehints = "none"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosectionlabel_prefix_document = True
default_role = "any"
numfig = True
python_use_unqualified_type_names = True
viewcode_follow_imported_members = False
viewcode_enable_epub = False
# Keep the basf2 extension contract, using a text renderer for offline Pages.
# No MathJax distribution, CDN, browser fetch or runtime JavaScript is needed.
html_math_renderer = "offline-text"
mathjax_path = ""
intersphinx_mapping = {}  # No downloads; see maintaining.rst for local inventories.
html_theme = "classic"
html_static_path = ["_static"]
html_css_files = ["wiki.css"]
html_theme_options = {"body_max_width": "1180px", "sidebarwidth": "270px"}
html_sidebars = {"**": ["globaltoc.html", "relations.html", "searchbox.html"]}
html_show_sourcelink = False
html_last_updated_fmt = None
html_copy_source = False
html_use_index = True
html_domain_indices = True
html_title = "HyperTagging documentation"
html_short_title = "HyperTagging"
html_extra_path = [".nojekyll"]
smartquotes = False
epub_show_urls = "footnote"


def setup(app):
    """Retain math markup and equation numbers with an offline text fallback."""
    from sphinx.ext.mathjax import html_visit_displaymath, html_visit_math
    app.add_html_math_renderer(
        "offline-text",
        inline_renderers=(html_visit_math, None),
        block_renderers=(html_visit_displaymath, None),
    )
