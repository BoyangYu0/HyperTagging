"""Standalone documentation root sharing the offline wiki configuration."""
from pathlib import Path
import runpy

_shared = runpy.run_path(str(Path(__file__).resolve().parent / "wiki" / "conf.py"))
globals().update({name: value for name, value in _shared.items() if not name.startswith("_")})
wiki_content_path = "wiki"
html_static_path = ["wiki/_static"]
html_extra_path = ["wiki/.nojekyll"]
