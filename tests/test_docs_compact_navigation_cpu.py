"""Compact chrome keeps accessible routes without changing document content."""
from pathlib import Path
from types import SimpleNamespace

from jinja2 import ChoiceLoader, DictLoader, Environment, FileSystemLoader
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("prefix", ["wiki", "docs/wiki", ""])
def test_compact_navigation_preserves_layout_relative_routes(prefix):
    environment = Environment(loader=FileSystemLoader(ROOT / "docs/_templates"))
    rendered = environment.get_template("compact_navigation.html").render(
        wiki_content_path=prefix, pathto=lambda value: value + ".html")
    base = prefix + "/" if prefix else ""
    for target in ("index", "studies", "_generated/status/index", "_generated/status/downloads",
                   "_generated/api/index", "_generated/catalog/index"):
        assert f'href="{base}{target}.html"' in rendered
    assert "All metric downloads" in rendered


def test_layout_retains_top_breadcrumbs_previous_next_and_body_once():
    environment = Environment(loader=ChoiceLoader([
        FileSystemLoader(ROOT / "docs/_templates"),
        DictLoader({"!layout.html": "{% block relbar1 %}{% endblock %}<main>{{ body }}</main>{% block relbar2 %}DUPLICATE BAR{% endblock %}"}),
    ]))
    rendered = environment.get_template("layout.html").render(
        master_doc="index", pathto=lambda value: value + ".html", body="FULL DOCUMENT BODY",
        parents=[{"link": "parent.html", "title": "Parent"}],
        prev={"link": "previous.html"}, next={"link": "next.html"})
    assert rendered.count("FULL DOCUMENT BODY") == 1
    assert "DUPLICATE BAR" not in rendered
    for target in ("index.html", "parent.html", "previous.html", "next.html"):
        assert f'href="{target}"' in rendered
    assert 'aria-label="Page navigation"' in rendered
    # Classic styles target div.related; semantic nav loses link contrast.
    assert '<div class="related" role="navigation"' in rendered
    assert '<nav class="related"' not in rendered
    # Float controls precede inline breadcrumbs so narrow wrapping increases
    # the colored bar height instead of leaving white links below its background.
    assert rendered.index('rel="next"') < rendered.index('rel="prev"') < rendered.index('href="index.html"')
