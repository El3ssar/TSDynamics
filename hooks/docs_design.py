"""Build-time presentation helpers for documentation templates.

Markdown fences are highlighted by SuperFences. Template-only examples use the
same Pygments token vocabulary so they remain highlighted without JavaScript or
a network request, including when navigating with Material's instant loading.
"""

import sys
from pathlib import Path
from runpy import run_path
from textwrap import dedent

from markupsafe import Markup
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name


def on_pre_build(config):
    """Keep illustrated recipes synchronized with the code readers can copy.

    These guards inspect checked-in source hashes and assets only. A normal
    documentation build never needs to encode the showcase movies again.
    """
    from mkdocs.exceptions import PluginError

    tooling = Path(__file__).resolve().parents[1] / "docs" / "_tooling"
    failures = []
    previous_path = sys.path.copy()
    try:
        sys.path.insert(0, str(tooling))
        for generator in (
            "make_viz_recipes.py",
            "make_animation_showcase.py",
            "make_interactive_showcase.py",
        ):
            namespace = run_path(str(tooling / generator))
            failures.extend(namespace["validate_assets"]())
    finally:
        sys.path[:] = previous_path
    if failures:
        raise PluginError("Illustrated examples need regeneration:\n" + "\n".join(failures))


def highlight_code(source: str, language: str = "python") -> Markup:
    """Render copyable, escaped code with the site's server-side syntax theme."""
    lexer = get_lexer_by_name(language)
    formatter = HtmlFormatter(
        cssclass=f"language-{lexer.aliases[0]} highlight",
        wrapcode=True,
        filename=lexer.name,
    )
    return Markup(highlight(dedent(source).strip("\n") + "\n", lexer, formatter))


def on_env(env, *, config, files):
    """Make syntax highlighting available to Jinja page templates."""
    env.filters["tsd_highlight"] = highlight_code
    return env
