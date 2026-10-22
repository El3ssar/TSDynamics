"""Build-time presentation helpers for documentation templates.

Markdown fences are highlighted by SuperFences. Template-only examples use the
same Pygments token vocabulary so they remain highlighted without JavaScript or
a network request, including when navigating with Material's instant loading.
"""

from textwrap import dedent

from markupsafe import Markup
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name


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
