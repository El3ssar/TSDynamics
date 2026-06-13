"""
MkDocs hook: auto-generate the system catalogue at build time.

For every system in :mod:`tsdynamics.registry` this hook generates

- one page per system (title, summary, LaTeX equations rendered from the
  actual symbolic definition, parameter table, literature reference, phase
  portrait, usage snippet),
- one index page per category, and
- navigation entries under the three family sections,

and renders the phase-portrait figures with an on-disk content-hash cache
(``.cache/docs-figures``) so unchanged systems never re-render.

Adding a system to the library therefore adds its documentation with zero
manual steps.  Set ``TSD_DOCS_FIGURES=0`` for a fast figure-less preview
build (``mkdocs serve``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mkdocs.structure.files import File

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "docs" / "_tooling"))

import equations as _equations  # noqa: E402  (docs/_tooling)
import figures as _figures  # noqa: E402  (docs/_tooling)

WITH_FIGURES = os.environ.get("TSD_DOCS_FIGURES", "1") != "0"

FAMILY_DIRS = {"ode": "systems/continuous", "dde": "systems/delay", "map": "systems/discrete"}
FAMILY_LABELS = {"ode": "Continuous", "dde": "Delay", "map": "Discrete"}

#: uri → markdown source for every generated page.
_GENERATED: dict[str, str] = {}
_VERSION = "?"


def _slug(category: str) -> str:
    return category.replace("_", "-")


def _title(category: str) -> str:
    return category.replace("_", " ").capitalize()


def _summary(cls) -> str:
    doc = (cls.__doc__ or "").strip()
    if not doc:
        return ""
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    return " ".join(first.split())


def _system_page(entry, fig_path: Path | None) -> str:
    family_dir = FAMILY_DIRS[entry.family]
    kicker = f"Systems · {FAMILY_LABELS[entry.family]} · {_title(entry.category)}"
    parts: list[str] = [f"# {entry.name}", "", f"<small>{kicker}</small>", ""]

    summary = _summary(entry.cls)
    if summary:
        parts += [summary, ""]

    facts = [f"**Dimension:** {entry.dim if entry.dim is not None else 'variable'}"]
    if entry.reference:
        facts.append(f"**Reference:** {entry.reference}")
    parts += [" · ".join(facts), ""]

    parts += ["## Equations", "", _equations.equations_markdown(entry), ""]

    if entry.params:
        parts += ["## Parameters", "", "| parameter | default |", "|---|---|"]
        parts += [f"| `{k}` | `{v!r}` |" for k, v in entry.params.items()]
        parts.append("")
        if getattr(entry.cls, "variables", None):
            parts += [f"**Variables:** `{', '.join(entry.cls.variables)}`", ""]

    if fig_path is not None:
        depth = family_dir.count("/") + 2  # family dir + category dir
        rel = "../" * depth + f"assets/figures/systems/{entry.name}.png"
        parts += [
            '<figure markdown="span">',
            f'  <img src="{rel}" alt="{entry.name} attractor" loading="lazy">',
            "</figure>",
            "",
        ]

    call = (
        "iterate(steps=10_000)" if entry.family == "map" else "integrate(final_time=100.0, dt=0.01)"
    )
    parts += [
        "## Usage",
        "",
        "```python",
        "import tsdynamics as ts",
        "",
        f"sys = ts.{entry.name}()",
        f"traj = sys.{call}",
        "```",
        "",
        f"[Back to {_title(entry.category)}](index.md)",
        "",
    ]
    return "\n".join(parts)


def _category_index(family: str, category: str, entries: list) -> str:
    parts = [
        f"# {_title(category)}",
        "",
        f"<small>Systems · {FAMILY_LABELS[family]}</small>",
        "",
        f"{len(entries)} systems in this category.",
        "",
        "| system | dim | summary |",
        "|---|---|---|",
    ]
    for e in entries:
        summary = _summary(e.cls).split(". ")[0][:90]
        dim = e.dim if e.dim is not None else "n"
        parts.append(f"| [{e.name}]({e.name}.md) | {dim} | {summary} |")
    parts.append("")
    return "\n".join(parts)


def _patch_nav(nav: list, family_index: str, sections: list) -> bool:
    """Find the nav list containing ``family_index`` and extend it in place."""
    for item in nav:
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, list):
                    if family_index in value:
                        value.extend(sections)
                        return True
                    if _patch_nav(value, family_index, sections):
                        return True
    return False


def on_config(config):
    """Generate pages + figures before MkDocs scans the docs directory."""
    global _VERSION

    import tsdynamics
    from tsdynamics import registry

    _VERSION = tsdynamics.__version__
    _GENERATED.clear()

    skipped_figures: list[str] = []

    for family, family_dir in FAMILY_DIRS.items():
        categories: dict[str, list] = {}
        for entry in registry.all_systems(family=family):
            categories.setdefault(entry.category, []).append(entry)

        family_sections: list = []
        for category in sorted(categories):
            entries = sorted(categories[category], key=lambda e: e.name)
            cat_dir = f"{family_dir}/{_slug(category)}"

            _GENERATED[f"{cat_dir}/index.md"] = _category_index(family, category, entries)
            section_items: list = [f"{cat_dir}/index.md"]

            for entry in entries:
                fig = _figures.render(entry) if WITH_FIGURES else None
                if WITH_FIGURES and fig is None:
                    skipped_figures.append(entry.name)
                _GENERATED[f"{cat_dir}/{entry.name}.md"] = _system_page(entry, fig)
                section_items.append({entry.name: f"{cat_dir}/{entry.name}.md"})

            family_sections.append({_title(category): section_items})

        if config.nav and not _patch_nav(config.nav, f"{family_dir}/index.md", family_sections):
            print(f"docs_autogen: WARNING — nav anchor {family_dir}/index.md not found")

    n_pages = len(_GENERATED)
    print(f"docs_autogen: generated {n_pages} pages", end="")
    if WITH_FIGURES:
        print(f", figures skipped for {len(skipped_figures)}: {skipped_figures or 'none'}")
    else:
        print(" (figures disabled via TSD_DOCS_FIGURES=0)")
    return config


def on_files(files, config):
    """Register the generated pages as virtual files."""
    for uri, content in _GENERATED.items():
        files.append(File.generated(config, uri, content=content))
    return files


def on_page_markdown(markdown, page, config, files):
    """Substitute build-time tokens."""
    return markdown.replace("{{ tsdynamics_version }}", _VERSION)
