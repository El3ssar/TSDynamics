r"""
MkDocs hook: generate the **Systems** catalogue at build time, in the new IA.

The information architecture is **type → subcategory → system**:

    Systems  (systems/index.md)
      ├─ ODEs   (systems/ode/index.md)
      │    ├─ Chaotic attractors   (systems/ode/chaotic_attractors/index.md)
      │    │    ├─ Lorenz          (systems/ode/chaotic_attractors/Lorenz.md)
      │    │    └─ …
      │    └─ …
      ├─ DDEs   (systems/dde/index.md)
      ├─ SDEs   (systems/sde/index.md)
      └─ Maps   (systems/maps/index.md)

Every fact (counts, dimensions, families, categories) is derived from the live
:mod:`tsdynamics.registry` through the :mod:`catalog` merge layer — never
hardcoded.  Editorial decoration (blurbs, parameter roles, projections,
behaviour tags) comes from :file:`docs/_tooling/editorial.json` and only ever
*adds* to the registry facts.

For every registry system this hook generates one rich page (title + subtitle +
tag pills, an interactive three.js attractor — or a static field/projection
figure for spatial systems — equations, a parameter table, the four computed
property cards, a real-API code block, and a literature reference card) and the
three browser tiers (the Systems index, one page per type, one page per
subcategory).  It patches ``config.nav`` so the whole tree is navigable with
``navigation.indexes`` (each ``index.md`` is its section landing).

Rendering is delegated to the Phase-0 foundation modules under
:mod:`docs/_tooling` — :mod:`catalog` (IA + editorial merge), :mod:`equations`
(symbolic → LaTeX), :mod:`figures` (cached static figures), :mod:`threejs_viewer`
(cached interactive viewers), and :mod:`properties` (cached stat cards) — so this
hook is pure orchestration.  Adding a system to the library therefore adds its
documentation with zero manual steps.

Environment flags
-----------------
``TSD_DOCS_FIGURES=0``   skip every heavy render (viewers + figures); the pages
                         still generate with a placeholder where the attractor
                         would be.
``TSD_DOCS_ONLY=A,B,…``  build per-system pages only for the named systems (the
                         browser/index/type/subcategory pages are always built so
                         the nav stays whole) — a fast single-page preview.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mkdocs.structure.files import File

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "docs" / "_tooling"))

import catalog as _catalog  # noqa: E402  (docs/_tooling)
import equations as _equations  # noqa: E402
import figures as _figures  # noqa: E402
import plot_dt as _plot_dt  # noqa: E402  (re-exported for downstream tuning)
import properties as _properties  # noqa: E402
import threejs_viewer as _viewer  # noqa: E402

# Silence "imported but unused" for the re-exported tuning hook — it is part of
# the foundation surface and may be wired into figures/viewer dt selection.
_ = _plot_dt

WITH_FIGURES = os.environ.get("TSD_DOCS_FIGURES", "1") != "0"

#: Optional comma-separated allow-list of system *names* — a fast preview that
#: builds only those systems' per-system pages.  The browser pages (index / type
#: / subcategory) are always built so the nav and counts stay whole.
_ONLY = {n.strip() for n in os.environ.get("TSD_DOCS_ONLY", "").split(",") if n.strip()}

#: Where the Systems tree is rooted in the docs site.
_SYSTEMS_ROOT = "systems"

# ---------------------------------------------------------------------------
# Build-time state (uri → source) populated in ``on_config``.
# ---------------------------------------------------------------------------
#: uri → markdown source for every generated page.
_GENERATED: dict[str, str] = {}
#: uri → HTML for every generated interactive three.js viewer.
_VIEWERS: dict[str, str] = {}
_VERSION = "?"


# ===========================================================================
# Small helpers
# ===========================================================================
def _slug(category: str) -> str:
    """URL slug for a registry category (module stem) — kept underscore-free."""
    return category.replace("_", "-")


def _system_dir(rec) -> str:
    """``systems/<type>/<subcat-slug>`` — the directory holding a system page."""
    return f"{_SYSTEMS_ROOT}/{_catalog.type_slug(rec.family)}/{_slug(rec.category)}"


def _system_uri(rec) -> str:
    """``systems/<type>/<subcat-slug>/<Name>.md`` — the per-system page uri."""
    return f"{_system_dir(rec)}/{rec.name}.md"


def _rel(from_uri: str, to_root_path: str) -> str:
    """Site-root-relative path ``to_root_path`` as seen from page ``from_uri``.

    MkDocs rewrites relative ``![]()`` / ``[]()`` links against the *source* uri,
    so the number of ``../`` is the page's directory depth.  ``systems/a/b/X.md``
    sits three directories deep → three ``../`` reach the site root.
    """
    depth = from_uri.count("/")
    return "../" * depth + to_root_path


def _subtitle(rec) -> str:
    """One-line subtitle: the editorial blurb, else the class docstring's lead."""
    if rec.blurb:
        return rec.blurb
    doc = (rec.cls.__doc__ or "").strip()
    if not doc:
        return ""
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    return " ".join(first.split())


def _dim_label(rec) -> str:
    """``"3 dimensions"`` / ``"1 dimension"`` / ``"variable dimension"``."""
    if rec.dim is None:
        return "variable dimension"
    return f"{rec.dim} dimension" + ("" if rec.dim == 1 else "s")


def _dim_cell(rec) -> str:
    """Compact dimension for a table cell (``"N"`` for variable-dim)."""
    return "N" if rec.dim is None else str(rec.dim)


def _continuity_word(family: str) -> str:
    """``continuous`` / ``delay`` / ``stochastic`` / ``discrete`` lead tag word."""
    return {
        "ode": "continuous",
        "dde": "delay",
        "sde": "stochastic",
        "map": "discrete",
    }.get(family, "continuous")


def _family_short(family: str) -> str:
    """Short family tag (``ODE`` / ``DDE`` / ``SDE`` / ``map``)."""
    return {"ode": "ODE", "dde": "DDE", "sde": "SDE", "map": "map"}.get(family, family.upper())


def _pill(text: str, *, accent: bool = False) -> str:
    """Return a tag pill ``<span>`` — the brand ``.system-tag`` (teal) styling."""
    cls = "system-tag system-tag--accent" if accent else "system-tag"
    return f'<span class="{cls}">{text}</span>'


def _behavior_label(rec) -> str:
    """Return a short behaviour word for a table cell (first editorial tag, else ``—``)."""
    if rec.behavior:
        return rec.behavior[0]
    return "—"


# ===========================================================================
# Per-system page
# ===========================================================================
def _tag_pills(rec) -> str:
    """Build the header tag pills: type · family, dimension, behaviour, geometry."""
    pills = [_pill(f"{_continuity_word(rec.family)} · {_family_short(rec.family)}", accent=True)]
    pills.append(_pill(_dim_label(rec)))
    for tag in rec.behavior:
        pills.append(_pill(tag))
    return '<p class="ts-tags">' + "".join(pills) + "</p>"


def _attractor_block(rec, uri: str, has_viewer: bool, has_figure: bool) -> list[str]:
    """Build the attractor block: an interactive viewer iframe, a static figure, or a note."""
    if has_viewer:
        # The viewer iframe ``src`` is resolved against the *output* (directory-URL)
        # location: a page ``systems/<t>/<c>/<Name>.md`` serves at
        # ``systems/<t>/<c>/<Name>/`` (one deeper than the source), so reaching the
        # site-root ``assets/threejs/<Name>.html`` needs one extra ``../`` over the
        # source depth.  MkDocs does not rewrite ``<iframe src>``.
        depth = uri.count("/") + 1
        src = "../" * depth + f"assets/threejs/{rec.name}.html"
        title = f"{rec.name} attractor — drag to orbit, scroll to zoom (plays automatically)"
        iframe = (
            f'<iframe class="ts-attractor" src="{src}" loading="lazy" '
            f'title="{title}" scrolling="no"></iframe>'
        )
        caption = "interactive attractor · three.js — drag to orbit, scroll to zoom"
        return [
            '<figure class="ts-attractor-fig" markdown>',
            iframe,
            f'<figcaption class="ts-attractor-cap">{caption}</figcaption>',
            "</figure>",
            "",
        ]
    if has_figure:
        rel = _rel(uri, f"assets/figures/systems/{rec.name}.png")
        cap = "spatiotemporal field" if rec.is_spatial else "phase portrait"
        return [
            '<figure class="ts-attractor-fig" markdown>',
            f"![{rec.name} {cap}]({rel}){{ loading=lazy .ts-attractor-img }}",
            f'<figcaption class="ts-attractor-cap">{cap} · matplotlib</figcaption>',
            "</figure>",
            "",
        ]
    return [
        '!!! note "Attractor figure skipped"',
        "    Build with figures enabled (`TSD_DOCS_FIGURES=1`) to render the",
        "    interactive attractor / static field image for this system.",
        "",
    ]


def _fmt_default(v) -> str:
    """Render a parameter default cleanly (no ``2.6666666666666665`` floats)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:.6g}"
    return repr(v)


def _parameter_table(rec) -> list[str]:
    """Build a SYMBOL / DEFAULT / ROLE table; role from editorial ``param_roles``."""
    if not rec.params:
        return []
    parts = ["## Parameters", "", "| Symbol | Default | Role |", "|---|---|---|"]
    for key, val in rec.params.items():
        role = rec.param_roles.get(key, "")
        parts.append(f"| `{key}` | `{_fmt_default(val)}` | {role} |")
    parts.append("")
    if rec.variables:
        parts += [f"**State variables:** `{', '.join(rec.variables)}`", ""]
    if rec.is_field and rec.field_labels:
        parts += [f"**Field blocks:** `{', '.join(rec.field_labels)}`", ""]
    return parts


def _define_block(rec) -> list[str]:
    """Build a "Define it in TSDynamics" code block using the **real** library API."""
    if rec.family == "map":
        run = "traj = sys.iterate(steps=10_000)"
    elif rec.family == "dde":
        run = "traj = sys.integrate(final_time=500.0, dt=0.5)"
    else:
        run = "traj = sys.integrate(final_time=100.0, dt=0.01)"

    lines = [
        "## Define it in TSDynamics",
        "",
        "```python",
        "import tsdynamics as ts",
        "",
        f"sys = ts.systems.{rec.name}()",
        run,
    ]
    # A Lyapunov line where it is meaningful (flows + maps; not DDE/SDE — those have
    # their own dedicated estimators, shown elsewhere).
    if rec.family in ("ode", "map"):
        lines += [
            "",
            "exps = sys.lyapunov_spectrum()",
            "ts.kaplan_yorke_dimension(exps)",
        ]
    elif rec.family == "dde":
        lines += [
            "",
            "# DDE Lyapunov uses the infinite-dimensional-history estimator:",
            "exps = sys.lyapunov_spectrum(n_exp=1, dt=0.5, ic=traj.y[-1])",
        ]
    lines += ["```", ""]
    return lines


def _sde_equations_md(rec) -> str:
    r"""LaTeX for an Itô SDE ``dX_k = f_k dt + g_k dW_k`` (drift + diffusion).

    :mod:`equations` only renders ``_equations``-bearing families (ODE/DDE/map);
    a :class:`StochasticSystem` carries ``_drift`` + ``_diffusion`` instead, so the
    SDE block is built here.  Falls back to a source fence if symbolic lowering
    fails (a NumPy body, a variable-dim system, …) — never raises.
    """
    try:
        import symengine
        import sympy

        sys_obj = rec.cls()
        dim = sys_obj.dim
        if dim is None or dim > 8:
            raise ValueError("dimension too large for symbolic rendering")
        names = (
            list(rec.variables)
            if rec.variables and len(rec.variables) == dim
            else [f"y_{{{i}}}" for i in range(dim)]
        )
        syms = [symengine.Symbol(n) for n in names]
        t = symengine.Symbol("t")
        structural = getattr(rec.cls, "_structural_params", frozenset())
        params = {
            k: (v if k in structural else symengine.Symbol(k)) for k, v in sys_obj.params.items()
        }

        def y(i):
            return syms[int(i)]

        drift = list(rec.cls._drift(y, t, **params))
        diffusion = list(rec.cls._diffusion(y, t, **params))
        lines = []
        for n, f_k, g_k in zip(names, drift, diffusion, strict=True):
            f_latex = sympy.latex(symengine.sympify(f_k)._sympy_())
            g_latex = sympy.latex(symengine.sympify(g_k)._sympy_())
            lines.append(
                rf"d{n} &= \left({f_latex}\right)\,dt + \left({g_latex}\right)\,dW_{{{n}}}"
            )
        body = " \\\\\n".join(lines)
        return f"$$\n\\begin{{aligned}}\n{body}\n\\end{{aligned}}\n$$"
    except Exception:  # noqa: BLE001 — fall back to the source body
        try:
            import inspect
            import textwrap

            drift_src = textwrap.dedent(inspect.getsource(rec.cls.__dict__["_drift"]))
            diff_src = textwrap.dedent(inspect.getsource(rec.cls.__dict__["_diffusion"]))
            return f"```python\n{drift_src}\n{diff_src}```"
        except Exception:  # noqa: BLE001
            return "_Equations could not be rendered — see the source._"


def _equations_md(rec) -> str:
    """Equations block for any family — SDE via the local drift/diffusion renderer."""
    if rec.family == "sde":
        return _sde_equations_md(rec)
    return _equations.equations_markdown(rec)


def _bibtex(rec) -> str | None:
    """Return a minimal BibTeX ``@misc`` entry for the system's reference, or ``None``."""
    if not rec.reference:
        return None
    key = rec.name.lower()
    fields = [f"  title = {{{rec.name} system}}", f"  note = {{{rec.reference}}}"]
    if rec.doi:
        fields.append(f"  doi = {{{rec.doi}}}")
    body = ",\n".join(fields)
    return f"@misc{{{key},\n{body}\n}}"


def _reference_block(rec) -> list[str]:
    """Build a reference card: citation + DOI link + a collapsible BibTeX block."""
    if not rec.reference:
        return []
    parts = ["## Reference", ""]
    cite = rec.reference
    if rec.doi:
        doi_url = f"https://doi.org/{rec.doi}"
        cite_line = f"{cite}  \n[doi:{rec.doi}]({doi_url})"
    else:
        cite_line = cite
    parts += [
        '<div class="ts-reference" markdown>',
        cite_line,
        "</div>",
        "",
    ]
    bib = _bibtex(rec)
    if bib:
        parts += [
            '??? quote "BibTeX"',
            "    ```bibtex",
            *[f"    {line}" for line in bib.splitlines()],
            "    ```",
            "",
        ]
    return parts


def _system_page(rec, *, has_viewer: bool, has_figure: bool) -> str:
    """Build the full markdown source for one system's page."""
    uri = _system_uri(rec)
    crumb = (
        f"[Systems]({_rel(uri, _SYSTEMS_ROOT + '/index.md')}) "
        f"/ [{rec.type_label}]({_rel(uri, _SYSTEMS_ROOT + '/' + _catalog.type_slug(rec.family) + '/index.md')}) "
        f"/ {rec.subcategory_label}"
    )

    parts: list[str] = [f"# {rec.name}", "", f'<p class="ts-kicker">{crumb}</p>', ""]

    subtitle = _subtitle(rec)
    if subtitle:
        parts += [f'<p class="ts-subtitle">{subtitle}</p>', ""]

    parts += [_tag_pills(rec), ""]

    # Attractor (viewer / figure / note).
    parts += _attractor_block(rec, uri, has_viewer, has_figure)

    # Definition: prose lead-in + the symbolic equations.
    parts += ["## Definition", ""]
    parts += [_equations_md(rec), ""]

    # Parameters.
    parts += _parameter_table(rec)

    # Properties (computed stat cards — Lyapunov / Kaplan–Yorke / divergence /
    # equilibria, each with a graceful TODO where ill-defined or too slow).
    parts += ["## Properties", ""]
    try:
        props = _properties.compute_properties(rec)
        parts += [_properties.to_markdown(props), ""]
    except Exception as exc:  # noqa: BLE001 — a property card must never break a build
        parts += [f"_Properties unavailable ({type(exc).__name__})._", ""]

    # Define it (real API).
    parts += _define_block(rec)

    # Reference.
    parts += _reference_block(rec)

    return "\n".join(parts)


# ===========================================================================
# Browser pages: Systems index → type → subcategory
# ===========================================================================
def _examples(records, n: int = 3) -> str:
    """Comma-joined first ``n`` system names (featured first), with an ellipsis."""
    ordered = sorted(records, key=lambda r: (not r.featured, r.name))
    names = [r.name for r in ordered[:n]]
    suffix = " …" if len(records) > n else ""
    return ", ".join(names) + suffix


def _dim_range(records) -> str:
    """Dimension range across a record set (``"3"`` / ``"2–4"`` / ``"2–4, N"``)."""
    dims = sorted({r.dim for r in records if r.dim is not None})
    has_var = any(r.dim is None for r in records)
    if not dims:
        return "N"
    span = str(dims[0]) if len(dims) == 1 else f"{dims[0]}–{dims[-1]}"
    return span + (", N" if has_var else "")


def _systems_index(catalog) -> str:
    """Build the Systems landing: intro + three derived stats + tiny example + type table."""
    counts = catalog.counts()
    grouped = catalog.by_type()
    total = counts["total"]
    n_types = len(grouped)
    n_subs = sum(len(cats) for cats in grouped.values())

    parts = [
        "# Systems",
        "",
        '<p class="ts-subtitle">A curated library of canonical dynamical systems — '
        "each one defined, integrated, and analysis-ready out of the box. Browse by "
        "type, or drop straight into a definition and start simulating.</p>",
        "",
        '<div class="ts-stats" markdown>',
        f'<div class="ts-stat"><div class="ts-stat-num">{total}</div>'
        '<div class="ts-stat-label">SYSTEMS</div></div>',
        f'<div class="ts-stat"><div class="ts-stat-num">{n_types}</div>'
        '<div class="ts-stat-label">TYPES</div></div>',
        f'<div class="ts-stat"><div class="ts-stat-num">{n_subs}</div>'
        '<div class="ts-stat-label">SUBCATEGORIES</div></div>',
        "</div>",
        "",
        "## One system, start to finish",
        "",
        "```python",
        "import tsdynamics as ts",
        "",
        "lorenz = ts.systems.Lorenz()",
        "traj = lorenz.integrate(final_time=100.0, dt=0.01)   # trajectory → attractor",
        "exps = lorenz.lyapunov_spectrum()                    # [≈ +0.91, 0, ≈ −14.57]",
        "ts.kaplan_yorke_dimension(exps)                      # ≈ 2.06",
        "```",
        "",
        "## Browse by type",
        "",
        "| Type | Subcategories | Systems | Examples |",
        "|---|--:|--:|---|",
    ]
    for family, cats in grouped.items():
        recs = [r for cat in cats.values() for r in cat]
        slug = _catalog.type_slug(family)
        label = _catalog.type_label(family)
        link = f"[{label}]({slug}/index.md)"
        parts.append(f"| {link} | {len(cats)} | {len(recs)} | {_examples(recs)} |")
    parts.append("")
    return "\n".join(parts)


def _type_index(family: str, cats: dict) -> str:
    """Build one type page: blurb + the real define-a-system block + subcategory table."""
    label = _catalog.type_label(family)
    n_systems = sum(len(recs) for recs in cats.values())

    parts = [
        f"# {label}",
        "",
        f'<p class="ts-kicker">[Systems](../index.md) / {label}</p>',
        "",
    ]
    blurb = _catalog.type_blurb(family)
    if blurb:
        parts += [f'<p class="ts-subtitle">{blurb}</p>', ""]
    definition = _catalog.type_definition(family)
    if definition:
        parts += [f"$$ {definition} $$", ""]

    parts += [
        f'<p class="ts-count">{n_systems} systems · {len(cats)} subcategories</p>',
        "",
        "## Define one",
        "",
        "```python",
        *_type_define_snippet(family),
        "```",
        "",
        "## Subcategories",
        "",
        "| Subcategory | Systems | Dim | Examples |",
        "|---|--:|--:|---|",
    ]
    for category, recs in cats.items():
        cat_slug = _slug(category)
        cat_label = _catalog.subcategory_label(category)
        link = f"[{cat_label}]({cat_slug}/index.md)"
        parts.append(f"| {link} | {len(recs)} | {_dim_range(recs)} | {_examples(recs)} |")
    parts.append("")
    return "\n".join(parts)


def _type_define_snippet(family: str) -> list[str]:
    """Return a real-API "how you'd define one of these" snippet for a type page."""
    if family == "ode":
        return [
            "import tsdynamics as ts",
            "",
            "# Pick any continuous system and integrate it:",
            "sys = ts.systems.Lorenz()",
            "traj = sys.integrate(final_time=100.0, dt=0.01)",
        ]
    if family == "dde":
        return [
            "import numpy as np",
            "import tsdynamics as ts",
            "",
            "# A delay system carries its own delay τ; supply a past history:",
            "sys = ts.systems.MackeyGlass()",
            "traj = sys.integrate(",
            "    final_time=500.0, dt=0.5,",
            "    history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)],",
            ")",
        ]
    if family == "sde":
        return [
            "import tsdynamics as ts",
            "",
            "# A stochastic system integrates with a seeded noise realisation:",
            "sys = ts.systems.OrnsteinUhlenbeck()",
            "traj = sys.integrate(final_time=100.0, dt=0.01, seed=0)",
        ]
    # map
    return [
        "import tsdynamics as ts",
        "",
        "# Discrete maps iterate — no integration step required:",
        "sys = ts.systems.Henon()",
        "traj = sys.iterate(steps=10_000)",
        "exps = sys.lyapunov_spectrum()",
    ]


def _subcategory_index(family: str, category: str, records, generated: set[str]) -> str:
    """One subcategory page: blurb + a System / Dim / Behavior table of its systems.

    A system links to its page only when that page was generated (always true in a
    full build; under ``TSD_DOCS_ONLY`` a non-generated system renders as plain text
    so the browser page stays whole without a broken link).
    """
    label = _catalog.subcategory_label(category)
    type_label = _catalog.type_label(family)

    parts = [
        f"# {label}",
        "",
        f'<p class="ts-kicker">[Systems](../../index.md) '
        f"/ [{type_label}](../index.md) / {label}</p>",
        "",
    ]
    blurb = _catalog.subcategory_blurb(category)
    if blurb:
        parts += [f'<p class="ts-subtitle">{blurb}</p>', ""]
    parts += [
        f'<p class="ts-count">{len(records)} systems</p>',
        "",
        "| System | Dim | Behavior |",
        "|---|--:|---|",
    ]
    for rec in records:
        name = f"[{rec.name}]({rec.name}.md)" if rec.name in generated else rec.name
        parts.append(f"| {name} | {_dim_cell(rec)} | {_behavior_label(rec)} |")
    parts.append("")
    return "\n".join(parts)


# ===========================================================================
# Nav patching
# ===========================================================================
def _build_nav(catalog) -> list:
    """Build the ``Systems`` nav subtree (index + type/subcategory/system sections).

    Uses ``navigation.indexes`` — the first entry of each section list is its
    ``index.md`` landing page.  Per-system pages omitted by ``TSD_DOCS_ONLY``
    simply do not appear (the browser pages stay whole).
    """
    grouped = catalog.by_type()
    type_sections: list = []
    for family, cats in grouped.items():
        slug = _catalog.type_slug(family)
        type_items: list = [f"{_SYSTEMS_ROOT}/{slug}/index.md"]
        for category in cats:
            cat_slug = _slug(category)
            cat_dir = f"{_SYSTEMS_ROOT}/{slug}/{cat_slug}"
            cat_items: list = [f"{cat_dir}/index.md"]
            for rec in cats[category]:
                if _ONLY and rec.name not in _ONLY:
                    continue
                cat_items.append({rec.name: _system_uri(rec)})
            cat_items_label = _catalog.subcategory_label(category)
            type_sections_inner = {cat_items_label: cat_items}
            type_items.append(type_sections_inner)
        type_sections.append({_catalog.type_label(family): type_items})
    return [f"{_SYSTEMS_ROOT}/index.md", *type_sections]


def _patch_systems_nav(nav: list, subtree: list) -> bool:
    """Replace the ``Systems`` section's children with the generated ``subtree``.

    Matches the top-level ``{"Systems": [...]}`` entry (or any nesting level) and
    swaps its list for the freshly generated tree.  Returns ``True`` on success.
    """
    for item in nav:
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "Systems" and isinstance(value, list):
                    item[key] = subtree
                    return True
                if isinstance(value, list) and _patch_systems_nav(value, subtree):
                    return True
    return False


# ===========================================================================
# MkDocs hooks
# ===========================================================================
def on_config(config):
    """Generate every Systems page + its figures/viewers; patch the nav."""
    global _VERSION

    import tsdynamics

    _VERSION = tsdynamics.__version__
    _GENERATED.clear()
    _VIEWERS.clear()

    # Build the merged catalogue fresh (registry → effective dims → editorial).
    _catalog.load_catalog.cache_clear()
    catalog = _catalog.load_catalog()
    grouped = catalog.by_type()

    # --- per-system tier (first, so the browser tables know what was built) -
    viewer_names: list[str] = []
    figure_names: list[str] = []
    skipped: list[str] = []
    generated_systems: set[str] = set()

    for cats in grouped.values():
        for records in cats.values():
            for rec in records:
                if _ONLY and rec.name not in _ONLY:
                    continue

                viewer_html = _viewer.render_html(rec) if WITH_FIGURES else None
                has_viewer = viewer_html is not None
                if has_viewer:
                    _VIEWERS[f"assets/threejs/{rec.name}.html"] = viewer_html
                    viewer_names.append(rec.name)

                # The static figure is the page image for non-viewer systems (fields /
                # projections / stiff) AND the viewer's WebGL/no-JS fallback poster.
                has_figure = False
                if WITH_FIGURES:
                    fig = _figures.render(rec)
                    has_figure = fig is not None
                    if has_figure:
                        figure_names.append(rec.name)
                    elif not has_viewer:
                        skipped.append(rec.name)

                _GENERATED[_system_uri(rec)] = _system_page(
                    rec, has_viewer=has_viewer, has_figure=(has_figure and not has_viewer)
                )
                generated_systems.add(rec.name)

    n_pages_systems = len(generated_systems)

    # --- browser tier: index → type → subcategory --------------------------
    _GENERATED[f"{_SYSTEMS_ROOT}/index.md"] = _systems_index(catalog)
    for family, cats in grouped.items():
        slug = _catalog.type_slug(family)
        _GENERATED[f"{_SYSTEMS_ROOT}/{slug}/index.md"] = _type_index(family, cats)
        for category, records in cats.items():
            cat_slug = _slug(category)
            uri = f"{_SYSTEMS_ROOT}/{slug}/{cat_slug}/index.md"
            _GENERATED[uri] = _subcategory_index(family, category, records, generated_systems)

    # --- nav ---------------------------------------------------------------
    if config.nav is not None:
        subtree = _build_nav(catalog)
        if not _patch_systems_nav(config.nav, subtree):
            print("docs_autogen: WARNING — could not find a 'Systems' nav section to patch")

    # --- build summary -----------------------------------------------------
    counts = catalog.counts()
    by_fam = ", ".join(f"{k}:{v}" for k, v in counts["by_family"].items())
    print(
        f"docs_autogen: {len(_GENERATED)} pages "
        f"({n_pages_systems} system pages, {counts['total']} catalogued: {by_fam})"
    )
    if WITH_FIGURES:
        print(
            f"docs_autogen: {len(viewer_names)} interactive viewers, "
            f"{len(figure_names)} static figures"
            + (f", figures skipped for {len(skipped)}: {skipped}" if skipped else "")
        )
    else:
        print("docs_autogen: figures/viewers disabled (TSD_DOCS_FIGURES=0)")
    if _ONLY:
        print(f"docs_autogen: TSD_DOCS_ONLY restricted system pages to {sorted(_ONLY)}")
    return config


def on_files(files, config):
    """Register generated pages + viewers, replacing any physical placeholder file."""
    for uri, content in _GENERATED.items():
        existing = files.get_file_from_path(uri)
        if existing is not None:
            # A physical stub (e.g. the old ``systems/index.md``) would collide with
            # our generated page — drop it so the generated source wins.
            files.remove(existing)
        files.append(File.generated(config, uri, content=content))
    for uri, content in _VIEWERS.items():
        existing = files.get_file_from_path(uri)
        if existing is not None:
            files.remove(existing)
        files.append(File.generated(config, uri, content=content))
    return files


def on_page_markdown(markdown, page, config, files):
    """Substitute build-time tokens (the library version)."""
    return markdown.replace("{{ tsdynamics_version }}", _VERSION)
