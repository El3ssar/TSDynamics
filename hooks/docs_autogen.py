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

from mkdocs.structure.files import File, InclusionLevel

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "docs" / "_tooling"))

import catalog as _catalog  # noqa: E402  (docs/_tooling)
import equations as _equations  # noqa: E402
import field_movies as _field_movies  # noqa: E402
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
#: site uri → absolute source path for every generated field-movie asset
#: (the ``.mp4`` / ``.gif`` movie and its poster PNG — binary blobs registered
#: from disk, not in-memory strings).
_FIELD_MOVIE_ASSETS: dict[str, str] = {}
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


def _viewer_iframe(rec, uri: str, *, second: bool, projection_label: str) -> str:
    """One interactive-viewer ``<iframe>`` (the primary view, or the ``projection2``).

    The iframe ``src`` is resolved against the *output* (directory-URL) location: a
    page ``systems/<t>/<c>/<Name>.md`` serves at ``systems/<t>/<c>/<Name>/`` (one
    deeper than the source), so reaching the site-root ``assets/threejs/<Name>.html``
    needs one extra ``../`` over the source depth.  MkDocs does not rewrite
    ``<iframe src>``.  The second view lives at ``<Name>-b.html``.
    """
    depth = uri.count("/") + 1
    suffix = "-b" if second else ""
    src = "../" * depth + f"assets/threejs/{rec.name}{suffix}.html"
    title = (
        f"{rec.name} attractor{projection_label} — drag to orbit, "
        "scroll to zoom (plays automatically)"
    )
    return (
        f'<iframe class="ts-attractor" src="{src}" loading="lazy" '
        f'title="{title}" scrolling="no"></iframe>'
    )


def _projection_caption(rec, *, second: bool) -> str:
    """Return a human ``(x, y, z)`` projection label for a viewer caption, or ``""``."""
    proj = rec.projection2 if second else rec.projection
    if not proj:
        return ""
    names = list(rec.variables or [])
    default = ["x", "y", "z", "w", "v", "u"]

    def label(i):
        try:
            i = int(i)
        except (TypeError, ValueError):
            return str(i)
        if 0 <= i < len(names):
            return names[i]
        return default[i] if i < len(default) else f"y{i}"

    return " (" + ", ".join(label(i) for i in proj) + ")"


#: The exact interactive-viewer caption the maintainer asked for — nothing else.
_INTERACTIVE_CAPTION = "Interactive: drag to rotate"


def _static_caption(rec) -> str:
    """Return a short, honest caption for a *static* figure (no interactivity claim).

    Reads the render intent the way :mod:`figures` does — a map with a
    ``bifurcation`` override renders its return map beside a library-generated
    bifurcation diagram; an SDE is a sample path; a spatial system a field image;
    an editorial ``timeseries`` / ``polar`` view names itself; everything else is a
    phase portrait.
    """
    fig_opts = _figures.FIG_OVERRIDES.get(rec.name, {})
    map_opts = _figures.MAP_OVERRIDES.get(rec.name, {})
    viewer = rec.viewer if isinstance(rec.viewer, dict) else {}

    if rec.family == "sde":
        return "sample path"
    if rec.family == "map":
        if map_opts.get("bifurcation"):
            return "return map · bifurcation diagram"
        return "attractor"
    if rec.is_spatial or fig_opts.get("kind") in ("field", "spacetime"):
        return "spatiotemporal field"
    if fig_opts.get("kind") == "timeseries" or viewer.get("static_kind") == "timeseries":
        return "time series"
    return "phase portrait"


#: The caption under an animated spatial-field movie hero.
_FIELD_MOVIE_CAPTION = "spatiotemporal field — plays automatically"


def _attractor_block(
    rec,
    uri: str,
    has_viewer: bool,
    has_figure: bool,
    has_viewer2: bool = False,
    field_movie: str | None = None,
) -> list[str]:
    """Build the attractor block: a field movie, interactive viewer(s), a static figure, or a note.

    A **spatial-field system** (``_field_shape``) leads with its animated field
    **movie** — the ``kind="field"`` heatmap evolution embedded as an autoplaying
    ``<video>`` (its final-frame poster shown until it loads), exactly the hero the
    three.js viewer is for a 3-D flow.  Otherwise: a 4-D-plus flow with a second
    editorial ``projection2`` shows **two** animated viewers over different coordinate
    combinations; every interactive viewer carries the exact caption "Interactive:
    drag to rotate"; a static figure gets a short honest label.
    """
    if field_movie is not None:
        return [
            '<figure class="ts-attractor-fig ts-field-movie-fig" markdown>',
            field_movie,
            f'<figcaption class="ts-attractor-cap">{_FIELD_MOVIE_CAPTION}</figcaption>',
            "</figure>",
            "",
        ]
    if has_viewer:
        primary = _viewer_iframe(
            rec, uri, second=False, projection_label=_projection_caption(rec, second=False)
        )
        if has_viewer2:
            secondary = _viewer_iframe(
                rec, uri, second=True, projection_label=_projection_caption(rec, second=True)
            )
            cap1 = f"projection{_projection_caption(rec, second=False)}"
            cap2 = f"projection{_projection_caption(rec, second=True)}"
            return [
                '<div class="ts-attractor-pair">',
                '<figure class="ts-attractor-fig" markdown>',
                primary,
                f'<figcaption class="ts-attractor-cap">{cap1}</figcaption>',
                "</figure>",
                '<figure class="ts-attractor-fig" markdown>',
                secondary,
                f'<figcaption class="ts-attractor-cap">{cap2}</figcaption>',
                "</figure>",
                "</div>",
                f'<p class="ts-attractor-cap ts-attractor-cap--pair">{_INTERACTIVE_CAPTION}</p>',
                "",
            ]
        return [
            '<figure class="ts-attractor-fig" markdown>',
            primary,
            f'<figcaption class="ts-attractor-cap">{_INTERACTIVE_CAPTION}</figcaption>',
            "</figure>",
            "",
        ]
    if has_figure:
        rel = _rel(uri, f"assets/figures/systems/{rec.name}.png")
        cap = _static_caption(rec)
        return [
            '<figure class="ts-attractor-fig" markdown>',
            f"![{rec.name} {cap}]({rel}){{ loading=lazy .ts-attractor-img }}",
            f'<figcaption class="ts-attractor-cap">{cap}</figcaption>',
            "</figure>",
            "",
        ]
    if not WITH_FIGURES:
        # Fast preview only (``TSD_DOCS_FIGURES=0``): the render was skipped on
        # purpose, so tell the reader how to see it.
        return [
            '!!! note "Attractor figure skipped"',
            "    Build with figures enabled (`TSD_DOCS_FIGURES=1`) to render the",
            "    interactive attractor / static field image for this system.",
            "",
        ]
    # Figures are enabled but this system genuinely has no attractor image — say
    # nothing rather than the misleading "build with figures enabled" note.
    return []


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


def _out_rel(from_uri: str, to_root_path: str) -> str:
    """Site-root-relative path as seen from the *rendered* page of ``from_uri``.

    A source page ``systems/a/b/X.md`` serves at ``systems/a/b/X/`` (a directory
    URL, one level deeper than the source uri's directory depth), so a raw-HTML
    link — which MkDocs does **not** rewrite — needs one extra ``../`` over the
    source-relative :func:`_rel`.  Index pages (``…/index.md``) serve at their own
    directory, so their output depth equals the source depth; pass those through
    ``_rel`` instead.
    """
    depth = from_uri.count("/") + 1
    return "../" * depth + to_root_path


def _breadcrumb(rel_fn, links: list[tuple[str, str | None]]) -> str:
    r"""Build a rendered HTML breadcrumb ``<p class="ts-kicker">``.

    Each element is ``(label, target_root_path | None)``: a path yields a real
    ``<a href>`` (resolved site-root-relative via ``rel_fn`` so it links correctly
    at any depth), and ``None`` yields the plain trailing crumb.  Emitting real
    ``<a>`` anchors (rather than markdown ``[](…)`` inside raw HTML, which MkDocs
    does not process) is what makes the breadcrumb render as clickable links
    instead of literal ``[Systems](…)`` text.
    """
    sep = ' <span class="ts-crumb-sep">/</span> '
    crumbs = []
    for label, target in links:
        if target:
            crumbs.append(f'<a href="{rel_fn(target)}">{label}</a>')
        else:
            crumbs.append(f"<span>{label}</span>")
    return '<p class="ts-kicker">' + sep.join(crumbs) + "</p>"


def _system_page(
    rec,
    *,
    has_viewer: bool,
    has_figure: bool,
    has_viewer2: bool = False,
    field_movie: str | None = None,
) -> str:
    """Build the full markdown source for one system's page."""
    uri = _system_uri(rec)
    crumb = _breadcrumb(
        lambda p: _out_rel(uri, p),
        [
            ("Systems", _SYSTEMS_ROOT + "/index.html"),
            (rec.type_label, _SYSTEMS_ROOT + "/" + _catalog.type_slug(rec.family) + "/index.html"),
            (rec.subcategory_label, None),
        ],
    )

    parts: list[str] = [f"# {rec.name}", "", crumb, ""]

    subtitle = _subtitle(rec)
    if subtitle:
        parts += [f'<p class="ts-subtitle">{subtitle}</p>', ""]

    parts += [_tag_pills(rec), ""]

    # Attractor (field movie / viewer / figure / note).
    parts += _attractor_block(
        rec, uri, has_viewer, has_figure, has_viewer2=has_viewer2, field_movie=field_movie
    )

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

    uri = f"{_SYSTEMS_ROOT}/{_catalog.type_slug(family)}/index.md"
    crumb = _breadcrumb(
        lambda p: _rel(uri, p),
        [("Systems", _SYSTEMS_ROOT + "/index.html"), (label, None)],
    )
    parts = [
        f"# {label}",
        "",
        crumb,
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
    """One subcategory page: blurb + a two-column System / Dimensions table.

    A system links to its page only when that page was generated (always true in a
    full build; under ``TSD_DOCS_ONLY`` a non-generated system renders as plain text
    so the browser page stays whole without a broken link).

    The table is exactly ``System | Dimensions`` — the old ``Behaviour`` column
    (a mix of editorial tags with stray letters where a bare-string behaviour tag
    had been spread character-by-character) is dropped.
    """
    label = _catalog.subcategory_label(category)
    type_label = _catalog.type_label(family)
    uri = f"{_SYSTEMS_ROOT}/{_catalog.type_slug(family)}/{_slug(category)}/index.md"

    crumb = _breadcrumb(
        lambda p: _rel(uri, p),
        [
            ("Systems", _SYSTEMS_ROOT + "/index.html"),
            (type_label, _SYSTEMS_ROOT + "/" + _catalog.type_slug(family) + "/index.html"),
            (label, None),
        ],
    )
    parts = [
        f"# {label}",
        "",
        crumb,
        "",
    ]
    blurb = _catalog.subcategory_blurb(category)
    if blurb:
        parts += [f'<p class="ts-subtitle">{blurb}</p>', ""]
    parts += [
        f'<p class="ts-count">{len(records)} systems</p>',
        "",
        "| System | Dimensions |",
        "|---|--:|",
    ]
    for rec in records:
        name = f"[{rec.name}]({rec.name}.md)" if rec.name in generated else rec.name
        parts.append(f"| {name} | {_dim_cell(rec)} |")
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
    movie_names: list[str] = []
    skipped: list[str] = []
    generated_systems: set[str] = set()
    _FIELD_MOVIE_ASSETS.clear()

    for cats in grouped.values():
        for records in cats.values():
            for rec in records:
                if _ONLY and rec.name not in _ONLY:
                    continue

                # ``viewer_payloads`` is the generator front door: it returns EVERY
                # interactive view for a system in one call, primary first — a lone
                # ``[{"suffix": ""}]`` for a 3-D flow / DDE / low-dim state, plus a
                # ``{"suffix": "-b"}`` second projection for a 4-D-plus flow that
                # declares an editorial ``projection2`` (whose primary rendered).  A
                # second view is never emitted without its primary, so the page can
                # never show a lone "-b" attractor.
                payloads = _viewer.viewer_payloads(rec) if WITH_FIGURES else []
                has_viewer = bool(payloads)
                has_viewer2 = False
                if has_viewer:
                    viewer_names.append(rec.name)
                    for view in payloads:
                        suffix = view["suffix"]
                        _VIEWERS[f"assets/threejs/{rec.name}{suffix}.html"] = view["html"]
                        if suffix == "-b":
                            has_viewer2 = True

                # The static figure is the page image for non-viewer systems (fields /
                # maps / stiff) AND the viewer's WebGL/no-JS fallback poster.
                has_figure = False
                if WITH_FIGURES:
                    fig = _figures.render(rec)
                    has_figure = fig is not None
                    if has_figure:
                        figure_names.append(rec.name)
                    elif not has_viewer:
                        skipped.append(rec.name)

                # A spatial-field system (``_field_shape``) leads with its animated
                # field **movie** (the ``kind="field"`` heatmap evolution) instead of
                # the static field PNG — the field analogue of the three.js hero.
                # Registered as generated binary assets in ``on_files``; the static
                # figure stays as the movie's no-video / TSD_DOCS_FIGURES=0 fallback.
                field_movie_html: str | None = None
                if WITH_FIGURES:
                    movie = _field_movies.render(rec)
                    if movie is not None:
                        uri = _system_uri(rec)
                        key = _field_movies.cache_key(rec)
                        movie_uri = f"assets/field-movies/{rec.name}-{key}.{movie['ext']}"
                        _FIELD_MOVIE_ASSETS[movie_uri] = str(movie["movie"])
                        poster_uri: str | None = None
                        if movie["poster"] is not None:
                            poster_uri = f"assets/field-movies/{rec.name}-{key}.png"
                            _FIELD_MOVIE_ASSETS[poster_uri] = str(movie["poster"])
                        field_movie_html = _field_movies.embed_html(
                            rec,
                            uri,
                            movie_uri=movie_uri,
                            poster_uri=poster_uri,
                            ext=movie["ext"],
                        )
                        movie_names.append(rec.name)

                _GENERATED[_system_uri(rec)] = _system_page(
                    rec,
                    has_viewer=has_viewer,
                    has_figure=(has_figure and not has_viewer),
                    has_viewer2=has_viewer2,
                    field_movie=field_movie_html,
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
            f"{len(figure_names)} static figures, "
            f"{len(movie_names)} field movies"
            + (f" ({movie_names})" if movie_names else "")
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
    # Field-movie binary assets (``.mp4`` / ``.gif`` + poster PNG) — registered from
    # their cached path on disk (``abs_src_path``), never as in-memory strings.  Each
    # is INCLUDED so mkdocs copies it into the site even though it lives outside the
    # docs tree; the per-system ``<video>`` embed references it at its site uri.
    for uri, abs_src in _FIELD_MOVIE_ASSETS.items():
        existing = files.get_file_from_path(uri)
        if existing is not None:
            files.remove(existing)
        files.append(
            File.generated(config, uri, abs_src_path=abs_src, inclusion=InclusionLevel.INCLUDED)
        )
    # The viewers import the shared three.js loader from ``_static/`` — a tree
    # ``exclude_docs`` drops, so mkdocs never copies it.  Emit it as a generated
    # file when any viewer shipped, so the iframe import resolves instead of
    # 404-ing (which silently degrades every viewer to its static PNG poster).
    if _VIEWERS:
        loader = _viewer.loader_asset()
        if loader is not None:
            loader_uri, loader_src = loader
            existing = files.get_file_from_path(loader_uri)
            if existing is not None:
                files.remove(existing)
            # ``_static/`` is in ``exclude_docs`` (tooling, not a page), so a plain
            # generated file at that URI is dropped.  Force INCLUDED so the loader
            # ships regardless — the viewer iframes import it from ``_static/``.
            files.append(
                File.generated(
                    config, loader_uri, content=loader_src, inclusion=InclusionLevel.INCLUDED
                )
            )
    return files


def on_page_markdown(markdown, page, config, files):
    """Substitute build-time tokens (the library version)."""
    return markdown.replace("{{ tsdynamics_version }}", _VERSION)
