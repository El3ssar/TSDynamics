"""Canonical per-layer style vocabulary + the figure-level theme system.

This module is the **presentation surface** of the viz IR: a single, documented,
validated, introspectable vocabulary for *how* a layer looks (the
:data:`STYLE_KEYS` table + :func:`normalize_style`) and a figure-level
:class:`Theme` (palette / font / background / grid / line defaults) with named
built-ins and an optional **global default**.

It is **pure data + helpers** and imports **no plotting backend** and — crucially
— **does not import** :mod:`tsdynamics.viz.spec` (the dependency runs the other
way: ``spec.py`` imports :class:`Theme`, :func:`normalize_style`,
:func:`get_theme` *from here*).  Keeping the edge one-way avoids an import cycle.

The pieces
----------
- :class:`StyleKey` + :data:`STYLE_KEYS` — the closed, canonical per-layer style
  vocabulary: each key carries its accepted aliases, the backends that honor it,
  an optional validator, and a docstring.  :func:`normalize_style` is the single
  choke point that canonicalizes aliases, validates values, and drops (with one
  warning) unknown keys — killing the "typo silently ignored" failure mode.
- :class:`Theme` — a figure-level look (palette, background/foreground ink, font,
  grid defaults, default line/marker sizes), serializable and mergeable.
- :data:`THEMES` + :func:`register_theme` / :func:`get_theme` / :func:`set_theme`
  / :func:`themes` / :func:`resolve_palette` — the theme registry and the single
  mutable global (the active default theme name), kept isolated and reset-safe.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from .._utils.lookup import is_hashable
from ._visibility import dir_without, listing_dir

#: The figure layout algorithms a renderer may be asked to apply.  Spelled as a
#: closed literal (not a bare ``str``) so a backend can pass it straight through
#: to matplotlib's ``Figure(layout=...)``, whose accepted set is exactly these.
LayoutEngineName = Literal["constrained", "compressed", "tight"]

__all__ = [
    "DEFAULT_PALETTE",
    "STYLE_KEYS",
    "THEMES",
    "StyleKey",
    "Theme",
    "get_theme",
    "normalize_style",
    "register_theme",
    "resolve_palette",
    "set_theme",
    "styles",
    "themes",
]

__dir__ = listing_dir(__all__)


# ---------------------------------------------------------------------------
# Degradation warning (lazy, cycle-free)
# ---------------------------------------------------------------------------


def _degraded_warning_class() -> type[Warning]:
    """Return the canonical ``VisualizationDegraded`` class, or a plain fallback.

    The canonical warning lives in :mod:`tsdynamics.viz.render.caps` (the render
    layer), which imports the spec IR — importing it at *this* module's top level
    would create a cycle.  Resolve it lazily so ``normalize_style`` warns with the
    same class the renderers use, while ``style.py`` stays free of any render/spec
    dependency at import time.
    """
    try:
        from tsdynamics.viz.render.caps import VisualizationDegraded
    except Exception:  # pragma: no cover - render layer unavailable
        return UserWarning
    return VisualizationDegraded


# ---------------------------------------------------------------------------
# Value validators / coercers
# ---------------------------------------------------------------------------


def _validate_unit_interval(value: Any) -> float:
    """Coerce ``value`` to a float in ``[0, 1]`` (raises ``ValueError`` otherwise).

    A ``bool`` (``True``/``False``) is rejected — ``True`` is *not* a valid opacity
    of ``1.0`` (it is almost always a caller mistake), so it raises rather than
    silently coercing.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a value in [0, 1], got bool {value!r}")
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"expected a value in [0, 1], got {value!r}")
    return v


def _validate_positive(value: Any) -> float:
    """Coerce ``value`` to a non-negative float (raises ``ValueError`` otherwise).

    A ``bool`` (``True``/``False``) is rejected — a line/marker size is a magnitude,
    not a flag, so ``True`` raises rather than silently becoming ``1.0``.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a non-negative number, got bool {value!r}")
    v = float(value)
    if v < 0.0:
        raise ValueError(f"expected a non-negative number, got {value!r}")
    return v


def _validate_color(value: Any) -> Any:
    """Check a colour **at the door**, naming the vocabulary a backend would.

    Every other style key validates its value here; ``color`` did not, so
    ``ts.plot(traj, c="time")`` — a plausible guess, since ``color_by="time"``
    is real — was accepted, survived every tweak, and died inside ``.save()``
    with a raw matplotlib ``ValueError``: the one untranslated backend error in
    the whole session.

    Validation is delegated to matplotlib when it is importable (it is the
    reference renderer and owns the widest vocabulary), and skipped entirely
    when it is not — the viz layer must stay usable with no backend installed,
    and refusing a colour some *other* backend understands would be worse than
    letting it through.
    """
    if value is None or not isinstance(value, (str, tuple, list)):
        return value
    try:
        import matplotlib.colors as mcolors
    except ImportError:  # pragma: no cover - no backend installed
        return value
    if mcolors.is_color_like(value):
        return value
    hint = ""
    if isinstance(value, str) and value in ("time", "speed", "index", "arclength", "curvature"):
        hint = f" (did you mean color_by={value!r}? that colours the curve BY a quantity)"
    raise ValueError(
        f"{value!r} is not a colour{hint}. Give a CSS name ('crimson'), a hex string "
        "('#d81b60'), a grey level ('0.4'), or an (r, g, b[, a]) tuple of floats in [0, 1]"
    )


#: Canonical line-style names, plus the matplotlib short spellings we accept.
_LINESTYLE_ALIASES: dict[str, str] = {
    "solid": "solid",
    "dashed": "dashed",
    "dotted": "dotted",
    "dashdot": "dashdot",
    "-": "solid",
    "--": "dashed",
    ":": "dotted",
    "-.": "dashdot",
}


def _validate_linestyle(value: Any) -> str:
    """Canonicalize a line style to one of ``solid/dashed/dotted/dashdot``."""
    key = str(value).strip()
    if key in _LINESTYLE_ALIASES:
        return _LINESTYLE_ALIASES[key]
    raise ValueError(
        f"unknown linestyle {value!r}; expected one of "
        "solid, dashed, dotted, dashdot (or - -- : -.)"
    )


#: Canonical marker-shape names, plus the matplotlib single-char spellings.
_MARKER_ALIASES: dict[str, str] = {
    "circle": "circle",
    "square": "square",
    "triangle": "triangle",
    "diamond": "diamond",
    "cross": "cross",
    "x": "x",
    "star": "star",
    "none": "none",
    "o": "circle",
    "s": "square",
    "^": "triangle",
    "D": "diamond",
    "+": "cross",
    "*": "star",
}


def _validate_marker(value: Any) -> str:
    """Canonicalize a marker shape to one of the named shapes."""
    key = str(value).strip()
    if key in _MARKER_ALIASES:
        return _MARKER_ALIASES[key]
    raise ValueError(
        f"unknown marker {value!r}; expected one of circle, square, triangle, "
        "diamond, cross, x, star, none (or o s ^ D + x *)"
    )


def _validate_bool(value: Any) -> bool:
    """Coerce ``value`` to a plain ``bool``."""
    return bool(value)


def _validate_int(value: Any) -> int:
    """Require an integer (raises ``ValueError`` on a float or ``bool``).

    ``zorder`` is a draw order — a non-integer (``1.7``) or a ``bool``
    (``True``/``False``) is a caller mistake, so this *rejects* rather than
    truncating (the old ``int(1.7) == 1`` silent floor) or accepting a ``bool``
    (``int`` subclass).  Acceptance keys off :class:`numbers.Integral` (not
    ``int``) so a NumPy integer (``np.int64`` — *not* an ``int`` subclass) is a
    valid draw order, e.g. ``.style(zorder=np.arange(n)[i])``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"expected an int, got {value!r}")
    return int(value)


# ---------------------------------------------------------------------------
# StyleKey + STYLE_KEYS
# ---------------------------------------------------------------------------

#: The three visual backends every fully-portable style key is honored by.
_ALL_BACKENDS: frozenset[str] = frozenset({"matplotlib", "plotly", "threejs"})


@dataclass(frozen=True)
class StyleKey:
    """One canonical entry in the per-layer style vocabulary.

    Parameters
    ----------
    name : str
        The canonical key (the spelling stored on a layer's ``style`` dict).
    aliases : tuple of str, optional
        Accepted alternate spellings that :func:`normalize_style` rewrites to
        :attr:`name` (e.g. ``"lw"`` → ``"linewidth"``).
    honored_by : frozenset of str, optional
        The visual backends (``"matplotlib"`` / ``"plotly"`` / ``"threejs"``)
        that actually honor this key.  Drives the honest-degradation warnings: a
        backend not listed here is expected to *warn* rather than silently drop
        the knob.  Defaults to all three.
    validate : callable, optional
        A value coercer / validator ``(value) -> value`` that raises
        :class:`ValueError` on a bad value, or ``None`` to accept the value as-is.
    doc : str, optional
        A one-line human description (surfaced in the public introspection view).
    """

    name: str
    aliases: tuple[str, ...] = ()
    honored_by: frozenset[str] = _ALL_BACKENDS
    validate: Callable[[Any], Any] | None = None
    doc: str = ""

    def __dir__(self) -> list[str]:
        """Expose the vocabulary entry: ``name`` ``aliases`` ``honored_by`` ``doc``.

        Those four *are* the question ``ts.viz.styles`` exists to answer — what
        may I pass, what else is it spelled, will this backend draw it, and what
        does it do.  :attr:`validate` is the coercer
        :func:`normalize_style` runs on the way in; it is the machinery behind
        the vocabulary, not part of it.  Still public, still tested.
        """
        return dir_without(self, {"validate"})


def _build_style_keys() -> dict[str, StyleKey]:
    """Build the canonical :data:`STYLE_KEYS` table (one :class:`StyleKey` per key)."""
    keys = [
        StyleKey(
            name="color",
            aliases=("c",),
            honored_by=_ALL_BACKENDS,
            validate=_validate_color,
            doc="line/marker/fill color (CSS name, hex, or rgb tuple)",
        ),
        StyleKey(
            name="linewidth",
            aliases=("lw",),
            honored_by=_ALL_BACKENDS,
            validate=_validate_positive,
            doc="line width (pt)",
        ),
        StyleKey(
            name="linestyle",
            aliases=("ls",),
            honored_by=frozenset({"matplotlib", "plotly"}),
            validate=_validate_linestyle,
            doc="one of solid, dashed, dotted, dashdot (threejs does not honor it)",
        ),
        StyleKey(
            name="marker",
            honored_by=frozenset({"matplotlib", "plotly"}),
            validate=_validate_marker,
            doc=(
                "marker shape: circle, square, triangle, diamond, cross, x, star, "
                "none (threejs draws points only — shape not honored)"
            ),
        ),
        StyleKey(
            name="markersize",
            # ``"s"`` is deliberately **NOT** an alias (removed in v6).  It collided
            # head-on with two other meanings: matplotlib's ``s`` is a marker *area*
            # (pt²) — which is what every in-tree emitter assumed when it wrote
            # ``{"s": 40.0}`` — while ``markersize`` is a *diameter* (pt); and the
            # single-character marker **value** ``"s"`` means *square*, so
            # ``{"s": 1, "marker": "s"}`` used one letter for both a size key and a
            # shape value.  One canonical spelling, one unit: ``markersize``, pt
            # diameter.
            aliases=("ms",),
            honored_by=_ALL_BACKENDS,
            validate=_validate_positive,
            doc="marker size (pt diameter; 's' is NOT accepted — it meant area)",
        ),
        StyleKey(
            name="alpha",
            aliases=("opacity",),
            honored_by=_ALL_BACKENDS,
            validate=_validate_unit_interval,
            doc="0..1 opacity",
        ),
        StyleKey(
            name="cmap",
            aliases=("colormap", "colorscale"),
            honored_by=frozenset({"matplotlib", "plotly"}),
            doc=(
                "colormap name for the c / image channel (threejs uses a fixed "
                "built-in ramp — an arbitrary cmap name is not honored)"
            ),
        ),
        StyleKey(
            name="filled",
            honored_by=frozenset({"matplotlib", "plotly"}),
            validate=_validate_bool,
            doc=(
                "whether a marker is filled (True) or hollow/open (False) — the "
                "stable-vs-unstable distinction on a fixed-point overlay "
                "(threejs draws points only — fill is not honored)"
            ),
        ),
        StyleKey(
            name="fill",
            honored_by=frozenset({"matplotlib", "plotly"}),
            validate=_validate_bool,
            doc="fill the area under/between (bool)",
        ),
        StyleKey(
            name="fillalpha",
            honored_by=frozenset({"matplotlib", "plotly"}),
            validate=_validate_unit_interval,
            doc="0..1 fill opacity",
        ),
        StyleKey(
            name="zorder",
            honored_by=_ALL_BACKENDS,
            validate=_validate_int,
            doc="int draw order (threejs maps it to renderOrder)",
        ),
    ]
    return {k.name: k for k in keys}


#: The canonical per-layer style vocabulary: canonical name → :class:`StyleKey`.
#: Closed and reviewed — extending it is a deliberate contract change.  Each
#: entry exposes its accepted aliases, e.g.
#: ``STYLE_KEYS["linewidth"].aliases == ("lw",)``.
STYLE_KEYS: dict[str, StyleKey] = _build_style_keys()


def _build_alias_index() -> dict[str, str]:
    """Map every accepted spelling (canonical + alias) to its canonical name."""
    index: dict[str, str] = {}
    for key in STYLE_KEYS.values():
        index[key.name] = key.name
        for alias in key.aliases:
            index[alias] = key.name
    return index


#: alias-or-canonical spelling → canonical key name (built once from STYLE_KEYS).
_ALIAS_INDEX: dict[str, str] = _build_alias_index()


def style_names() -> frozenset[str]:
    """Every accepted spelling of a per-layer style keyword (canonical + alias).

    The single source of truth for "is this keyword about the *look* of the
    plot?", used by every front door that routes keywords by category —
    :func:`tsdynamics.viz.compose.split_presentation`,
    :meth:`tsdynamics.data.Trajectory.plot` and
    :meth:`tsdynamics.families._plottable.SystemPlottable.plot`.  One derived
    set rather than three hand-kept ones, so ``color=`` cannot mean a style at
    one door and an unknown keyword at its sibling.
    """
    return frozenset(_ALIAS_INDEX)


#: Mark-rendering **structural** knobs that ride on a layer's ``style`` dict but
#: are *not* part of the cross-backend aesthetic vocabulary (:data:`STYLE_KEYS`):
#: ``"interpolation"`` (an IMAGE's resampling filter) and ``"bins"`` (a HISTOGRAM's
#: bin count).  The renderers read these directly off ``layer.style``, so
#: :func:`normalize_style` must let them through **verbatim** — they are deliberately
#: kept out of :data:`STYLE_KEYS` so they do not enter the honoring contract /
#: degradation warnings (they are per-mark plumbing, not portable look-and-feel).
_PASSTHROUGH_KEYS: frozenset[str] = frozenset({"interpolation", "bins"})


# ---------------------------------------------------------------------------
# normalize_style
# ---------------------------------------------------------------------------


def normalize_style(style: Mapping[str, Any], *, warn: bool = True) -> dict[str, Any]:
    """Canonicalize aliases, validate values, and drop unknown style keys.

    This is the single choke point every style dict passes through — the
    ``.style()`` tweak when a user sets keys, and each renderer before it
    translates canonical keys to backend kwargs.  Aliases (``"lw"``, ``"c"``,
    ``"s"``, …) are rewritten to their canonical names (``"linewidth"``,
    ``"color"``, ``"markersize"``); values are coerced / validated by the key's
    validator (a bad value raises :class:`ValueError`); and unknown keys are
    dropped.  Keys in :data:`_PASSTHROUGH_KEYS` (the per-mark structural knobs
    ``"interpolation"`` / ``"bins"``) are kept **verbatim** — not validated, not
    treated as unknown, not warned.

    Parameters
    ----------
    style : Mapping
        The raw per-layer style mapping (canonical names and/or aliases).
    warn : bool, optional
        When ``True`` (default) and any key was dropped, emit **one**
        ``VisualizationDegraded`` naming the dropped keys.  Renderers pass
        ``warn=False`` (the dispatcher already emitted the consolidated warning).

    Returns
    -------
    dict
        A new dict keyed by canonical names with validated values (never mutates
        the input).

    Raises
    ------
    ValueError
        If a recognized key's value fails its validator.
    """
    out: dict[str, Any] = {}
    unknown: list[str] = []
    for raw_key, value in style.items():
        if raw_key in _PASSTHROUGH_KEYS:
            out[raw_key] = value
            continue
        canonical = _ALIAS_INDEX.get(raw_key)
        if canonical is None:
            unknown.append(raw_key)
            continue
        spec = STYLE_KEYS[canonical]
        if spec.validate is not None:
            try:
                value = spec.validate(value)
            except ValueError as exc:
                raise ValueError(f"invalid value for style key {canonical!r}: {exc}") from exc
        out[canonical] = value
    if unknown and warn:
        warnings.warn(
            "unknown style key(s) dropped: "
            + ", ".join(repr(k) for k in unknown)
            + "; known keys are "
            + ", ".join(sorted(STYLE_KEYS)),
            _degraded_warning_class(),
            stacklevel=2,
        )
    return out


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

#: The default qualitative color cycle (a clean 10-color matplotlib-ish set).
DEFAULT_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


@dataclass(frozen=True)
class Theme:
    """A figure-level look: palette, background/foreground ink, font, grid, sizes.

    A :class:`Theme` carries the presentation defaults a renderer applies *before*
    per-layer style: the background facecolor, the default ink (text / axes /
    ticks color), the font family + sizes, whether gridlines show, and the default
    line / marker sizes.  Layers that carry no explicit ``color`` are auto-colored
    from :attr:`palette` (the color cycle).

    .. versionchanged:: 6.0
        :attr:`figsize` / :attr:`dpi` / :attr:`layout_engine` **are** theme fields
        now.  They used to live only in ``PlotSpec.meta`` (set by
        :meth:`~tsdynamics.viz.spec.PlotSpec.size`), which meant the
        ``"publication"`` theme produced matplotlib's default 6.4×4.8 in @ 100 dpi
        figure — publication *ink* on a screenshot-resolution canvas.  A theme now
        carries its output geometry too.  ``PlotSpec.meta["figsize"]`` /
        ``meta["dpi"]`` still **win** when set (an explicit ``spec.size(...)`` beats
        the theme); the theme is the default underneath.

    Parameters
    ----------
    name : str, optional
        The theme's name.  Default ``"default"``.
    palette : tuple of str, optional
        The color cycle for auto-colored layers.  Default :data:`DEFAULT_PALETTE`.
    background : str, optional
        Figure / axes facecolor, or ``None`` for the backend default.
    foreground : str, optional
        Default ink — text / axes / ticks color — or ``None``.
    font_family : str, optional
        Default font family, or ``None``.
    font_size : float, optional
        Default (tick/label) font size, or ``None``.
    title_size : float, optional
        Title font size, or ``None``.
    grid : bool, optional
        Default gridline visibility.  Default ``False``.
    grid_color : str, optional
        Default gridline color, or ``None``.
    grid_alpha : float, optional
        Default gridline opacity, or ``None``.
    line_width : float, optional
        Default line width, or ``None``.
    marker_size : float, optional
        Default marker size, or ``None``.
    figsize : tuple of float, optional
        Default figure size ``(width, height)`` in inches, or ``None`` for the
        backend default.  Overridden by ``PlotSpec.meta["figsize"]``.
    dpi : float, optional
        Default output resolution (dots per inch), or ``None``.  Overridden by
        ``PlotSpec.meta["dpi"]``.
    layout_engine : {"constrained", "compressed", "tight"}, optional
        The figure layout algorithm a renderer should apply.  ``None`` (default)
        defers to the renderer's own default — which, since v6, is
        ``"constrained"``, so axis labels are never clipped out of the artifact.
    autostyle : bool, optional
        Whether renderers may derive density-aware line resolution (thinner,
        slightly transparent lines for a very densely sampled curve) when the
        caller set no explicit ``linewidth`` / ``alpha``.  Default ``True``; set
        ``False`` for a constant-width line at every sample count.
    """

    name: str = "default"
    palette: tuple[str, ...] = DEFAULT_PALETTE
    background: str | None = None
    foreground: str | None = None
    font_family: str | None = None
    font_size: float | None = None
    title_size: float | None = None
    grid: bool = False
    grid_color: str | None = None
    grid_alpha: float | None = None
    line_width: float | None = None
    marker_size: float | None = None
    figsize: tuple[float, float] | None = None
    dpi: float | None = None
    layout_engine: LayoutEngineName | None = None
    autostyle: bool = True

    def __dir__(self) -> list[str]:
        """Expose the sixteen fields; hide the three construction/serialization helpers.

        ``styling.md`` reads ``.palette`` ``.font_family`` ``.title_size``
        ``.background`` ``.name`` straight off a theme, so every field stays.

        :meth:`merged` leaves the listing because the **taught** derive route is
        ``ts.viz.themes.register("mine", "dark", palette=(...))`` — one call that
        derives *and* names *and* registers, which is what a user actually wants
        and which ``merged`` alone does not do.  ``to_dict`` / ``from_dict`` are
        the envelope's.  All three stay public and are what ``themes.register``
        calls.
        """
        return dir_without(self, {"from_dict", "merged", "to_dict"})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this theme."""
        return {
            "name": self.name,
            "palette": list(self.palette),
            "background": self.background,
            "foreground": self.foreground,
            "font_family": self.font_family,
            "font_size": None if self.font_size is None else float(self.font_size),
            "title_size": None if self.title_size is None else float(self.title_size),
            "grid": bool(self.grid),
            "grid_color": self.grid_color,
            "grid_alpha": None if self.grid_alpha is None else float(self.grid_alpha),
            "line_width": None if self.line_width is None else float(self.line_width),
            "marker_size": None if self.marker_size is None else float(self.marker_size),
            "figsize": None if self.figsize is None else [float(v) for v in self.figsize],
            "dpi": None if self.dpi is None else float(self.dpi),
            "layout_engine": self.layout_engine,
            "autostyle": bool(self.autostyle),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Theme:
        """Rebuild a :class:`Theme` from :meth:`to_dict` output (tolerates missing keys)."""
        palette = d.get("palette")
        figsize = d.get("figsize")
        return cls(
            name=d.get("name", "default"),
            palette=tuple(palette) if palette is not None else DEFAULT_PALETTE,
            background=d.get("background"),
            foreground=d.get("foreground"),
            font_family=d.get("font_family"),
            font_size=d.get("font_size"),
            title_size=d.get("title_size"),
            grid=bool(d.get("grid", False)),
            grid_color=d.get("grid_color"),
            grid_alpha=d.get("grid_alpha"),
            line_width=d.get("line_width"),
            marker_size=d.get("marker_size"),
            figsize=(float(figsize[0]), float(figsize[1])) if figsize is not None else None,
            dpi=d.get("dpi"),
            layout_engine=cast("LayoutEngineName | None", d.get("layout_engine")),
            autostyle=bool(d.get("autostyle", True)),
        )

    def merged(self, **overrides: Any) -> Theme:
        """Return a copy of this theme with ``overrides`` applied.

        Parameters
        ----------
        **overrides
            Any :class:`Theme` field to override (``palette``, ``background``,
            ``font_family``, …).

        Returns
        -------
        Theme
            A new theme; the original is unchanged.
        """
        return replace(self, **overrides)


# ---------------------------------------------------------------------------
# Built-in themes + registry + global default
# ---------------------------------------------------------------------------

#: A brighter palette for the dark theme (readable on a near-black background).
_DARK_PALETTE: tuple[str, ...] = (
    "#4cc9f0",
    "#f72585",
    "#4ad66d",
    "#ffd166",
    "#b5179e",
    "#80ed99",
    "#ff9e00",
    "#c8b6ff",
    "#06d6a0",
    "#ef476f",
)

#: A muted palette for the minimal theme.
_MINIMAL_PALETTE: tuple[str, ...] = (
    "#4c6173",
    "#a6695a",
    "#5a8a6b",
    "#9c5b6b",
    "#736a8a",
    "#8a7a5a",
    "#6b8a9c",
    "#888888",
    "#9c8a5a",
    "#5a8a8a",
)

#: A colorblind-safe palette for the publication theme (Wong 2011, 8 hues).
_PUBLICATION_PALETTE: tuple[str, ...] = (
    "#000000",
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
)


def _build_builtin_themes() -> dict[str, Theme]:
    """Construct the four built-in themes (default / dark / minimal / publication)."""
    return {
        "default": Theme(
            name="default",
            palette=DEFAULT_PALETTE,
            background=None,
            foreground=None,
            font_family="sans-serif",
            font_size=10.0,
            title_size=12.0,
            grid=False,
            grid_color="#b0b0b0",
            grid_alpha=0.5,
            line_width=1.5,
            marker_size=6.0,
        ),
        "dark": Theme(
            name="dark",
            palette=_DARK_PALETTE,
            background="#11131a",
            foreground="#e6e6e6",
            font_family="sans-serif",
            font_size=10.0,
            title_size=12.0,
            grid=False,
            grid_color="#3a3f4b",
            grid_alpha=0.6,
            line_width=1.5,
            marker_size=6.0,
        ),
        "minimal": Theme(
            name="minimal",
            palette=_MINIMAL_PALETTE,
            background=None,
            foreground=None,
            font_family="sans-serif",
            font_size=10.0,
            title_size=12.0,
            grid=False,
            grid_color="#cccccc",
            grid_alpha=0.4,
            line_width=1.2,
            marker_size=5.0,
        ),
        "publication": Theme(
            name="publication",
            palette=_PUBLICATION_PALETTE,
            background="#ffffff",
            foreground="#000000",
            font_family="serif",
            font_size=12.0,
            title_size=15.0,
            grid=False,
            grid_color="#999999",
            grid_alpha=0.5,
            line_width=1.5,
            marker_size=6.0,
            # The point of the "publication" theme is publication *output*, not
            # only publication ink: a single-column figure at print resolution.
            # Without these it rendered at matplotlib's 6.4x4.8 in @ 100 dpi.
            figsize=(5.0, 3.5),
            dpi=300.0,
            layout_engine="constrained",
        ),
    }


#: name → :class:`Theme`.  Built-ins below; out-of-tree themes :func:`register_theme`.
THEMES: dict[str, Theme] = _build_builtin_themes()

#: The only mutable global state in viz: the active default theme name.  Read /
#: written by :func:`get_theme` / :func:`set_theme`; reset by ``set_theme("default")``.
_ACTIVE: str = "default"


def register_theme(theme: Theme) -> None:
    """Register (or replace) a named :class:`Theme` in :data:`THEMES`.

    Parameters
    ----------
    theme : Theme
        The theme to register; it is filed under ``theme.name``.
    """
    THEMES[theme.name] = theme


def get_theme(name: str | None = None) -> Theme:
    """Return a theme by name, or the active global default when ``name`` is ``None``.

    Parameters
    ----------
    name : str, optional
        A registered theme name.  ``None`` (default) returns the active global
        default theme (``THEMES[_ACTIVE]``).

    Returns
    -------
    Theme

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``name`` is given but is not a registered theme.
    """
    from tsdynamics.errors import InvalidParameterError

    if name is None:
        return THEMES[_ACTIVE]
    # ``name not in THEMES`` hashes ``name`` first, so an unhashable theme= died
    # as a raw TypeError naming a dict the caller never heard of, one line before
    # the typed error below would have named the registered themes.
    if not is_hashable(name) or name not in THEMES:
        raise InvalidParameterError(
            f"unknown theme {name!r}; registered themes are {', '.join(themes())}"
        )
    return THEMES[name]


def set_theme(theme: str | Theme) -> None:
    """Set the global default theme (by registered name or a :class:`Theme` instance).

    Passing a :class:`Theme` registers it (under its name) and makes it active;
    passing a name makes that registered theme active.  ``set_theme("default")``
    returns to the baseline.

    Parameters
    ----------
    theme : str or Theme
        The theme (or its name) to make the active default.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If a *name* is given that is not a registered theme.
    """
    from tsdynamics.errors import InvalidParameterError

    global _ACTIVE
    if isinstance(theme, Theme):
        register_theme(theme)
        _ACTIVE = theme.name
        return
    if theme not in THEMES:
        raise InvalidParameterError(
            f"unknown theme {theme!r}; registered themes are {', '.join(themes())}"
        )
    _ACTIVE = theme


class _ThemeRegistry:
    """``ts.viz.themes`` — the theme registry, in the shared four-verb shape.

    Every registry in the library answers to the same four verbs, so learning one
    teaches the rest::

        ts.viz.themes.names()          # what is there
        ts.viz.themes.get("dark")      # one of them
        ts.viz.themes.find(dark=True)  # the ones matching a filter
        ts.viz.themes.register("lab", palette=(...), dpi=200)
        ts.viz.themes.use("lab")       # ...and make it this session's default

    ``register`` taking **keywords** is what removed the only reason
    :class:`Theme` was ever exported (corollary C1: a name exported because a
    signature demands it is a signature bug).  Measured before v6:
    ``register_theme({...})`` gave ``AttributeError: 'dict' object has no
    attribute 'name'`` and ``set_theme({...})`` a raw ``TypeError: cannot use
    'dict' as a dict key``.

    Calling the object (``ts.viz.themes()``) still returns the sorted name list,
    so the pre-v6 spelling keeps working.
    """

    __slots__ = ()

    def __call__(self) -> list[str]:
        """Return the sorted theme names (the pre-v6 ``themes()`` spelling)."""
        return self.names()

    def names(self) -> list[str]:
        """Return the sorted names of all registered themes (built-in + user)."""
        return sorted(THEMES)

    def get(self, name: str | None = None) -> Theme:
        """Return a theme by name, or the active default when ``name`` is ``None``."""
        return get_theme(name)

    def find(self, what: str = "", /, **filters: Any) -> list[str]:
        """Return the **names** of the themes matching a free-text query and filters.

        The fourth shared registry verb, in the shape all four now have —
        ``find(text, /, **filters) -> list[str]``::

            ts.viz.themes.find("dark")        # free text over the name
            ts.viz.themes.find(grid=True)     # by declared field

        .. versionchanged:: 6.0
            It took no positional argument (``themes.find("dark")`` was a
            ``TypeError``) and returned :class:`Theme` objects while its three
            siblings returned names — two of the four ways the "learn one, know
            all four" promise was false.  ``get(name)`` is how you reach the
            record, which is the same lesson everywhere.
        """
        unknown = [k for k in filters if not hasattr(Theme, k) and k not in Theme.__annotations__]
        if unknown:
            from tsdynamics.errors import InvalidParameterError

            fields = ", ".join(sorted(Theme.__annotations__))
            raise InvalidParameterError(
                f"themes have no field {unknown[0]!r}; the fields are {fields}."
            )
        text = what.lower()
        return [
            name
            for name, t in sorted(THEMES.items())
            if (not text or text in name.lower())
            and all(getattr(t, k, None) == v for k, v in filters.items())
        ]

    def register(self, name: str, theme: Theme | None = None, /, **fields: Any) -> Theme:
        """Build (or derive) a theme, file it under ``name``, and return it.

        Two spellings, one verb::

            ts.viz.themes.register("lab", palette=("#264653", "#e76f51"), dpi=200)
            ts.viz.themes.register("paper", ts.viz.themes.get("publication"),
                                   font_family="Charter", dpi=600)

        Parameters
        ----------
        name : str
            The registry key; also the built theme's :attr:`Theme.name`.
        theme : Theme, optional
            A base to derive from.  ``None`` (default) starts from the built-in
            ``"default"`` theme, so only the fields you name change.
        **fields
            :class:`Theme` fields to override.

        Returns
        -------
        Theme
            The registered theme.
        """
        from tsdynamics.errors import InvalidParameterError

        base = theme if theme is not None else THEMES["default"]
        if not isinstance(base, Theme):
            raise InvalidParameterError(
                f"the base of a theme must be a Theme (from ts.viz.themes.get(...)), not "
                f"{type(base).__name__}; pass the fields as keywords instead: "
                'ts.viz.themes.register("lab", palette=(...), dpi=200).'
            )
        unknown = sorted(set(fields) - set(Theme.__annotations__) - {"name"})
        if unknown:
            known = ", ".join(sorted(set(Theme.__annotations__) - {"name"}))
            raise InvalidParameterError(
                f"themes have no field {unknown[0]!r}; the fields are {known}."
            )
        built = base.merged(name=name, **fields)
        register_theme(built)
        return built

    def use(self, theme: str | Theme) -> Theme:
        """Make ``theme`` this session's default, and return it.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            If a *name* is given that is not registered — naming the ones that are.
        """
        from tsdynamics.errors import InvalidParameterError

        if isinstance(theme, str) and theme not in THEMES:
            raise InvalidParameterError(
                f"unknown theme {theme!r}; registered themes are {', '.join(self.names())}."
            )
        if not isinstance(theme, (str, Theme)):
            raise InvalidParameterError(
                f"use() takes a registered theme name or a Theme, not {type(theme).__name__}; "
                'build one first with ts.viz.themes.register("lab", ...).'
            )
        set_theme(theme)
        return get_theme()

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is a registered theme."""
        return name in THEMES

    def __repr__(self) -> str:
        """List the registered themes and name the active one."""
        active = get_theme().name
        listed = ", ".join(f"*{n}" if n == active else n for n in self.names())
        return f"ts.viz.themes: {listed}  (* = active; .use(name) to switch)"


#: ``ts.viz.themes`` — the theme registry (also callable, returning the names).
themes = _ThemeRegistry()


class _StyleTable:
    """``ts.viz.styles`` — the closed per-layer style vocabulary, and who honors it.

    The answer to "what can I pass to ``color=``?" and "will plotly draw my
    ``linestyle``?", printed::

        >>> import tsdynamics as ts
        >>> ts.viz.styles.names()[:3]
        ['alpha', 'cmap', 'color']
        >>> ts.viz.styles.get("linewidth").aliases
        ('lw',)

    The **same** vocabulary lands at every plotting door — ``ts.plot(...)``,
    ``traj.plot(...)``, ``system.plot(...)`` and ``Plot.style(...)`` — plus the
    aliases (``lw`` / ``c`` / ``ms`` / ``"--"`` / ``"o"``).
    """

    __slots__ = ()

    def __call__(self) -> list[str]:
        """Return the canonical style-key names (aliases excluded)."""
        return self.names()

    def names(self, *, aliases: bool = False) -> list[str]:
        """Return the sorted canonical style keys, optionally including the aliases."""
        return sorted(style_names()) if aliases else sorted(STYLE_KEYS)

    def get(self, name: str) -> StyleKey:
        """Return one :class:`StyleKey`, resolving an alias to its canonical key."""
        canonical = _ALIAS_INDEX.get(name, name)
        if canonical not in STYLE_KEYS:
            from tsdynamics.errors import InvalidParameterError

            raise InvalidParameterError(
                f"unknown style key {name!r}; the vocabulary is "
                f"{', '.join(self.names())} (plus aliases)."
            )
        return STYLE_KEYS[canonical]

    def find(self, what: str = "", /, *, honored_by: str | None = None) -> list[str]:
        """Return the **names** of the style keys matching a query and filters.

        ``ts.viz.styles.find(honored_by="threejs")`` is the honest answer to "what
        will actually change if I switch backend?" — the same declaration the
        dispatcher's :class:`~tsdynamics.viz.render.caps.VisualizationDegraded`
        warning is generated from.  ``ts.viz.styles.find("color")`` is free text
        over the key name and its aliases.

        .. versionchanged:: 6.0
            Took no positional query and returned :class:`StyleKey` objects;
            all four registries answer ``find(text, /, **filters) -> list[str]``
            now, and ``get(name)`` is the one way to reach a record.
        """
        text = what.lower()
        out = []
        for name in sorted(STYLE_KEYS):
            key = STYLE_KEYS[name]
            if honored_by is not None and honored_by not in key.honored_by:
                continue
            if text and text not in name.lower() and not any(text in a for a in key.aliases):
                continue
            out.append(name)
        return out

    def register(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Refuse, by name: the style vocabulary is **closed**, and says why.

        The other four registries are extension points; this one is a *contract*.
        Every :class:`StyleKey` declares ``honored_by`` — which backends genuinely
        render it — and ``tests/test_viz_honoring_contract.py`` draws every claim
        and inspects the artifact.  A key a user adds is honored by no backend, so
        registering one would only buy a keyword that silently does nothing, which
        is the exact defect ``normalize_style``'s drop-with-a-warning exists to
        prevent.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            Always — naming the two things that *are* extensible.
        """
        del args, kwargs
        from tsdynamics.errors import InvalidParameterError, remedy

        raise InvalidParameterError(
            f"the style vocabulary is closed, so {name!r} cannot be registered: a style key "
            "is a contract each backend must honor (StyleKey.honored_by), and one no "
            "renderer draws would be a keyword that silently does nothing. What IS "
            "extensible: a look (a theme) and a way of drawing (a primitive)."
            + remedy(
                "ts.viz.themes.register('mine', palette=('#264653', '#e76f51'))",
                "ts.viz.primitives.register('mine', requires=('x', 'y'))",
            )
        )

    def __iter__(self) -> Any:
        """Iterate the canonical :class:`StyleKey` records, sorted by name."""
        return iter(STYLE_KEYS[n] for n in sorted(STYLE_KEYS))

    def __len__(self) -> int:
        """Return how many canonical style keys there are."""
        return len(STYLE_KEYS)

    def __getitem__(self, name: str) -> StyleKey:
        """``ts.viz.styles["lw"]`` — the same lookup as :meth:`get`."""
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is a style key or one of its aliases."""
        return isinstance(name, str) and (name in STYLE_KEYS or name in _ALIAS_INDEX)

    def __repr__(self) -> str:
        """Render the vocabulary as a table: key, aliases, and who honors it."""
        rows = ["ts.viz.styles — the style vocabulary (same at every plotting door)", ""]
        width = max(len(n) for n in STYLE_KEYS)
        for key in self:
            alias = f"({', '.join(key.aliases)})" if key.aliases else ""
            honored = ", ".join(sorted(key.honored_by)) or "nothing"
            rows.append(f"  {key.name:<{width}} {alias:<12} honored by: {honored}")
        return "\n".join(rows)


#: ``ts.viz.styles`` — the style vocabulary table (also callable, returning names).
styles = _StyleTable()


def resolve_palette(p: str | Sequence[str]) -> tuple[str, ...]:
    """Resolve a palette spec to a tuple of color strings.

    Parameters
    ----------
    p : str or sequence of str
        Either the name of a registered theme (whose :attr:`Theme.palette` is
        returned), a single color string that is *not* a registered theme name
        (e.g. ``"#ff0000"`` / ``"red"`` → a **one-color palette**), or an explicit
        sequence of color strings.

    Returns
    -------
    tuple of str
        The resolved color cycle.  A bare non-theme color string resolves to a
        1-tuple ``(p,)`` rather than raising.
    """
    if isinstance(p, str):
        if p in THEMES:
            return get_theme(p).palette
        return (p,)
    return tuple(p)
