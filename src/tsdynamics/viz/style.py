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

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

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
    "themes",
]


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
    """Coerce ``value`` to a float in ``[0, 1]`` (raises ``ValueError`` otherwise)."""
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"expected a value in [0, 1], got {value!r}")
    return v


def _validate_positive(value: Any) -> float:
    """Coerce ``value`` to a non-negative float (raises ``ValueError`` otherwise)."""
    v = float(value)
    if v < 0.0:
        raise ValueError(f"expected a non-negative number, got {value!r}")
    return v


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
    """Coerce ``value`` to a plain ``int``."""
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


def _build_style_keys() -> dict[str, StyleKey]:
    """Build the canonical :data:`STYLE_KEYS` table (one :class:`StyleKey` per key)."""
    keys = [
        StyleKey(
            name="color",
            aliases=("c",),
            honored_by=_ALL_BACKENDS,
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
            aliases=("ms", "s"),
            honored_by=_ALL_BACKENDS,
            validate=_validate_positive,
            doc="marker size (pt)",
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
            honored_by=_ALL_BACKENDS,
            doc="colormap name for the c / image channel",
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
#: Closed and reviewed — extending it is a deliberate contract change.
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
    dropped.

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


@dataclass
class Theme:
    """A figure-level look: palette, background/foreground ink, font, grid, sizes.

    A :class:`Theme` carries the presentation defaults a renderer applies *before*
    per-layer style: the background facecolor, the default ink (text / axes /
    ticks color), the font family + sizes, whether gridlines show, and the default
    line / marker sizes.  Layers that carry no explicit ``color`` are auto-colored
    from :attr:`palette` (the color cycle).  ``figsize`` / ``dpi`` are **not** on a
    theme — they live in ``PlotSpec.meta`` (set by ``PlotSpec.size``).

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
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Theme:
        """Rebuild a :class:`Theme` from :meth:`to_dict` output (tolerates missing keys)."""
        palette = d.get("palette")
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
    KeyError
        If ``name`` is given but is not a registered theme.
    """
    if name is None:
        return THEMES[_ACTIVE]
    if name not in THEMES:
        raise KeyError(f"unknown theme {name!r}; registered themes are {', '.join(themes())}")
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
    KeyError
        If a *name* is given that is not a registered theme.
    """
    global _ACTIVE
    if isinstance(theme, Theme):
        register_theme(theme)
        _ACTIVE = theme.name
        return
    if theme not in THEMES:
        raise KeyError(f"unknown theme {theme!r}; registered themes are {', '.join(themes())}")
    _ACTIVE = theme


def themes() -> list[str]:
    """Return the sorted names of all registered themes (built-in + user)."""
    return sorted(THEMES)


def resolve_palette(p: str | Sequence[str]) -> tuple[str, ...]:
    """Resolve a palette spec to a tuple of color strings.

    Parameters
    ----------
    p : str or sequence of str
        Either the name of a registered theme (whose :attr:`Theme.palette` is
        returned) or an explicit sequence of color strings.

    Returns
    -------
    tuple of str
        The resolved color cycle.

    Raises
    ------
    KeyError
        If ``p`` is a string that is not a registered theme name.
    """
    if isinstance(p, str):
        return get_theme(p).palette
    return tuple(p)
