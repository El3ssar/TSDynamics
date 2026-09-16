"""Back-compat shims over the registered plot transforms.

Every builder here is now a **two-line shim** over
:mod:`tsdynamics.viz.transforms`: the geometry, the compatibility row and the
spec assembly live on the registered transform of the same name, and these
functions exist because they are cited in the documentation and are therefore
de-facto public.

.. versionchanged:: 6.0
   These were the pre-registry *producers*.  Migrating them was deliberately
   picture-preserving — ``tests/test_viz_golden.py`` pins each one's rendered
   figure and full spec fingerprint — so that any later difference in a plot is
   attributable to the change that made it, not to the migration.  The one
   intended difference is that every layer now carries ``Layer.transform``
   provenance, which no renderer reads.

New code should prefer the transform surface, which is strictly larger::

    ts.plot(traj, "phase_portrait")                    # the same picture
    ts.plot(traj, "phase_portrait", primitive="density")   # and a swap
    ts.viz.compatibility("phase_portrait")             # what else it can be

What stays here rather than moving: the **density-aware line resolution** law
(:func:`autostyle_line` / :func:`autostyle_enabled`), which is a *renderer*
concern — the matplotlib and plotly backends import it at module scope — and has
nothing to do with which transform produced a curve.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from tsdynamics.viz.transforms import build_spec

if TYPE_CHECKING:
    from tsdynamics.data import Trajectory
    from tsdynamics.viz.spec import PlotSpec

__all__ = [
    "autostyle_line",
    "cobweb",
    "delay_embedding",
    "phase_portrait",
    "phase_portrait_field",
    "spacetime",
    "spatial_field",
    "time_series",
    "vector_field",
]

# ---------------------------------------------------------------------------
# Density-aware line resolution (H1)
# ---------------------------------------------------------------------------

#: Sample count at or below which :func:`autostyle_line` is the identity — the
#: theme's own ``line_width`` and full opacity.  Chosen so every plot the library
#: drew before density-aware resolution landed (docs figures, result plots, short
#: orbits) is **bit-identical**; only genuinely dense curves are thinned.
AUTOSTYLE_PIVOT: int = 2000

#: Exponent of the ``(pivot / n)`` width law.  Calibrated (see
#: ``docs/visualization/`` and the contact sheet in the H1 deliverable) so a
#: 100k-sample Lorenz attractor lands on the hand-tuned ``linewidth=0.25`` /
#: ``alpha=0.85`` reference that recovers its laminar sheet structure.
AUTOSTYLE_EXPONENT: float = 0.46

#: Floor on the derived linewidth (pt).  Below ~0.2 pt matplotlib's Agg
#: rasteriser stops resolving the stroke at typical dpi and the curve fades out.
AUTOSTYLE_MIN_LINEWIDTH: float = 0.2

#: Floor on the derived alpha.  Below ~0.55 a single-pass curve reads as grey
#: rather than as the theme colour.
AUTOSTYLE_MIN_ALPHA: float = 0.60

#: Alpha decays by this much per decade of sample count above the pivot.
AUTOSTYLE_ALPHA_PER_DECADE: float = 0.09


def autostyle_line(
    n: int,
    *,
    line_width: float | None,
    enabled: bool = True,
) -> tuple[float | None, float | None]:
    r"""Resolve ``(linewidth, alpha)`` for a curve of ``n`` samples.

    The library's canonical object is a long chaotic trajectory, and a constant
    stroke width is the wrong default for it: at the theme's 1.5 pt a
    100,000-sample Lorenz attractor renders as a solid blob in which every trace
    of the laminar sheet structure is destroyed, while the *same data* at
    ``linewidth=0.25, alpha=0.85`` is a publication-quality figure.  This
    function makes that adjustment automatic and, crucially, **inspectable**
    rather than magic.

    The law is

    .. math::

        w(n) = \\mathrm{clip}\\!\\left(w_0 \\left(\\frac{n_0}{n}\\right)^{p},\\;
                                       w_\\min,\\; w_0\\right), \\qquad
        a(n) = \\mathrm{clip}\\!\\left(1 - d\\,\\log_{10}\\frac{n}{n_0},\\;
                                       a_\\min,\\; 1\\right)

    with pivot :math:`n_0` = :data:`AUTOSTYLE_PIVOT`, exponent :math:`p` =
    :data:`AUTOSTYLE_EXPONENT`, and :math:`w_0` the theme's ``line_width``.

    Two properties are contractual and pinned by tests:

    1. **Non-regression** — for ``n <= AUTOSTYLE_PIVOT`` the result is exactly
       ``(line_width, None)``, so no existing figure moves.
    2. **Monotonicity** — the width is non-increasing in ``n`` and never leaves
       ``[AUTOSTYLE_MIN_LINEWIDTH, line_width]``.

    It is applied **only** where the caller supplied no explicit value: an
    explicit ``.style(linewidth=…)`` / ``alpha=…`` always wins (the renderers
    consult the layer's normalized style first).

    Parameters
    ----------
    n : int
        Number of samples in the curve.
    line_width : float or None
        The theme's default stroke width (:attr:`tsdynamics.viz.style.Theme.line_width`).
        ``None`` disables the width law (there is no baseline to scale) but the
        alpha law still applies.
    enabled : bool, optional
        When ``False`` the function is the identity — the escape hatch behind
        ``spec.meta["autostyle"] = False``.  Default ``True``.

    Returns
    -------
    (float or None, float or None)
        The derived ``(linewidth, alpha)``.  A ``None`` component means "leave
        the backend/theme default alone".
    """
    if not enabled or n <= AUTOSTYLE_PIVOT:
        return line_width, None
    ratio = AUTOSTYLE_PIVOT / float(n)
    width: float | None = None
    if line_width is not None:
        w0 = float(line_width)
        width = float(np.clip(w0 * ratio**AUTOSTYLE_EXPONENT, AUTOSTYLE_MIN_LINEWIDTH, w0))
    decades = float(np.log10(float(n) / AUTOSTYLE_PIVOT))
    alpha = float(np.clip(1.0 - AUTOSTYLE_ALPHA_PER_DECADE * decades, AUTOSTYLE_MIN_ALPHA, 1.0))
    return width, alpha


def autostyle_enabled(spec: Any) -> bool:
    """Whether density-aware line resolution applies to ``spec``.

    Two escape hatches, most specific first:

    1. ``spec.meta["autostyle"]`` — a per-figure override;
    2. :attr:`Theme.autostyle <tsdynamics.viz.style.Theme.autostyle>` on the
       spec's resolved theme — the *documented* way to ask for a constant-width
       line at every sample count for a whole session
       (``ts.viz.set_theme(...)``) or one figure (``spec.theme(...)``).

    Reading the theme is load-bearing, not cosmetic: ``Theme.autostyle`` is a
    public, documented, round-tripped field, and a documented knob that is
    accepted and then ignored is exactly the silent no-op this layer must not
    have.  Default ``True``.
    """
    meta = getattr(spec, "meta", None)
    if isinstance(meta, dict) and "autostyle" in meta:
        return bool(meta["autostyle"])
    theme = getattr(spec, "resolved_theme", None)
    if theme is not None:
        return bool(getattr(theme, "autostyle", True))
    return True


# ---------------------------------------------------------------------------
# The shims — one call each into the registered transform of the same name
# ---------------------------------------------------------------------------


def time_series(
    source: Trajectory,
    *,
    components: Sequence[int | str] | None = None,
    color_by: str | np.ndarray | Callable[..., np.ndarray] | None = None,
    legend: bool = True,
) -> PlotSpec:
    """Build an overlaid component-vs-time spec (``TIME_SERIES``).

    Shim over the ``time_series`` transform; see
    :func:`tsdynamics.viz.transforms._data.time_series` for the parameters.
    """
    return build_spec(
        source, "time_series", components=components, color_by=color_by, legend=legend
    )


def phase_portrait(
    source: Trajectory,
    *,
    components: Sequence[int | str] | None = None,
    color_by: str | np.ndarray | Callable[..., np.ndarray] | None = None,
) -> PlotSpec:
    """Build a phase portrait over an arbitrary component pair or triple.

    Shim over the ``phase_portrait`` transform; see
    :func:`tsdynamics.viz.transforms._data.phase_portrait` for the parameters.
    """
    return build_spec(source, "phase_portrait", components=components, color_by=color_by)


def delay_embedding(
    series: np.ndarray | Trajectory,
    delay: int | None = None,
    *,
    delay_time: float | None = None,
    components: int | str = 0,
    label: str = "x",
) -> PlotSpec:
    """Build the ``x(t)`` vs ``x(t - delay)`` delay-coordinate reconstruction.

    Shim over the ``delay_embedding`` transform — ``delay`` in **samples**,
    ``delay_time`` in **time units** (exactly one); see
    :func:`tsdynamics.viz.transforms._data.delay_embedding`.
    """
    return build_spec(
        series,
        "delay_embedding",
        delay=delay,
        delay_time=delay_time,
        components=components,
        label=label,
    )


def vector_field(
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid: int = 20,
    normalize: bool = True,
    labels: tuple[str, str] = ("x", "y"),
) -> PlotSpec:
    """Build a ``QUIVER`` grid of a 2-D right-hand side (``VECTOR_FIELD``).

    Shim over the ``vector_field`` transform; see
    :func:`tsdynamics.viz.transforms._data.vector_field`.  ``normalize``
    defaults to the transform's own default (unit arrows), because a shim that
    pins a *different* default is exactly the drift the migration removed — and
    ``tests/test_viz_golden.py`` measures it.
    """
    return build_spec(
        rhs,
        "vector_field",
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        normalize=normalize,
        labels=labels,
    )


def phase_portrait_field(
    rhs: Callable[[np.ndarray], np.ndarray],
    source: Trajectory | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int = 20,
    normalize: bool = True,
    components: Sequence[int | str] = (0, 1),
) -> PlotSpec:
    """Build a ``QUIVER`` field optionally overlaid with a trajectory.

    Shim over the ``phase_portrait_field`` transform; see
    :func:`tsdynamics.viz.transforms._data.phase_portrait_field`.
    """
    return build_spec(
        rhs,
        "phase_portrait_field",
        source=source,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        normalize=normalize,
        components=components,
    )


def cobweb(
    series: np.ndarray | Trajectory,
    *,
    components: int | str = 0,
    label: str = "x",
) -> PlotSpec:
    """Build the 1-D cobweb staircase (``COBWEB``).

    Shim over the ``cobweb`` transform; see
    :func:`tsdynamics.viz.transforms._data.cobweb`.
    """
    return build_spec(series, "cobweb", components=components, label=label)


def spacetime(source: Trajectory, *, transpose: bool = False) -> PlotSpec:
    """Build a component-index vs time ``IMAGE`` (``SPACETIME``).

    Shim over the ``spacetime`` transform; see
    :func:`tsdynamics.viz.transforms._data.spacetime`.
    """
    return build_spec(source, "spacetime", transpose=transpose)


def spatial_field(
    source: Trajectory,
    *,
    field_shape: tuple[int, ...] | None = None,
    components: int | str | None = None,
) -> PlotSpec:
    """Build a ``SPATIAL_FIELD`` spec from a field trajectory.

    Shim over the ``spatial_field`` transform; see
    :func:`tsdynamics.viz.transforms._data.spatial_field`.
    """
    return build_spec(source, "spatial_field", field_shape=field_shape, components=components)
