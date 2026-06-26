"""Pure spec builders for trajectory and system draw-views (stream GAPFILL-A).

The *producers* are engine-free, backend-free functions that turn arrays (or a
:class:`~tsdynamics.data.Trajectory`) into a :class:`~tsdynamics.viz.spec.PlotSpec`.
They carry **no rendering** and import **no plotting library** — they are the
flexible, parameterised counterpart of the uniform
:meth:`tsdynamics.data.Trajectory.to_plot_spec` auto-dispatch.  Where
``to_plot_spec(self, kind=None)`` must keep a fixed signature (so it cannot accept
``components=`` / ``color_by=`` / ``tau=`` / a vector-field grid), these producers
take exactly those parameters and return the corresponding semantic spec.

Each producer emits one of the closed semantic
:class:`~tsdynamics.viz.spec.PlotKind` members and uses only the closed
:class:`~tsdynamics.viz.spec.Layer` channel vocabulary, so every produced spec
round-trips losslessly through
:meth:`~tsdynamics.viz.spec.PlotSpec.to_dict` /
:meth:`~tsdynamics.viz.spec.PlotSpec.from_dict`.

The catalogue
-------------
- :func:`time_series` — one ``LINE`` (discrete: ``SCATTER``) per named component,
  overlaid with a legend; optional colour-by-time / colour-by-speed.
- :func:`phase_portrait` — an arbitrary component pair *or* triple (a 3-D
  ``LINE3D`` / ``PHASE_PORTRAIT_3D``), not hardcoded to the first three
  components; optional colour-by-time / colour-by-speed; discrete orbits scatter.
- :func:`delay_embedding` — the ``x(t)`` vs ``x(t - tau)`` reconstruction of a
  scalar series (Packard et al. 1980; Takens 1981).
- :func:`vector_field` / :func:`phase_portrait_field` — a ``QUIVER`` grid of the
  right-hand side over a 2-D slice, optionally over a host trajectory.
- :func:`cobweb` — the 1-D staircase ``x_{n+1}`` vs ``x_n`` with the ``y = x``
  diagonal (a map's orbit geometry).
- :func:`spacetime` — component index vs time as an ``IMAGE`` (a Lorenz-96-style
  field plot).

References
----------
.. [1] Packard, N. H., Crutchfield, J. P., Farmer, J. D. & Shaw, R. S. (1980).
   "Geometry from a Time Series." *Physical Review Letters*, 45(9), 712-716.
.. [2] Takens, F. (1981). "Detecting Strange Attractors in Turbulence." In
   *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics 898,
   366-381.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from tsdynamics.viz.spec import Axis, Colorbar, Layer, Legend, PlotKind, PlotSpec

if TYPE_CHECKING:
    from tsdynamics.data import Trajectory

__all__ = [
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
# Internal coercion helpers (no engine / no Trajectory import at module scope)
# ---------------------------------------------------------------------------


def _split_traj(source: Any) -> tuple[np.ndarray, np.ndarray, tuple[str, ...] | None, bool]:
    """Return ``(t, y, names, is_discrete)`` from a Trajectory-like source.

    ``y`` is coerced to a 2-D ``(T, dim)`` array.  ``names`` are the declared
    component names if the source carries ``variables``, else ``None``.  The
    discreteness flag reads ``source.system.is_discrete`` defensively (``False``
    when absent).  A source is treated as a trajectory when it exposes both
    ``t`` and ``y`` attributes; otherwise a :class:`TypeError` is raised.
    """
    t = getattr(source, "t", None)
    y = getattr(source, "y", None)
    if t is None or y is None:
        raise TypeError(
            "expected a Trajectory (with `t` and `y`); pass raw arrays to the "
            "array-shaped producers instead."
        )
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.atleast_2d(np.asarray(y, dtype=float))
    if y_arr.shape[0] == 1 and t_arr.shape[0] != 1:
        y_arr = y_arr.T
    names = getattr(source, "variables", None)
    names = tuple(names) if names is not None else None
    return t_arr, y_arr, names, _is_discrete(source)


def _is_discrete(source: Any) -> bool:
    """Read ``source.system.is_discrete`` defensively (default ``False``)."""
    system = getattr(source, "system", None)
    flag = getattr(system, "is_discrete", False)
    try:
        return bool(flag)
    except Exception:  # pragma: no cover - defensive
        return False


def _component_index(name_or_index: int | str, names: tuple[str, ...] | None, dim: int) -> int:
    """Resolve a component selector (name or integer) to an integer index."""
    if isinstance(name_or_index, str):
        if names is None:
            raise KeyError(
                f"cannot resolve component {name_or_index!r}: the source declares no "
                f"`variables`; select components by integer index instead."
            )
        try:
            return names.index(name_or_index)
        except ValueError:
            raise KeyError(
                f"unknown component {name_or_index!r}; declared variables: {names}"
            ) from None
    idx = int(name_or_index)
    if not -dim <= idx < dim:
        raise IndexError(f"component index {idx} out of range for dim {dim}")
    return idx % dim


def _label(idx: int, names: tuple[str, ...] | None) -> str:
    """Axis label for component ``idx`` — its declared name, else ``y<idx>``."""
    if names is not None and idx < len(names):
        return names[idx]
    return f"y{idx}"


def _speed(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Per-point speed ``|d(pts)/dt|`` (forward/centred finite difference).

    ``pts`` is ``(T, k)``; the returned magnitude is ``(T,)`` and aligned to the
    sample points (the endpoints reuse the one-sided difference).
    """
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0], dtype=float)
    dpts = np.gradient(pts, t, axis=0)
    return np.asarray(np.linalg.norm(dpts, axis=1), dtype=float)


def _color_channel(
    color_by: str | np.ndarray | None,
    t: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray | None:
    """Resolve a ``color_by`` request into a per-point ``"c"`` channel array.

    ``color_by`` is ``"time"`` (the time vector), ``"speed"`` (the local
    trajectory speed ``|du/dt|``), an explicit per-point array, or ``None``.
    """
    if color_by is None:
        return None
    if isinstance(color_by, str):
        if color_by == "time":
            return np.asarray(t, dtype=float)
        if color_by == "speed":
            return _speed(t, pts)
        raise ValueError(f"unknown color_by={color_by!r}; use 'time', 'speed', or an array.")
    arr = np.asarray(color_by, dtype=float)
    if arr.shape[0] != pts.shape[0]:
        raise ValueError(
            f"color_by array length {arr.shape[0]} does not match the "
            f"{pts.shape[0]} trajectory points."
        )
    return arr


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def time_series(
    source: Trajectory,
    *,
    components: Sequence[int | str] | None = None,
    color_by: str | np.ndarray | None = None,
    legend: bool = True,
) -> PlotSpec:
    """Build an overlaid component-vs-time spec (``TIME_SERIES``).

    One :class:`~tsdynamics.viz.spec.Layer` is drawn per selected component —
    a ``LINE`` for a flow, a ``SCATTER`` for a discrete map (its orbit is a
    point sequence, not a connected curve).  With ``>= 2`` layers a
    :class:`~tsdynamics.viz.spec.Legend` is attached.

    Parameters
    ----------
    source : Trajectory
        The trajectory to draw.
    components : sequence of int or str, optional
        Which components to overlay (names when ``variables`` are declared, or
        integer indices).  ``None`` draws every component.
    color_by : {"time", "speed"} or ndarray, optional
        Colour each line/scatter by the ``"c"`` channel: the time vector, the
        local speed ``|du/dt|``, or an explicit per-point array.  ``None`` leaves
        the layers single-coloured.
    legend : bool, optional
        Whether to attach a legend when more than one component is drawn.
        Default ``True``.

    Returns
    -------
    PlotSpec
        A ``TIME_SERIES`` spec, ``ndim=1``.
    """
    t, y, names, is_discrete = _split_traj(source)
    dim = y.shape[1]
    sel = (
        list(range(dim))
        if components is None
        else [_component_index(c, names, dim) for c in components]
    )
    mark = PlotKind.SCATTER if is_discrete else PlotKind.LINE
    cbar = False
    layers: list[Layer] = []
    for idx in sel:
        col = y[:, idx]
        data: dict[str, np.ndarray] = {"x": t, "y": col}
        c = _color_channel(color_by, t, col[:, None])
        if c is not None:
            data["c"] = c
            cbar = True
        layers.append(Layer(mark, data, label=_label(idx, names)))
    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        ndim=1,
        title=_title(source),
        x=Axis(label="t"),
        y=Axis(label=_label(sel[0], names) if len(sel) == 1 else ""),
        layers=layers,
        legend=Legend() if (legend and len(layers) > 1) else None,
        colorbar=Colorbar(label="time" if color_by == "time" else "speed") if cbar else None,
        meta=_meta(source),
    )
    return spec.autocolor()


# ---------------------------------------------------------------------------
# Phase portrait (arbitrary component pair / triple)
# ---------------------------------------------------------------------------


def phase_portrait(
    source: Trajectory,
    *,
    components: Sequence[int | str] | None = None,
    color_by: str | np.ndarray | None = None,
) -> PlotSpec:
    """Build a phase portrait over an arbitrary component pair or triple.

    A two-component selection yields a ``PHASE_PORTRAIT_2D``; a three-component
    selection a ``PHASE_PORTRAIT_3D`` (a ``LINE3D`` mark).  The components are
    **not** hardcoded to the first axes — pass any pair/triple of names or
    indices.  A discrete-map orbit is drawn as a ``SCATTER`` (point cloud), a
    flow as a connected ``LINE`` / ``LINE3D``.

    Parameters
    ----------
    source : Trajectory
        The trajectory to draw.
    components : sequence of int or str, optional
        Two or three component selectors (names or integer indices) naming the
        display axes.  ``None`` uses the first two (or three) components.
    color_by : {"time", "speed"} or ndarray, optional
        Colour the curve/cloud by the ``"c"`` channel (see :func:`time_series`).

    Returns
    -------
    PlotSpec
        A ``PHASE_PORTRAIT_2D`` (``ndim=2``) or ``PHASE_PORTRAIT_3D``
        (``ndim=3``) spec.

    Raises
    ------
    ValueError
        If fewer than two or more than three components are selected.
    """
    t, y, names, is_discrete = _split_traj(source)
    dim = y.shape[1]
    if components is None:
        sel = list(range(min(3, dim)))
    else:
        sel = [_component_index(c, names, dim) for c in components]
    if not 2 <= len(sel) <= 3:
        raise ValueError(
            f"a phase portrait needs 2 or 3 components, got {len(sel)}; "
            f"use time_series() for a single component."
        )
    want_3d = len(sel) == 3
    pts = y[:, sel]
    data: dict[str, np.ndarray] = {"x": pts[:, 0], "y": pts[:, 1]}
    if want_3d:
        data["z"] = pts[:, 2]
    c = _color_channel(color_by, t, pts)
    cbar = c is not None
    if c is not None:
        data["c"] = c
    flow_mark = PlotKind.LINE3D if want_3d else PlotKind.LINE
    mark = PlotKind.SCATTER if is_discrete else flow_mark
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D if want_3d else PlotKind.PHASE_PORTRAIT_2D,
        ndim=3 if want_3d else 2,
        aspect="equal",
        title=_title(source),
        x=Axis(label=_label(sel[0], names)),
        y=Axis(label=_label(sel[1], names)),
        z=Axis(label=_label(sel[2], names)) if want_3d else None,
        layers=[Layer(mark, data)],
        colorbar=Colorbar(label="time" if color_by == "time" else "speed") if cbar else None,
        meta=_meta(source),
    )
    return spec.autocolor()


# ---------------------------------------------------------------------------
# Delay embedding (DDE / scalar-series reconstruction)
# ---------------------------------------------------------------------------


def delay_embedding(
    series: np.ndarray | Trajectory,
    tau: int,
    *,
    component: int | str = 0,
    label: str = "x",
) -> PlotSpec:
    """Build the ``x(t)`` vs ``x(t - tau)`` delay-coordinate reconstruction.

    The delay embedding of a scalar observable (Packard et al. 1980; Takens
    1981) — the natural 2-D view of a delay-differential trajectory: plot the
    series against a ``tau``-shifted copy of itself.

    Parameters
    ----------
    series : ndarray or Trajectory
        A 1-D scalar series, or a trajectory from which ``component`` is taken.
    tau : int
        The integer delay (in samples) by which the series is shifted; must be
        ``>= 1`` and shorter than the series length.
    component : int or str, optional
        Which trajectory component to embed when ``series`` is a Trajectory.
        Default ``0``.
    label : str, optional
        Base axis label; the axes read ``label(t)`` and ``label(t - tau)``.

    Returns
    -------
    PlotSpec
        A ``PHASE_PORTRAIT_2D`` spec, ``aspect="equal"``.

    Raises
    ------
    ValueError
        If ``tau < 1`` or ``tau`` is not shorter than the series.
    """
    x = _scalar_series(series, component)
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}.")
    if tau >= x.shape[0]:
        raise ValueError(f"tau={tau} must be shorter than the series length {x.shape[0]}.")
    x0 = x[:-tau]
    x1 = x[tau:]
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        ndim=2,
        aspect="equal",
        title=_title(series),
        x=Axis(label=f"{label}(t)"),
        y=Axis(label=f"{label}(t - {tau})"),
        layers=[Layer(PlotKind.LINE, {"x": x0, "y": x1})],
        meta=_meta(series),
    )


# ---------------------------------------------------------------------------
# Vector field / phase-portrait field (QUIVER grid)
# ---------------------------------------------------------------------------


def vector_field(
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid: int = 20,
    normalize: bool = False,
    labels: tuple[str, str] = ("x", "y"),
) -> PlotSpec:
    """Build a ``QUIVER`` grid of a 2-D right-hand side (``VECTOR_FIELD``).

    Samples ``rhs`` on a regular ``grid x grid`` lattice over the box
    ``xlim x ylim`` and emits the arrows as a single ``QUIVER`` layer carrying
    the ``"x"`` / ``"y"`` positions and ``"u"`` / ``"v"`` components.

    Parameters
    ----------
    rhs : callable
        A 2-D field ``rhs([x, y]) -> [u, v]`` (the system's right-hand side, or
        any 2-D vector field).
    xlim, ylim : tuple of float
        The ``(lo, hi)`` extent of the sampling box along each axis.
    grid : int, optional
        Number of samples per axis (a ``grid x grid`` lattice).  Default ``20``.
    normalize : bool, optional
        Whether to unit-normalize each arrow (direction field) rather than draw
        true magnitudes.  Default ``False``.
    labels : tuple of str, optional
        The ``(x, y)`` axis labels.  Default ``("x", "y")``.

    Returns
    -------
    PlotSpec
        A ``VECTOR_FIELD`` spec, ``ndim=2``, ``aspect="equal"``.
    """
    layer = _quiver_layer(rhs, xlim, ylim, grid, normalize)
    return PlotSpec(
        kind=PlotKind.VECTOR_FIELD,
        ndim=2,
        aspect="equal",
        x=Axis(label=labels[0], limits=xlim),
        y=Axis(label=labels[1], limits=ylim),
        layers=[layer],
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

    A ``PHASE_PORTRAIT_FIELD``: the right-hand-side direction field drawn under
    an optional host trajectory (the field first, the orbit on top).  When a
    ``source`` trajectory is given and the limits are omitted, the sampling box
    is taken from the trajectory's in-plane extent (padded).

    Parameters
    ----------
    rhs : callable
        A 2-D field ``rhs([x, y]) -> [u, v]``.
    source : Trajectory, optional
        A trajectory to overlay on the field (its selected two components).
    xlim, ylim : tuple of float, optional
        Sampling-box extent.  When ``None`` and ``source`` is given, taken from
        the trajectory's extent; otherwise default to ``(-1, 1)``.
    grid : int, optional
        Samples per axis.  Default ``20``.
    normalize : bool, optional
        Unit-normalize the arrows (a direction field under the orbit).  Default
        ``True``.
    components : sequence of int or str, optional
        The two trajectory components forming the plane.  Default ``(0, 1)``.

    Returns
    -------
    PlotSpec
        A ``PHASE_PORTRAIT_FIELD`` spec, ``ndim=2``, ``aspect="equal"``.
    """
    labels = ("x", "y")
    overlay: Layer | None = None
    if source is not None:
        _, y, names, _ = _split_traj(source)
        dim = y.shape[1]
        i, j = (_component_index(c, names, dim) for c in components)
        xi, yj = y[:, i], y[:, j]
        labels = (_label(i, names), _label(j, names))
        if xlim is None:
            xlim = _pad_range(xi)
        if ylim is None:
            ylim = _pad_range(yj)
        overlay = Layer(PlotKind.LINE, {"x": xi, "y": yj}, label="trajectory")
    if xlim is None:
        xlim = (-1.0, 1.0)
    if ylim is None:
        ylim = (-1.0, 1.0)
    field = _quiver_layer(rhs, xlim, ylim, grid, normalize)
    layers = [field] if overlay is None else [field, overlay]
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_FIELD,
        ndim=2,
        aspect="equal",
        title=_title(source) if source is not None else "",
        x=Axis(label=labels[0], limits=xlim),
        y=Axis(label=labels[1], limits=ylim),
        layers=layers,
        legend=Legend() if overlay is not None else None,
        meta=_meta(source) if source is not None else {},
    )


def _quiver_layer(
    rhs: Callable[[np.ndarray], np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid: int,
    normalize: bool,
) -> Layer:
    """Sample ``rhs`` on a lattice and return the ``QUIVER`` layer (x/y/u/v)."""
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    gx, gy = np.meshgrid(xs, ys)
    u = np.empty_like(gx)
    v = np.empty_like(gy)
    for r in range(grid):
        for c in range(grid):
            uv = np.asarray(rhs(np.array([gx[r, c], gy[r, c]], dtype=float)), dtype=float)
            u[r, c], v[r, c] = uv[0], uv[1]
    if normalize:
        mag = np.hypot(u, v)
        mag = np.where(mag < np.finfo(float).tiny, 1.0, mag)
        u, v = u / mag, v / mag
    return Layer(
        PlotKind.QUIVER,
        {"x": gx.ravel(), "y": gy.ravel(), "u": u.ravel(), "v": v.ravel()},
    )


# ---------------------------------------------------------------------------
# Cobweb (1-D map staircase)
# ---------------------------------------------------------------------------


def cobweb(
    series: np.ndarray | Trajectory,
    *,
    component: int | str = 0,
    label: str = "x",
) -> PlotSpec:
    """Build the 1-D cobweb staircase (``COBWEB``).

    The staircase geometry of a 1-D map orbit: vertical segments from
    ``(x_n, x_n)`` to ``(x_n, x_{n+1})`` and horizontal segments back to the
    diagonal, drawn as a single ``LINE`` with the ``y = x`` identity line as a
    second layer.

    Parameters
    ----------
    series : ndarray or Trajectory
        A 1-D orbit ``x_0, x_1, ...`` (or a trajectory's ``component``).
    component : int or str, optional
        Component to read when ``series`` is a Trajectory.  Default ``0``.
    label : str, optional
        Axis-label base; the axes read ``x_n`` / ``x_{n+1}`` style.  Default
        ``"x"``.

    Returns
    -------
    PlotSpec
        A ``COBWEB`` spec, ``ndim=2``, ``aspect="equal"``.

    Raises
    ------
    ValueError
        If the orbit has fewer than two points.
    """
    x = _scalar_series(series, component)
    if x.shape[0] < 2:
        raise ValueError("a cobweb needs at least two orbit points.")
    # Staircase vertices: (x0,x0) -> (x0,x1) -> (x1,x1) -> (x1,x2) -> ...
    stair_x = np.empty(2 * (x.shape[0] - 1) + 1, dtype=float)
    stair_y = np.empty_like(stair_x)
    stair_x[0] = x[0]
    stair_y[0] = x[0]
    for n in range(x.shape[0] - 1):
        stair_x[2 * n + 1] = x[n]
        stair_y[2 * n + 1] = x[n + 1]
        stair_x[2 * n + 2] = x[n + 1]
        stair_y[2 * n + 2] = x[n + 1]
    lo = float(min(x.min(), stair_y.min()))
    hi = float(max(x.max(), stair_y.max()))
    diag = np.array([lo, hi], dtype=float)
    return PlotSpec(
        kind=PlotKind.COBWEB,
        ndim=2,
        aspect="equal",
        title=_title(series),
        x=Axis(label=f"{label}_n"),
        y=Axis(label=f"{label}_(n+1)"),
        layers=[
            Layer(PlotKind.LINE, {"x": diag, "y": diag}, label="y = x"),
            Layer(PlotKind.LINE, {"x": stair_x, "y": stair_y}, label="orbit"),
        ],
        legend=Legend(),
        meta=_meta(series),
    )


# ---------------------------------------------------------------------------
# Spacetime (component-index vs time IMAGE)
# ---------------------------------------------------------------------------


def spacetime(source: Trajectory, *, transpose: bool = False) -> PlotSpec:
    """Build a component-index vs time ``IMAGE`` (``SPACETIME``).

    The spatiotemporal field view of a high-dimensional flow (e.g. a Lorenz-96
    lattice): the state ``y`` (shape ``(T, dim)``) is drawn as a single
    color-mapped ``IMAGE``, with time along ``x`` and component index along
    ``y`` (``transpose`` swaps the axes).

    Parameters
    ----------
    source : Trajectory
        The trajectory to image.
    transpose : bool, optional
        Draw time along ``y`` and component index along ``x`` instead.  Default
        ``False``.

    Returns
    -------
    PlotSpec
        A ``SPACETIME`` spec, ``ndim=2``, with a :class:`Colorbar` and inferred
        ``clim``.
    """
    t, y, names, _ = _split_traj(source)
    dim = y.shape[1]
    comp_idx = np.arange(dim, dtype=float)
    if transpose:
        img = y
        x_axis = Axis(label="component")
        y_axis = Axis(label="t")
        x_data, y_data = comp_idx, t
    else:
        img = y.T
        x_axis = Axis(label="t")
        y_axis = Axis(label="component")
        x_data, y_data = t, comp_idx
    layer = Layer(PlotKind.IMAGE, {"x": x_data, "y": y_data, "c": img.ravel(), "z": img})
    spec = PlotSpec(
        kind=PlotKind.SPACETIME,
        ndim=2,
        title=_title(source),
        x=x_axis,
        y=y_axis,
        layers=[layer],
        colorbar=Colorbar(label="state"),
        meta={**_meta(source), "component_names": list(names) if names is not None else None},
    )
    return spec.autocolor()


# ---------------------------------------------------------------------------
# Spatial field (a spatially-extended system's field, played over time)
# ---------------------------------------------------------------------------


def spatial_field(
    source: Trajectory,
    *,
    field_shape: tuple[int, ...] | None = None,
    component: int | str | None = None,
) -> PlotSpec:
    """Build a :data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` spec from a field trajectory.

    A *spatial field* is the state of a spatially-extended system (a method-of-
    lines PDE) reshaped to its spatial grid: each per-time state vector becomes a
    spatial snapshot, and the per-frame plot's shape follows the field's spatial
    dimensionality —

    - a **1-D field** ``u(x)`` (``field_shape=(N,)`` or none) is a **line** (the
      profile); animating it is a travelling-wave movie;
    - a **2-D field** ``u(x, y)`` (``field_shape=(Ny, Nx)``) is an ``IMAGE``
      heatmap; animating it is an evolving-field movie.

    One semantic kind covers both (the renderer dispatches on the field's spatial
    ndim, like :meth:`~tsdynamics.data.Trajectory.to_plot_spec` already
    auto-dispatches on component count).

    The full per-time field stack rides on the layer's ``"frames"`` channel (shape
    ``(T, *spatial)``) so a ``frames``-mode animation plays the genuinely-evolving
    field; the final-time snapshot is the layer's static data (``z`` for 2-D, ``y``
    for 1-D), so a still render / a backend that cannot animate draws the field at
    the final time.

    Parameters
    ----------
    source : Trajectory
        A field trajectory.  ``field_shape`` (the spatial grid) is taken from the
        argument, else ``source.meta["field_shape"]`` (recorded by a system that
        declares ``_field_shape``); without either, the state vector is treated as
        a 1-D profile (honest — never guessing a 2-D grid).
    field_shape : tuple of int, optional
        The spatial grid ``(Ny, Nx)`` (2-D) or ``(N,)`` (1-D) one field block
        occupies.  Overrides the trajectory's recorded shape.
    component : int or str, optional
        Which field **block** to plot when the state packs several
        (``source.meta["field_labels"]``, e.g. Gray–Scott's ``("u", "v")``).  A
        name resolves against the labels, an integer indexes them.  ``None``
        selects the **last** block (the activator convention: Gray–Scott's ``v``),
        or the whole state for a single-block field.

    Returns
    -------
    PlotSpec
        A ``SPATIAL_FIELD`` spec — ``ndim=2`` with an ``IMAGE`` layer for a 2-D
        field, ``ndim=1`` with a ``LINE`` layer for a 1-D profile.
    """
    _, y, _, _ = _split_traj(source)
    shape, labels = _resolve_field_shape(source, field_shape)
    block = _select_field_block(y, shape, labels, component)
    # ``block`` is (T, prod(shape)); the per-time field stack reshaped to the grid.
    frames = block.reshape(block.shape[0], *shape)
    title = _title(source)
    meta = {**_meta(source), "field_shape": tuple(int(n) for n in shape)}

    if len(shape) >= 2:
        return _spatial_field_2d(frames, title, meta)
    return _spatial_field_1d(frames, title, meta)


def _spatial_field_2d(frames: np.ndarray, title: str, meta: dict[str, Any]) -> PlotSpec:
    """Build the 2-D heatmap ``SPATIAL_FIELD`` spec (an ``IMAGE`` + the frame stack)."""
    final = frames[-1]
    ny, nx = final.shape
    layer = Layer(
        PlotKind.IMAGE,
        {
            "x": np.arange(nx, dtype=float),
            "y": np.arange(ny, dtype=float),
            "z": final,
            "frames": frames,
        },
    )
    spec = PlotSpec(
        kind=PlotKind.SPATIAL_FIELD,
        ndim=2,
        aspect="equal",
        title=title,
        x=Axis(label="x"),
        y=Axis(label="y"),
        layers=[layer],
        colorbar=Colorbar(label="u"),
        meta=meta,
    )
    # Fix the colour range across all frames (the full field stack) so an animated
    # movie never re-scales between frames; a static render uses the same range.
    finite = frames[np.isfinite(frames)]
    if finite.size:
        spec.clim = (float(finite.min()), float(finite.max()))
    return spec.autocolor()


def _spatial_field_1d(frames: np.ndarray, title: str, meta: dict[str, Any]) -> PlotSpec:
    """Build the 1-D profile ``SPATIAL_FIELD`` spec (a ``LINE`` + the frame stack)."""
    final = frames[-1]
    x = np.arange(final.shape[0], dtype=float)
    layer = Layer(PlotKind.LINE, {"x": x, "y": final, "frames": frames}, label="u(x)")
    return PlotSpec(
        kind=PlotKind.SPATIAL_FIELD,
        ndim=1,
        title=title,
        x=Axis(label="x"),
        y=Axis(label="u"),
        layers=[layer],
        meta=meta,
    )


def _resolve_field_shape(
    source: Any, field_shape: tuple[int, ...] | None
) -> tuple[tuple[int, ...], tuple[str, ...] | None]:
    """Resolve ``(spatial_shape, field_labels)`` from the argument or the trajectory meta.

    Priority: an explicit ``field_shape`` argument, else ``meta["field_shape"]``.
    Returns ``None`` shape only as the 1-D fallback handled by the caller.
    """
    meta = getattr(source, "meta", None)
    labels = None
    if isinstance(meta, dict):
        ml = meta.get("field_labels")
        labels = tuple(str(s) for s in ml) if ml else None
    if field_shape is not None:
        return tuple(int(n) for n in field_shape), labels
    if isinstance(meta, dict):
        ms = meta.get("field_shape")
        if ms is not None:
            return tuple(int(n) for n in ms), labels
    # No field metadata: treat the whole state vector as a 1-D profile (honest;
    # never guess a 2-D grid).
    dim = int(np.atleast_2d(np.asarray(getattr(source, "y", np.empty((0, 0))))).shape[1])
    return (dim,), labels


def _select_field_block(
    y: np.ndarray,
    shape: tuple[int, ...],
    labels: tuple[str, ...] | None,
    component: int | str | None,
) -> np.ndarray:
    """Slice the chosen field block (shape ``(T, prod(shape))``) out of the state.

    A single-block field returns the leading ``prod(shape)`` columns.  When the
    state packs several blocks (``labels``), ``component`` picks one — a name
    resolved against ``labels``, an integer index — defaulting to the **last**
    block (the activator convention, e.g. Gray–Scott's ``v``).
    """
    cells = int(np.prod(shape))
    n_blocks = max(1, y.shape[1] // cells) if cells else 1
    if labels is not None:
        n_blocks = len(labels)
    if component is None:
        block = n_blocks - 1  # the activator / last block by convention
    elif isinstance(component, str):
        if labels is None:
            raise KeyError(
                f"cannot resolve field block {component!r}: the trajectory declares no "
                f"`field_labels`; select a block by integer index instead."
            )
        try:
            block = labels.index(component)
        except ValueError:
            raise KeyError(
                f"unknown field block {component!r}; declared field_labels: {labels}"
            ) from None
    else:
        block = int(component) % n_blocks
    lo = block * cells
    return y[:, lo : lo + cells]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _scalar_series(series: np.ndarray | Trajectory, component: int | str) -> np.ndarray:
    """Coerce a scalar series from a 1-D array or a trajectory ``component``."""
    if getattr(series, "y", None) is not None and getattr(series, "t", None) is not None:
        _, y, names, _ = _split_traj(series)
        return y[:, _component_index(component, names, y.shape[1])]
    arr = np.asarray(series, dtype=float)
    if arr.ndim == 2:
        idx = component if isinstance(component, int) else 0
        return arr[:, idx]
    return arr.ravel()


def _title(source: Any) -> str:
    """Compose a title from a source's ``meta["system"]`` if present."""
    meta = getattr(source, "meta", None)
    if isinstance(meta, dict):
        system = meta.get("system")
        if system:
            return str(system)
    return ""


def _meta(source: Any) -> dict[str, Any]:
    """Return a shallow copy of a source's ``meta`` mapping (or ``{}``)."""
    meta = getattr(source, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def _pad_range(arr: np.ndarray, *, frac: float = 0.05) -> tuple[float, float]:
    """Return a slightly padded ``(min, max)`` range of a 1-D array."""
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return (lo - 1.0, hi + 1.0)
    pad = frac * (hi - lo)
    return (lo - pad, hi + pad)
