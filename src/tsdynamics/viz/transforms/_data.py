"""The in-tree transforms migrated from ``viz.producers`` (stream P1).

Eight registered transforms, one per pre-registry producer, plus the coercion
helpers they share.  The producers stay importable as two-line shims over these,
because they are cited in the docs and are de-facto public.

This module is the **worked example** every other transform copies: a
``compute`` function that returns :class:`~tsdynamics.viz.transforms.Geometry`,
one :func:`~tsdynamics.viz.transforms.plot_transform` decorator declaring the
compatibility row, and nothing else — no renderer edit, no
:class:`~tsdynamics.viz.spec.PlotKind` edit, no test edit.

Two facts the migration made visible and worth keeping in mind when adding one:

- **The default primitive can depend on the data.**  A discrete-map orbit is a
  point sequence and a flow is a connected curve; the geometry says so with
  ``primitive=``, and the transform's declared default covers the rest.
- **So can the presentation.**  A 2-D spatial field is drawn on equal axes and
  its 1-D profile is not, so the geometry carries ``aspect=``.

References
----------
.. [1] Packard, N. H., Crutchfield, J. P., Farmer, J. D. & Shaw, R. S. (1980).
   "Geometry from a Time Series." *Physical Review Letters*, 45(9), 712-716.
.. [2] Takens, F. (1981). "Detecting Strange Attractors in Turbulence." In
   *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics 898,
   366-381.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .._frames import FrameSpace, OverlayRole
from ..spec import PlotKind
from ._base import Geometry, Part, Presentation, make_frame
from ._registry import plot_transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.data import Trajectory

__all__ = [
    "cobweb",
    "delay_embedding",
    "phase_portrait",
    "phase_portrait_field",
    "spacetime",
    "spatial_field",
    "time_series",
]


# ---------------------------------------------------------------------------
# Coercion helpers (no engine import, no Trajectory import at module scope)
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


def _acceleration(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Per-point acceleration magnitude ``|d^2(pts)/dt^2|``."""
    if pts.shape[0] < 3:
        return np.zeros(pts.shape[0], dtype=float)
    acc = np.gradient(np.gradient(pts, t, axis=0), t, axis=0)
    return np.asarray(np.linalg.norm(acc, axis=1), dtype=float)


def _arclength(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length along the drawn curve (starts at 0)."""
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0], dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([np.zeros(1), np.cumsum(seg)]).astype(float)


def _as_curve(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Return the curve the *bend* fields use (``sagitta`` / ``curvature``).

    A 2-/3-component selection is already a space curve, so use it directly. A
    single component (a time series) is the graph ``(t, value)`` — using that 2-D
    curve keeps the bend measures well defined instead of degenerate.
    """
    if pts.shape[1] >= 2:
        return np.asarray(pts, dtype=float)
    return np.column_stack([np.asarray(t, dtype=float), pts[:, 0].astype(float)])


def _sagitta(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Per-point **relative** sagitta (bow / chord) of the drawn curve.

    Reuses the library's sagitta geometry —
    :func:`tsdynamics.analysis.sampling.sagitta_profile` — on the drawn points (a
    single component uses the ``(t, value)`` graph). *Relative* (bow / chord, with
    per-feature sigma-normalization) so it tracks how sharply the trajectory
    **bends** rather than how fast it moves. Endpoints are 0.
    """
    from tsdynamics.analysis.sampling import sagitta_profile

    return sagitta_profile(_as_curve(t, pts), relative=True)


def _curvature(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Per-point Frenet curvature (dimension-general).

    ``k = sqrt(|r'|^2 |r''|^2 - (r'.r'')^2) / |r'|^3``, on the drawn space curve (a
    single component uses the ``(t, value)`` graph, like :func:`_sagitta`).
    """
    curve = _as_curve(t, pts)
    if curve.shape[0] < 3:
        return np.zeros(curve.shape[0], dtype=float)
    r1 = np.gradient(curve, t, axis=0)
    r2 = np.gradient(r1, t, axis=0)
    s1 = np.einsum("ij,ij->i", r1, r1)
    num = np.sqrt(
        np.clip(s1 * np.einsum("ij,ij->i", r2, r2) - np.einsum("ij,ij->i", r1, r2) ** 2, 0.0, None)
    )
    den = np.power(s1, 1.5)
    return np.asarray(np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12), dtype=float)


#: Named per-point colour fields: ``name -> f(t, drawn_points) -> (T,) values`` for
#: a layer's ``"c"`` channel. Add a new ``color_by`` name by extending this table.
_COLOR_FIELDS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "time": lambda t, pts: np.asarray(t, dtype=float),
    "index": lambda t, pts: np.arange(pts.shape[0], dtype=float),
    "speed": _speed,
    "acceleration": _acceleration,
    "accel": _acceleration,
    "arclength": _arclength,
    "sagitta": _sagitta,
    "curvature": _curvature,
}


def _color_label(color_by: object) -> str:
    """Return a short colorbar label for a ``color_by`` request (its name, or ``"value"``)."""
    if isinstance(color_by, str):
        return color_by
    name = getattr(color_by, "__name__", None) if callable(color_by) else None
    return name if name and name != "<lambda>" else "value"


def _color_channel(
    color_by: str | np.ndarray | Callable[..., np.ndarray] | None,
    source: Trajectory,
    t: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray | None:
    """Resolve a ``color_by`` request into a per-point ``"c"`` channel array.

    ``color_by`` may be a **name** from :data:`_COLOR_FIELDS` (``"time"``,
    ``"speed"``, ``"sagitta"``, ``"curvature"``, ``"acceleration"``/``"accel"``,
    ``"arclength"``, ``"index"`` — computed from the drawn points), a **callable**
    ``f(trajectory) -> array`` (given the whole :class:`~tsdynamics.data.Trajectory`
    so it can use any named component), an explicit **per-point array**, or
    ``None``. The result is a 1-D array of length ``len(pts)``; a mismatch raises.
    """
    if color_by is None:
        return None
    if isinstance(color_by, str):
        field = _COLOR_FIELDS.get(color_by)
        if field is None:
            raise ValueError(
                f"unknown color_by={color_by!r}; use one of {sorted(_COLOR_FIELDS)}, "
                f"a per-point array, or a callable f(trajectory) -> array."
            )
        c = np.asarray(field(np.asarray(t, dtype=float), pts), dtype=float)
    elif callable(color_by):
        c = np.asarray(color_by(source), dtype=float)
    else:
        c = np.asarray(color_by, dtype=float)
    c = c.ravel()
    if c.shape[0] != pts.shape[0]:
        raise ValueError(
            f"color_by produced {c.shape[0]} values but the trajectory has {pts.shape[0]} points."
        )
    return c


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


# ---------------------------------------------------------------------------
# Example subjects for the compatibility gate
#
# These live here, next to the transforms, so that registering a transform is
# genuinely one call and nothing else: `tests/test_viz_compatibility.py` reads
# the example off the record and needs no edit when a row is added.
# ---------------------------------------------------------------------------


def _demo_flow(dim: int = 3, n: int = 200) -> Trajectory:
    """Return a small, deterministic 3-component flow trajectory."""
    from tsdynamics.data import Trajectory as _Trajectory

    t = np.linspace(0.0, 10.0, n)
    y = np.column_stack([np.sin(t + k) * (k + 1) for k in range(dim)])
    names = ("x", "y", "z", "w")[:dim]
    return _Trajectory(t, y, _DemoSystem(False, names), {"system": "demo", "dt": float(t[1])})


def _demo_map(n: int = 80) -> Trajectory:
    """Return a small, deterministic 2-component map orbit."""
    from tsdynamics.data import Trajectory as _Trajectory

    t = np.arange(n, dtype=float)
    y = np.column_stack([np.cos(0.3 * t), np.sin(0.4 * t)])
    return _Trajectory(t, y, _DemoSystem(True, ("a", "b")), {"system": "demo map"})


def _demo_field(shape: tuple[int, ...] = (8, 10), n: int = 6) -> Trajectory:
    """Return a small, deterministic 2-D spatial-field trajectory."""
    from tsdynamics.data import Trajectory as _Trajectory

    t = np.linspace(0.0, 1.0, n)
    cells = int(np.prod(shape))
    y = np.stack([np.sin(np.arange(cells) * 0.3 + ti) for ti in t])
    return _Trajectory(t, y, _DemoSystem(), {"system": "demo field", "field_shape": shape})


class _DemoSystem:
    """The minimal system stand-in the demo trajectories carry."""

    def __init__(self, discrete: bool = False, variables: tuple[str, ...] | None = None) -> None:
        self.is_discrete = discrete
        self.variables = variables


def _demo_rhs(u: np.ndarray) -> np.ndarray:
    """Evaluate a planar spiral right-hand side (the vector-field example)."""
    x, y = float(u[0]), float(u[1])
    return np.array([-y + 0.1 * x, x + 0.1 * y])


# ---------------------------------------------------------------------------
# time_series
# ---------------------------------------------------------------------------


@plot_transform(
    name="time_series",
    source="data",
    kind=PlotKind.TIME_SERIES,
    frame=FrameSpace.TIME,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "steps"),
    presentation=Presentation(autocolor=True),
    example=lambda primitive: (_demo_flow(), {}),
    doc="One curve per selected component against time.",
)
def time_series(
    source: Trajectory,
    *,
    components: Sequence[int | str] | None = None,
    color_by: str | np.ndarray | Callable[..., np.ndarray] | None = None,
    legend: bool = True,
) -> Geometry:
    """Component value versus time — one :class:`Part` per selected component.

    Parameters
    ----------
    source : Trajectory
        The trajectory to draw.
    components : sequence of int or str, optional
        Which components to overlay (names when ``variables`` are declared, or
        integer indices).  ``None`` draws every component.
    color_by : str, ndarray, or callable, optional
        Colour each curve by the ``"c"`` channel.  A name — ``"time"``,
        ``"speed"``, ``"sagitta"``, ``"curvature"``, ``"acceleration"`` (alias
        ``"accel"``), ``"arclength"`` or ``"index"`` — is computed from the drawn
        points; a callable ``f(trajectory) -> array`` is given the whole
        trajectory; or pass an explicit per-point array.
    legend : bool, optional
        Whether to attach a legend when more than one component is drawn.

    Returns
    -------
    Geometry
        A ``time``-framed geometry, ``ndim=1`` — so ``x(t)`` and ``y(t)``
        legitimately overlay while an ``(x, y)`` portrait and an ``(x, z)`` one
        do not.
    """
    t, y, names, is_discrete = _split_traj(source)
    dim = y.shape[1]
    sel = (
        list(range(dim))
        if components is None
        else [_component_index(c, names, dim) for c in components]
    )
    parts: list[Part] = []
    coloured = False
    for idx in sel:
        col = y[:, idx]
        channels: dict[str, Any] = {"x": t, "y": col}
        c = _color_channel(color_by, source, t, col[:, None])
        if c is not None:
            channels["c"] = c
            coloured = True
        parts.append(Part(channels, label=_label(idx, names)))
    y_label = _label(sel[0], names) if len(sel) == 1 else ""
    return Geometry(
        "time_series",
        make_frame(FrameSpace.TIME, ("t",)),
        parts,
        axis_labels=("t", y_label),
        primitive="points" if is_discrete else "line",
        title=_title(source),
        color_label=_color_label(color_by) if coloured else None,
        legend=bool(legend) and len(parts) > 1,
        meta=_meta(source),
    )


# ---------------------------------------------------------------------------
# phase_portrait
# ---------------------------------------------------------------------------


@plot_transform(
    name="phase_portrait",
    source="data",
    frame=(FrameSpace.STATE2, FrameSpace.STATE3),
    ndim=(2, 3),
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "line3d", "points", "points3d", "density"),
    presentation=Presentation(aspect="equal", autocolor=True),
    example=lambda primitive: (
        _demo_flow(),
        {"components": [0, 1]} if primitive in ("line", "points", "density") else {},
    ),
    doc="An orbit in state space over an arbitrary component pair or triple.",
)
def phase_portrait(
    source: Trajectory,
    *,
    components: Sequence[int | str] | None = None,
    color_by: str | np.ndarray | Callable[..., np.ndarray] | None = None,
) -> Geometry:
    """Compute the orbit itself, over any two or three state components.

    The components are **not** hardcoded to the leading axes — pass any pair or
    triple of names or indices.  A discrete-map orbit defaults to a point cloud
    (it *is* a point sequence, not a connected curve) and a flow to a line; both
    remain drawable as either.

    Parameters
    ----------
    source : Trajectory
        The trajectory to draw.
    components : sequence of int or str, optional
        Two or three component selectors naming the display axes.  ``None`` uses
        the first two (or three) components.
    color_by : str, ndarray, or callable, optional
        Colour the curve / cloud by the ``"c"`` channel (see :func:`time_series`).

    Returns
    -------
    Geometry
        ``state2`` or ``state3``, with the selected component names as its axes —
        which is what makes an ``(x, y)`` portrait refuse an ``(x, z)`` overlay.

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
    channels: dict[str, Any] = {"x": pts[:, 0], "y": pts[:, 1]}
    if want_3d:
        channels["z"] = pts[:, 2]
    c = _color_channel(color_by, source, t, pts)
    if c is not None:
        channels["c"] = c
    labels = tuple(_label(i, names) for i in sel)
    if is_discrete:
        primitive = "points3d" if want_3d else "points"
    else:
        primitive = "line3d" if want_3d else "line"
    return Geometry(
        "phase_portrait",
        make_frame(FrameSpace.STATE3 if want_3d else FrameSpace.STATE2, labels),
        channels=channels,
        axis_labels=labels,
        kind=PlotKind.PHASE_PORTRAIT_3D if want_3d else PlotKind.PHASE_PORTRAIT_2D,
        primitive=primitive,
        # A 3-component orbit is a space curve: the 2-D primitives would silently
        # drop z (and `density` would bin a projection while the spec still called
        # itself a 3-D portrait), so they are narrowed away for this geometry.
        primitives=("line3d", "points3d") if want_3d else ("line", "points", "density"),
        title=_title(source),
        color_label=_color_label(color_by) if c is not None else None,
        meta=_meta(source),
    )


# ---------------------------------------------------------------------------
# delay_embedding
# ---------------------------------------------------------------------------


@plot_transform(
    name="delay_embedding",
    source="data",
    kind=PlotKind.PHASE_PORTRAIT_2D,
    frame=FrameSpace.STATE2,
    ndim=2,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "density"),
    presentation=Presentation(aspect="equal"),
    example=lambda primitive: (_demo_flow(), {"delay": 7}),
    doc="The x(t) vs x(t - delay) reconstruction of a scalar observable.",
)
def delay_embedding(
    series: np.ndarray | Trajectory,
    delay: int | None = None,
    *,
    delay_time: float | None = None,
    component: int | str = 0,
    label: str = "x",
    tau: Any = None,
) -> Geometry:
    """Reconstruct one scalar observable in delay coordinates.

    Packard et al. (1980); Takens (1981) — the natural 2-D view of a
    delay-differential trajectory, and the bridge from a measured series back to
    phase space.

    **One concept, two units, two names.**  ``delay`` is a lag in *samples* (the
    library-wide meaning of an embedding delay — it is exactly what
    :func:`~tsdynamics.analysis.optimal_delay` returns, so the two compose);
    ``delay_time`` is the same lag expressed in the trajectory's *time units*, and
    is converted through its ``dt``.  Give exactly one.  The old ``tau`` spelling
    meant samples on one front door and time units on the other, so it now raises
    and names both.

    Parameters
    ----------
    series : ndarray or Trajectory
        A 1-D scalar series, or a trajectory from which ``component`` is taken.
    delay : int, optional
        The lag **in samples**; ``>= 1`` and shorter than the series.
    delay_time : float, optional
        The same lag **in time units**, converted via the trajectory's ``dt``.
        Needs a :class:`~tsdynamics.data.Trajectory` (a bare array has no time
        axis).
    component : int or str, optional
        Which trajectory component to embed when ``series`` is a Trajectory.
    label : str, optional
        Base axis label; the axes read ``label(t)`` and ``label(t - delay)``.
    tau : any, optional
        Rejected on sight — see above.

    Returns
    -------
    Geometry

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If neither or both of ``delay`` / ``delay_time`` are given, if ``tau`` is
        given, if ``delay_time`` is asked of a bare array, or if the resolved lag
        is not ``1 <= delay < len(series)``.
    """
    x = _scalar_series(series, component)
    lag = _resolve_delay(series, x.shape[0], delay, delay_time, tau)
    labels = (f"{label}(t)", f"{label}(t - {lag})")
    return Geometry(
        "delay_embedding",
        make_frame(FrameSpace.STATE2, labels),
        channels={"x": x[:-lag], "y": x[lag:]},
        axis_labels=labels,
        title=_title(series),
        meta=_meta(series),
    )


#: The one runnable pair every delay error quotes, so the reader never has to
#: work out which spelling carries which unit.
_DELAY_HINT = (
    "    ts.plot(traj, 'delay_embedding', delay=7)         # 7 SAMPLES\n"
    "    traj.plot(kind='delay', delay_time=0.12)          # 0.12 TIME UNITS\n"
    "(both spellings work on both front doors, and mean the same thing on each)"
)


def _resolve_delay(
    series: Any, length: int, delay: int | None, delay_time: float | None, tau: Any
) -> int:
    """Resolve ``delay`` / ``delay_time`` to a validated sample lag.

    The single place the two spellings are turned into samples, shared by the
    transform and by :meth:`tsdynamics.data.Trajectory.to_plot_spec`, so the two
    front doors cannot disagree about what a delay means.
    """
    from tsdynamics.data import Trajectory
    from tsdynamics.errors import InvalidParameterError

    if tau is not None:
        raise InvalidParameterError(
            "tau= is not a delay spelling in this library: it used to mean SAMPLES on "
            "ts.plot(...) and TIME UNITS on Trajectory.to_plot_spec(...). Say which you "
            f"mean:\n{_DELAY_HINT}"
        )
    if (delay is None) == (delay_time is None):
        raise InvalidParameterError(
            "a delay embedding needs exactly one of delay= (samples) or delay_time= "
            f"(time units):\n{_DELAY_HINT}"
        )
    if delay_time is not None:
        if not isinstance(series, Trajectory):
            raise InvalidParameterError(
                "delay_time= is in time units, and a bare array carries no time axis. "
                f"Pass the lag in samples instead:\n"
                f"    ts.plot(data, 'delay_embedding', delay={max(1, int(delay_time))})"
            )
        return series._delay_samples(delay_time)
    assert delay is not None  # narrowed by the exactly-one check above
    if int(delay) != delay:
        raise InvalidParameterError(
            f"delay= is a lag in SAMPLES and must be a whole number, got {delay!r}. For a "
            f"delay in time units use delay_time=:\n{_DELAY_HINT}"
        )
    lag = int(delay)
    if lag < 1:
        raise InvalidParameterError(
            f"delay= is a lag in SAMPLES and must be >= 1, got {delay!r}. For a delay in "
            f"time units use delay_time=:\n{_DELAY_HINT}"
        )
    if lag >= length:
        raise InvalidParameterError(
            f"delay={lag} samples must be shorter than the series length {length}."
        )
    return lag


# ---------------------------------------------------------------------------
# vector_field / phase_portrait_field
# ---------------------------------------------------------------------------


def _quiver_channels(
    rhs: Callable[[np.ndarray], np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid: int,
    normalize: bool,
) -> dict[str, np.ndarray]:
    """Sample ``rhs`` on a lattice and return the ``x`` / ``y`` / ``u`` / ``v`` channels."""
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
    return {"x": gx.ravel(), "y": gy.ravel(), "u": u.ravel(), "v": v.ravel()}


def _demo_planar_system() -> Any:
    """Return the small, fast planar **system** the field examples are drawn on."""
    from tsdynamics.systems.continuous.population_dynamics import LotkaVolterra

    return LotkaVolterra()


def _field_and_host(
    subject: Any, source: Any, components: Sequence[int | str]
) -> tuple[Callable[[np.ndarray], np.ndarray], Any]:
    """Resolve ``(rhs, host orbit)`` from a **system** or a bare 2-D callable.

    The repair that lets a ``source="model"`` transform be handed a model: given
    a system, the in-plane right-hand side is closed over from its numeric RHS
    (the other coordinates frozen at the system's own default start), and — when
    the caller named no host trajectory — a short run supplies the orbit that
    makes this kind a *portrait* rather than a bare field.
    """
    from tsdynamics.families import SystemBase

    if not isinstance(subject, SystemBase):
        return subject, source
    system = subject
    names = tuple(system.variables)
    dim = len(names)
    i, j = (_component_index(c, names, dim) for c in components)
    at = np.asarray(system._resolve_ic(None), dtype=float)
    evaluate = system._rhs_numeric()

    def rhs(point: np.ndarray) -> np.ndarray:
        state = at.copy()
        state[i], state[j] = float(point[0]), float(point[1])
        out = np.asarray(evaluate(state, 0.0), dtype=float)
        return np.array([out[i], out[j]], dtype=float)

    if source is None:
        source = system.run(final_time=20.0, dt=0.01)
    return rhs, source


@plot_transform(
    name="phase_portrait_field",
    source="model",
    kind=PlotKind.PHASE_PORTRAIT_FIELD,
    frame=FrameSpace.STATE2,
    ndim=2,
    role=OverlayRole.FIELD,
    default_primitive="quiver",
    primitives=("quiver",),
    presentation=Presentation(aspect="equal"),
    example=lambda primitive: (_demo_planar_system(), {"grid": 6}),
    doc="A direction field with a host orbit drawn on it.",
)
def phase_portrait_field(
    subject: Any,
    source: Trajectory | None = None,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grid: int = 20,
    normalize: bool = True,
    components: Sequence[int | str] = (0, 1),
) -> Geometry:
    """Sample a direction field, with an optional host trajectory over it.

    A **heterogeneous** geometry: the field part is drawn by whichever primitive
    the caller chose, while the orbit part pins ``line`` — no single primitive
    draws both, and pretending otherwise would silently turn the orbit into
    arrows.

    Parameters
    ----------
    subject : system or callable
        The **system** whose field to draw (the spelling that works from
        ``ts.plot``), or a bare 2-D field ``rhs([x, y]) -> [u, v]``.  It used to
        be the callable only, so ``ts.plot(vdp, "phase_portrait_field")``
        answered ``TypeError: 'VanDerPol' object is not callable`` — a transform
        declaring ``source="model"`` that could not be handed a model.
    source : Trajectory, optional
        A trajectory to overlay on the field (its selected two components).
        With a *system* subject and no ``source``, the system's own short run
        supplies the host orbit.
    xlim, ylim : tuple of float, optional
        Sampling-box extent.  ``None`` with a ``source`` takes the trajectory's
        padded in-plane extent; otherwise ``(-1, 1)``.
    grid : int, optional
        Samples per axis.
    normalize : bool, optional
        Unit-normalize the arrows (a direction field under the orbit).
    components : sequence of int or str, optional
        The two trajectory components forming the plane.

    Returns
    -------
    Geometry
    """
    rhs, source = _field_and_host(subject, source, components)
    labels: tuple[str, str] = ("x", "y")
    orbit: Part | None = None
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
        orbit = Part({"x": xi, "y": yj}, label="trajectory", primitive="line")
    if xlim is None:
        xlim = (-1.0, 1.0)
    if ylim is None:
        ylim = (-1.0, 1.0)
    field = Part(_quiver_channels(rhs, xlim, ylim, grid, normalize))
    return Geometry(
        "phase_portrait_field",
        make_frame(FrameSpace.STATE2, labels),
        [field] if orbit is None else [field, orbit],
        axis_labels=labels,
        axis_limits=(xlim, ylim),
        title=_title(source) if source is not None else "",
        legend=orbit is not None,
        meta=_meta(source) if source is not None else {},
    )


# ---------------------------------------------------------------------------
# cobweb
# ---------------------------------------------------------------------------


@plot_transform(
    name="cobweb",
    source="data",
    kind=PlotKind.COBWEB,
    frame=FrameSpace.STATE2,
    ndim=2,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points"),
    presentation=Presentation(aspect="equal"),
    example=lambda primitive: (_demo_map(), {"component": 0}),
    doc="The 1-D staircase x_{n+1} vs x_n with the y = x diagonal.",
)
def cobweb(
    series: np.ndarray | Trajectory,
    *,
    component: int | str = 0,
    label: str = "x",
) -> Geometry:
    """Compute the staircase geometry of a 1-D map orbit.

    Two parts: the ``y = x`` diagonal and the staircase itself — vertical
    segments from ``(x_n, x_n)`` to ``(x_n, x_{n+1})`` and horizontal segments
    back to the diagonal.

    Parameters
    ----------
    series : ndarray or Trajectory
        A 1-D orbit ``x_0, x_1, ...`` (or a trajectory's ``component``).
    component : int or str, optional
        Component to read when ``series`` is a Trajectory.
    label : str, optional
        Axis-label base; the axes read ``x_n`` / ``x_(n+1)``.

    Returns
    -------
    Geometry

    Raises
    ------
    ValueError
        If the orbit has fewer than two points.

    Notes
    -----
    This is the *data* form of a cobweb, so it draws the orbit but not the graph
    of ``f`` — that needs the map itself and is a model transform (phase P3).
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
    labels = (f"{label}_n", f"{label}_(n+1)")
    return Geometry(
        "cobweb",
        make_frame(FrameSpace.STATE2, labels),
        [
            Part({"x": diag, "y": diag}, label="y = x"),
            Part({"x": stair_x, "y": stair_y}, label="orbit"),
        ],
        axis_labels=labels,
        title=_title(series),
        meta=_meta(series),
    )


# ---------------------------------------------------------------------------
# spacetime
# ---------------------------------------------------------------------------


@plot_transform(
    name="spacetime",
    source="data",
    kind=PlotKind.SPACETIME,
    frame=FrameSpace.GRID2,
    ndim=2,
    role=OverlayRole.FIELD,
    default_primitive="image",
    primitives=("image", "surface3d", "contour"),
    presentation=Presentation(autocolor=True),
    example=lambda primitive: (_demo_flow(dim=4), {}),
    doc="Component index versus time as a colour-mapped lattice.",
)
def spacetime(source: Trajectory, *, transpose: bool = False) -> Geometry:
    """Image a high-dimensional flow as component index versus time (a Lorenz-96 lattice).

    Parameters
    ----------
    source : Trajectory
        The trajectory to image.
    transpose : bool, optional
        Draw time along ``y`` and component index along ``x`` instead.

    Returns
    -------
    Geometry
        A ``grid2`` lattice carrying both the 2-D ``z`` field and its flattened
        ``c`` channel, so ``image``, ``surface3d`` and ``contour`` all draw it.
    """
    t, y, names, _ = _split_traj(source)
    dim = y.shape[1]
    comp_idx = np.arange(dim, dtype=float)
    if transpose:
        img = y
        labels = ("component", "t")
        x_data, y_data = comp_idx, t
    else:
        img = y.T
        labels = ("t", "component")
        x_data, y_data = t, comp_idx
    return Geometry(
        "spacetime",
        make_frame(FrameSpace.GRID2, labels),
        channels={"x": x_data, "y": y_data, "c": img.ravel(), "z": img},
        axis_labels=labels,
        title=_title(source),
        color_label="state",
        meta={**_meta(source), "component_names": list(names) if names is not None else None},
    )


# ---------------------------------------------------------------------------
# spatial_field
# ---------------------------------------------------------------------------


@plot_transform(
    name="spatial_field",
    source="data",
    kind=PlotKind.SPATIAL_FIELD,
    frame=FrameSpace.GRID2,
    ndim=(1, 2),
    role=OverlayRole.FIELD,
    default_primitive="image",
    primitives=("image", "surface3d", "contour", "line", "points"),
    # A 1-D field is a profile and a 2-D field a heatmap, so the row splits by
    # dimensionality: the gate is handed whichever field the primitive can draw.
    example=lambda primitive: (
        (_demo_field(shape=(16,)), {}) if primitive in ("line", "points") else (_demo_field(), {})
    ),
    doc="A spatially-extended system's field, with the per-time stack for a movie.",
)
def spatial_field(
    source: Trajectory,
    *,
    field_shape: tuple[int, ...] | None = None,
    component: int | str | None = None,
) -> Geometry:
    """Reshape a method-of-lines PDE state onto its spatial grid.

    A **1-D** field ``u(x)`` is a profile (a line); a **2-D** field ``u(x, y)`` is
    a heatmap.  One transform covers both — the geometry reports the field's
    spatial dimensionality and the primitive follows, exactly the way the
    trajectory front door dispatches on component count.

    The full per-time field stack rides on the ``"frames"`` channel (shape
    ``(T, *spatial)``) so a ``frames``-mode animation plays a genuinely evolving
    field; the static channels hold the **final** field, so a still render draws
    something real.

    Parameters
    ----------
    source : Trajectory
        A field trajectory.  ``field_shape`` comes from the argument, else
        ``source.meta["field_shape"]``; with neither, the state vector is treated
        as a 1-D profile — honest, never guessing a 2-D grid.
    field_shape : tuple of int, optional
        The spatial grid one field block occupies.
    component : int or str, optional
        Which field **block** to plot when the state packs several
        (``meta["field_labels"]``, e.g. Gray-Scott's ``("u", "v")``).  ``None``
        selects the **last** block (the activator convention).

    Returns
    -------
    Geometry
    """
    _, y, _, _ = _split_traj(source)
    shape, labels = _resolve_field_shape(source, field_shape)
    block = _select_field_block(y, shape, labels, component)
    frames = block.reshape(block.shape[0], *shape)
    meta = {**_meta(source), "field_shape": tuple(int(n) for n in shape)}
    title = _title(source)

    if len(shape) >= 2:
        final = frames[-1]
        ny, nx = final.shape
        finite = frames[np.isfinite(frames)]
        return Geometry(
            "spatial_field",
            make_frame(FrameSpace.GRID2, ("x", "y")),
            channels={
                "x": np.arange(nx, dtype=float),
                "y": np.arange(ny, dtype=float),
                "z": final,
                "frames": frames,
            },
            axis_labels=("x", "y"),
            # A 2-D field is a lattice; `line` / `points` would pair the x and y
            # *axis* coordinates against each other, which draws nothing real.
            primitives=("image", "surface3d", "contour"),
            aspect="equal",
            title=title,
            color_label="u",
            # Fix the colour range across *all* frames so an animated movie never
            # re-scales between frames; a static render uses the same range.
            clim=(float(finite.min()), float(finite.max())) if finite.size else None,
            meta=meta,
        )
    final = frames[-1]
    return Geometry(
        "spatial_field",
        make_frame(FrameSpace.GRID2, ("x",), ndim=1),
        channels={"x": np.arange(final.shape[0], dtype=float), "y": final, "frames": frames},
        label="u(x)",
        axis_labels=("x", "u"),
        primitive="line",
        primitives=("line", "points"),
        title=title,
        meta=meta,
    )


def _resolve_field_shape(
    source: Any, field_shape: tuple[int, ...] | None
) -> tuple[tuple[int, ...], tuple[str, ...] | None]:
    """Resolve ``(spatial_shape, field_labels)`` from the argument or the trajectory meta."""
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
    """Slice the chosen field block (shape ``(T, prod(shape))``) out of the state."""
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


#: Read by ``tsdynamics.viz.producers`` — the migrated names, so the shim module
#: and this one cannot drift apart.  ``vector_field`` left this table in v6: the
#: name now belongs to the **system**-taking transform in ``planar.py`` (the
#: guessable spelling had been the broken one), and the bare-callable producer it
#: used to name survives only as ``phase_portrait_field``'s field sampler.
MIGRATED: Mapping[str, Callable[..., Geometry]] = {
    "cobweb": cobweb,
    "delay_embedding": delay_embedding,
    "phase_portrait": phase_portrait,
    "phase_portrait_field": phase_portrait_field,
    "spacetime": spacetime,
    "spatial_field": spatial_field,
    "time_series": time_series,
}
