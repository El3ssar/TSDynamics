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
    """Whether this trajectory came from a **map** — read from ``system.family``.

    It used to read ``system.is_discrete``, a name v6 removed.  ``getattr(…,
    False)`` has a *legal* default, so the rename orphaned this reader in
    silence: every map orbit was classified as a flow and drawn as a connected
    LINE instead of the point sequence it is (a map's iterates are not joined —
    that is the whole visual difference between the two families).  The rule
    CLAUDE.md states for this class of defect is *do not read a removed name as
    a string*; ``family`` is the live spelling.
    """
    system = getattr(source, "system", None)
    return getattr(system, "family", None) == "map" if system is not None else False


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
    ndim: int | None = None,
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
    ndim : {2, 3}, optional
        Force the projection's dimensionality without naming which components —
        ``ts.plot(traj, "phase_portrait", ndim=2)`` is the leading *pair* of a
        3-D orbit.  This is the positional spelling of what ``kind=
        "phase_portrait_2d"`` / ``"phase_portrait_3d"`` used to be the only way
        to ask for; naming the components explicitly is still clearer when you
        know them.  Ignored when ``components`` is given (they already say).

        .. versionadded:: 6.0
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
    tsdynamics.errors.InvalidParameterError
        If ``ndim`` is neither 2 nor 3, or the orbit has too few components for it.
    """
    from tsdynamics.errors import InvalidParameterError

    t, y, names, is_discrete = _split_traj(source)
    dim = y.shape[1]
    if ndim is not None and components is None:
        if ndim not in (2, 3):
            raise InvalidParameterError(
                f"a phase portrait is 2-D or 3-D, not {ndim}-D; "
                "for one component use ts.plot(traj, 'time_series')."
            )
        if dim < ndim:
            raise InvalidParameterError(
                f"ndim={ndim} needs {ndim} state components and this orbit has {dim}."
            )
    if components is None:
        width = ndim if ndim is not None else min(3, dim)
        sel = list(range(min(width, dim)))
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
    aliases=("delay",),  # the kind= recipe spelling
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
    components: int | str = 0,
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
    components : int or str, optional
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
    # Validate the lag BEFORE touching the subject: it needs no data, and a
    # missing delay= used to surface as "float() argument must be a string or a
    # real number, not 'VanDerPol'" from inside the series coercion — an internal
    # leak about the wrong thing, for the one keyword the caller actually forgot.
    _reject_missing_delay(delay, delay_time, tau)
    x = _scalar_series(series, components)
    lag = _resolve_delay(series, x.shape[0], delay, delay_time, tau)
    labels = _delay_labels(series, components, label, lag)
    return Geometry(
        "delay_embedding",
        make_frame(FrameSpace.STATE2, labels),
        channels={"x": x[:-lag], "y": x[lag:]},
        axis_labels=labels,
        title=_title(series),
        meta=_meta(series),
    )


def _delay_labels(series: Any, components: Any, label: str, lag: int) -> tuple[str, str]:
    """Name the delay axes after the **channel**, and say the lag is in samples.

    Two small lies fixed at once.  The axes read ``x(t)`` / ``x(t - 16)`` for
    every input — the default ``label="x"``, even when the trajectory declares
    ``variables=("voltage",)`` or the caller selected ``components="z"``, so a
    reconstruction of one channel was captioned as another.  And ``t - 16`` was
    typeset as a *time* on an axis whose ``t`` is a time, while the lag is a
    count of **samples**: when the subject knows its ``dt`` the label now states
    the real delay (``voltage(t - 0.16)``), and when it does not — a bare array
    has no time axis — it says the unit out loud (``x(t - 16 samples)``).

    ``label=`` still wins when the caller gives it — it is the explicit override.
    """
    name = label
    if label == "x":  # the default: prefer what the data actually calls itself
        names = None
        if getattr(series, "y", None) is not None and getattr(series, "t", None) is not None:
            _, y, names, _ = _split_traj(series)
            name = _label(_component_index(components, names, y.shape[1]), names)
        elif isinstance(components, str):
            name = components
    dt = getattr(series, "dt", None)
    if dt is None:
        meta = getattr(series, "meta", None)
        dt = meta.get("dt") if isinstance(meta, dict) else None
    try:
        span = float(dt) * lag if dt is not None else None
    except (TypeError, ValueError):  # pragma: no cover - a non-numeric meta dt
        span = None
    lag_text = f"{lag} samples" if span is None else f"{span:g}"
    return (f"{name}(t)", f"{name}(t - {lag_text})")


#: The one runnable pair every delay error quotes, so the reader never has to
#: work out which spelling carries which unit.
_DELAY_HINT = (
    "    ts.plot(traj, 'delay_embedding', delay=7)              # 7 SAMPLES\n"
    "    traj.plot.delay_embedding(delay_time=0.12)             # 0.12 TIME UNITS\n"
    "(both spellings work at every plotting door, and mean the same thing at each)"
)


def _reject_missing_delay(delay: int | None, delay_time: float | None, tau: Any) -> None:
    """Raise unless exactly one delay spelling was given — before any data is read."""
    from tsdynamics.errors import InvalidParameterError

    if tau is not None:
        raise InvalidParameterError(
            "tau= is not a delay spelling in this library: it used to mean SAMPLES on "
            "ts.plot(...) and TIME UNITS on the method door. Say which you "
            f"mean:\n{_DELAY_HINT}"
        )
    if (delay is None) == (delay_time is None):
        raise InvalidParameterError(
            "a delay embedding needs exactly one of delay= (samples) or delay_time= "
            f"(time units):\n{_DELAY_HINT}"
        )


def _resolve_delay(
    series: Any, length: int, delay: int | None, delay_time: float | None, tau: Any
) -> int:
    """Resolve ``delay`` / ``delay_time`` to a validated sample lag.

    The single place the two spellings are turned into samples, shared by the
    transform and by :meth:`tsdynamics.data.Trajectory.__plot_spec__`, so the two
    front doors cannot disagree about what a delay means.
    """
    from tsdynamics.data import Trajectory
    from tsdynamics.errors import InvalidParameterError

    if tau is not None:
        raise InvalidParameterError(
            "tau= is not a delay spelling in this library: it used to mean SAMPLES on "
            "ts.plot(...) and TIME UNITS on the method door. Say which you "
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
    # It evaluates the vector field at points no trajectory visits — a flow, or a
    # bare right-hand side f(u, t) on its own.
    subjects=("flow", "function"),
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


def _map_kernel(series: Any) -> Any:
    """Return a scalar ``f(x)`` for a 1-D map's subject, or ``None``.

    The subject of a ``data`` transform may be a bare series (no ``f`` exists) or
    a trajectory that remembers the system it came from (``traj.system``).  Only
    a one-dimensional :class:`~tsdynamics.families.discrete.DiscreteMap` has a
    graph that can be drawn on these axes.
    """
    system = getattr(series, "system", None)
    if system is None or getattr(system, "family", None) != "map":
        return None
    if int(getattr(system, "dim", 0)) != 1:
        return None
    step = getattr(type(system), "_step", None)
    params = getattr(system, "params", None)
    if step is None or params is None:
        return None
    values = np.asarray(params.as_tuple(), dtype=float)

    def f(grid: np.ndarray) -> np.ndarray:
        return np.array(
            [float(np.asarray(step(np.array([v]), *values)).ravel()[0]) for v in grid], dtype=float
        )

    return f


def _map_graph(series: Any, lo: float, hi: float, n: int = 400) -> Part | None:
    """Return the ``y = f(x)`` curve of a 1-D map over ``[lo, hi]``, or ``None``.

    The curve spans **the axis range** — whatever window the cobweb settled on —
    so the graph is never clipped shorter than the box it is drawn in.
    """
    f = _map_kernel(series)
    if f is None:
        return None
    gx = np.linspace(lo, hi, int(n))
    try:
        gy = f(gx)
    except Exception:  # pragma: no cover - a kernel that refuses a scalar probe
        return None
    if not np.all(np.isfinite(gy)):
        return None
    return Part({"x": gx, "y": gy}, label="f(x)")


#: Resolution of the probe grid :func:`_natural_domain` reads the map's shape off.
_DOMAIN_PROBE = 1024


def _natural_domain(series: Any, lo: float, hi: float) -> tuple[float, float] | None:
    """Return the interval a 1-D map's cobweb belongs in, or ``None``.

    **The orbit's bounding box is the wrong default and it was the shipped one.**
    A cobweb exists to show that the staircase's corners land on the graph and
    that the diagonal crosses the hump — and an orbit converging onto a fixed
    point never visits the hump, so the picture routinely omitted the one feature
    it is drawn for.  ``ts.plot(Logistic(r=2.8).run(steps=30), "cobweb")`` drew
    ``f`` over ``[0.2, 0.69]``: no critical point, no second fixed point, no
    reason for anything on the canvas.

    A 1-D map does not declare a domain, but it *is* one — so this reads it off
    the kernel, in three widenings of the orbit's own span, each of which only
    ever adds structure the picture needs:

    1. out to the nearest **fixed point** (where ``f`` crosses the diagonal) and
       the nearest **turning point** (the hump) on either side;
    2. out to the **image** ``f(window)``, so every horizontal leg of the
       staircase has somewhere to land;
    3. out to the connected component of ``{x : f(x) ∈ window}`` around the
       orbit — the largest interval the map keeps inside the box — **only when
       that component is bounded** inside the probe bracket.  This is the step
       that recovers the unit square for the logistic map (``f ≥ 0`` exactly on
       ``[0, 1]``) while leaving a map whose graph is bounded everywhere
       (``Gauss``) on its tight window instead of expanding to the probe edge.

    Returns ``None`` for a bare series, a multi-dimensional map, or a kernel that
    refuses a scalar probe — in which case the caller keeps the orbit's span.
    """
    f = _map_kernel(series)
    if f is None:
        return None
    probe = _probe(f, lo, hi)
    if probe is None:
        return None
    bracket, values = probe

    # (1) out to the nearest fixed point AND the nearest turning point each side.
    #     Taking only the *nearest feature of either sort* stops at the hump and
    #     leaves the second fixed point — which is what anchors the staircase —
    #     off the canvas, so each kind is followed separately and the farther wins.
    for edge in (_roots(bracket, values - bracket), _turning_points(bracket, values)):
        below, above = edge[edge <= lo], edge[edge >= hi]
        lo = min(lo, float(below.max())) if below.size else lo
        hi = max(hi, float(above.min())) if above.size else hi

    # (2) the image, so the staircase's horizontal legs land inside the box.
    inside = (bracket >= lo) & (bracket <= hi)
    if inside.any():
        lo = min(lo, float(values[inside].min()))
        hi = max(hi, float(values[inside].max()))

    # (3) the largest interval the map keeps inside the box, when it is bounded.
    kept = (values >= lo) & (values <= hi)
    run = _component_containing(kept, bracket, lo, hi, values=values, level=(lo, hi))
    if run is not None:
        lo, hi = min(lo, run[0]), max(hi, run[1])
    return (lo, hi) if hi > lo else None


def _leaves_at(
    grid: np.ndarray, values: np.ndarray, inside: int, outside: int, bounds: Any
) -> float:
    """Where between two samples ``values`` crosses the bound it violates.

    Sub-grid, because the answer is *read* — ``(0.0, 1.0)`` for the logistic map
    is the unit square, and ``(4e-07, 0.9987)`` is the probe's resolution showing
    through the picture.
    """
    lo, hi = bounds
    level = lo if values[outside] < lo else hi
    span = values[outside] - values[inside]
    if span == 0.0:
        return float(grid[inside])
    t = (level - values[inside]) / span
    t = min(max(float(t), 0.0), 1.0)
    return float(grid[inside] + t * (grid[outside] - grid[inside]))


def _probe(f: Any, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample ``f`` around ``[lo, hi]``, clipped to where it is finite.

    A kernel is entitled to be undefined outside its domain — ``Chebyshev``'s
    ``arccos`` is ``NaN`` off ``[-1, 1]`` — and that is *information*, not a
    failure: the finite run containing the orbit IS the domain.  The probe is
    silenced (``np.errstate`` plus a warning filter) because it deliberately
    evaluates out of range, and a plotting default must not turn the map's own
    ``RuntimeWarning`` into the caller's problem.
    """
    import warnings

    width = max(hi - lo, 0.5 * (abs(lo) + abs(hi)))
    if not np.isfinite(width) or width <= 0:
        width = 1.0
    bracket = np.linspace(lo - width, hi + width, _DOMAIN_PROBE)
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = f(bracket)
    except Exception:  # pragma: no cover - a kernel that refuses a scalar probe
        return None
    finite = np.isfinite(values)
    if finite.all():
        return bracket, values
    run = _component_containing(finite, bracket, lo, hi)
    if run is None:
        return None
    keep = (bracket >= run[0]) & (bracket <= run[1])
    return bracket[keep], values[keep]


def _roots(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the sign-change roots of ``values``, refined by linear interpolation.

    The refinement is what makes the logistic map's answer read ``(0.0, 1.0)``
    rather than ``(-0.0057, 1.0004)``: a bare sign-change index names the sample
    *before* the root, which is outside the domain by one grid step.
    """
    idx = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    if not idx.size:
        return np.empty(0)
    lo_v, hi_v = values[idx], values[idx + 1]
    t = lo_v / (lo_v - hi_v)
    return np.asarray(grid[idx] + t * (grid[idx + 1] - grid[idx]), dtype=float)


def _turning_points(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return where ``values`` changes direction — the humps, at sample resolution."""
    slope = np.diff(values)
    idx = np.flatnonzero(np.diff(np.sign(slope)) != 0) + 1
    return grid[idx] if idx.size else np.empty(0)


def _component_containing(
    kept: np.ndarray,
    grid: np.ndarray,
    lo: float,
    hi: float,
    *,
    values: np.ndarray | None = None,
    level: Any = None,
) -> tuple[float, float] | None:
    """Return the ``[a, b]`` run of ``kept`` around ``[lo, hi]``, or ``None`` if unbounded.

    ``None`` when the run reaches either end of the probe bracket: an interval
    that runs off the probe is not a domain the library measured, it is the probe
    running out, and widening to it would be an invented answer.

    With ``values``/``level`` the endpoints are refined to where ``values``
    actually leaves the band, instead of the last sample that happened to be in.
    """
    seed = np.flatnonzero(kept & (grid >= lo) & (grid <= hi))
    if not seed.size:
        return None
    start = int(seed[0])
    while start > 0 and kept[start - 1]:
        start -= 1
    stop = int(seed[-1])
    while stop < kept.size - 1 and kept[stop + 1]:
        stop += 1
    if start == 0 or stop == kept.size - 1:
        return None
    if values is None or level is None:
        return (float(grid[start]), float(grid[stop]))
    return (
        _leaves_at(grid, values, start, start - 1, level),
        _leaves_at(grid, values, stop, stop + 1, level),
    )


#: Orbit steps a cobweb draws when the caller does not say.  Chosen to be legible
#: rather than complete: a map's default run is 1000 steps, and a 1000-step
#: staircase is a solid block over the parabola and the diagonal it exists to be
#: read against.  Fifty shows a period-doubling settle in full and gives a
#: chaotic orbit enough corners to look chaotic.
_COBWEB_DEFAULT_STEPS = 50


@plot_transform(
    name="cobweb",
    subjects=("map",),  # a cobweb is the staircase of a 1-D MAP iteration
    source="data",
    kind=PlotKind.COBWEB,
    frame=FrameSpace.STATE2,
    ndim=2,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points"),
    presentation=Presentation(aspect="equal"),
    example=lambda primitive: (_demo_map(), {"components": 0}),
    doc="The 1-D staircase x_{n+1} vs x_n with the y = x diagonal.",
)
def cobweb(
    series: np.ndarray | Trajectory,
    *,
    components: int | str = 0,
    label: str = "x",
    domain: tuple[float, float] | None = None,
    steps: int | None = None,
) -> Geometry:
    """Compute the staircase geometry of a 1-D map orbit.

    Two parts: the ``y = x`` diagonal and the staircase itself — vertical
    segments from ``(x_n, x_n)`` to ``(x_n, x_{n+1})`` and horizontal segments
    back to the diagonal.

    Parameters
    ----------
    series : ndarray or Trajectory
        A 1-D orbit ``x_0, x_1, ...`` (or a trajectory's ``component``).
    components : int or str, optional
        Component to read when ``series`` is a Trajectory.
    label : str, optional
        Axis-label base; the axes read ``x_n`` / ``x_(n+1)``.
    domain : (float, float), optional
        The interval to draw ``f`` and the axes over — **the map's own domain**,
        which is the picture a cobweb is for::

            ts.plot(logistic_orbit, "cobweb", domain=(0, 1))

        Default ``None`` = **the map's own domain**, read off the kernel by
        :func:`_natural_domain` (the unit square for the logistic map, whatever
        ``r`` and whatever the orbit did).  It used to be the span the orbit
        actually visited, which is the wrong picture for any converging orbit:
        the point of a cobweb is that the staircase's corners land on the hump,
        and an orbit settling onto a fixed point never visits it.  A bare series
        — which has no ``f`` to read — still falls back to the orbit's span.
    steps : int, optional
        How many orbit steps to **draw**.  ``None`` uses
        :data:`_COBWEB_DEFAULT_STEPS`; pass ``0`` for the whole orbit.

        A cobweb is a teaching picture and nothing else, and the default was
        burying it: ``ts.plot(logistic, "cobweb")`` ran the map its default 1000
        steps and drew all of them, so the parabola and the diagonal — the two
        curves the staircase is supposed to be read against — came out under a
        solid green block.  Drawing the *first* steps rather than the last is
        deliberate: the walk-in is the part that teaches.

    Returns
    -------
    Geometry

    Raises
    ------
    ValueError
        If the orbit has fewer than two points.

    Notes
    -----
    **The map curve is drawn whenever it can be**: a cobweb without
    :math:`x_{n+1} = f(x_n)` cannot be read — the whole point is that the
    staircase's corners land ON the curve, and without it the picture is a
    zig-zag and a diagonal in an empty box.  The graph needs the map itself, so
    it is added when the subject carries one (``traj.system`` on a 1-D
    :class:`~tsdynamics.families.discrete.DiscreteMap`, which is what
    ``ts.plot(logistic, "cobweb")`` hands in after the registry runs it) and
    silently omitted for a bare series, which genuinely has no ``f``.
    """
    x = _scalar_series(series, components)
    if x.shape[0] < 2:
        raise ValueError("a cobweb needs at least two orbit points.")
    drawn = _COBWEB_DEFAULT_STEPS if steps is None else int(steps)
    if drawn > 0:
        x = x[: drawn + 1]
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
    if domain is not None:
        lo, hi = (float(domain[0]), float(domain[1]))
        if not (hi > lo):
            from tsdynamics.errors import InvalidParameterError

            raise InvalidParameterError(
                f"cobweb domain= is the interval to draw f over, so it needs lo < hi; "
                f"got {domain!r}. For the logistic map:  domain=(0, 1)"
            )
    else:
        lo = float(min(x.min(), stair_y.min()))
        hi = float(max(x.max(), stair_y.max()))
        natural = _natural_domain(series, lo, hi)
        if natural is not None:
            lo, hi = natural
    diag = np.array([lo, hi], dtype=float)
    labels = (f"{label}_n", f"{label}_(n+1)")
    parts = [Part({"x": diag, "y": diag}, label="y = x")]
    graph = _map_graph(series, lo, hi)
    if graph is not None:
        parts.append(graph)
    parts.append(Part({"x": stair_x, "y": stair_y}, label="orbit"))
    return Geometry(
        "cobweb",
        make_frame(FrameSpace.STATE2, labels),
        parts,
        axis_labels=labels,
        # The axes ARE the domain — the box the map lives in — whether it was
        # named or read off the kernel.  Leaving them unset let matplotlib
        # autoscale to the orbit, which put the graph's own ends outside the
        # frame it was computed for.
        axis_limits=((lo, hi), (lo, hi)),
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
    aliases=("field",),  # the kind= recipe spelling
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
    components: int | str | None = None,
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
    components : int or str, optional
        Which field **block** to plot when the state packs several
        (``meta["field_labels"]``, e.g. Gray-Scott's ``("u", "v")``).  ``None``
        selects the **last** block (the activator convention).

    Returns
    -------
    Geometry
    """
    _, y, _, _ = _split_traj(source)
    shape, labels = _resolve_field_shape(source, field_shape)
    block = _select_field_block(y, shape, labels, components)
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
    components: int | str | None,
) -> np.ndarray:
    """Slice the chosen field block (shape ``(T, prod(shape))``) out of the state."""
    cells = int(np.prod(shape))
    n_blocks = max(1, y.shape[1] // cells) if cells else 1
    if labels is not None:
        n_blocks = len(labels)
    if components is None:
        block = n_blocks - 1  # the activator / last block by convention
    elif isinstance(components, str):
        if labels is None:
            raise KeyError(
                f"cannot resolve field block {components!r}: the trajectory declares no "
                f"`field_labels`; select a block by integer index instead."
            )
        try:
            block = labels.index(components)
        except ValueError:
            raise KeyError(
                f"unknown field block {components!r}; declared field_labels: {labels}"
            ) from None
    else:
        block = int(components) % n_blocks
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
