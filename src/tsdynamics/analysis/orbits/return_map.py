"""First-return maps — the discrete dynamics hidden inside a flow.

A *return map* turns a continuous trajectory into a one-dimensional discrete
map by recording successive values of a recurring scalar observable and
plotting each value against its successor :math:`(v_n, v_{n+1})`.  Two classic
constructions:

- **Successive extrema** (Lorenz, 1963) — the map of one coordinate's local
  maxima (or minima).  The Lorenz attractor's :math:`z`-maxima trace out the
  famous single-humped cusp map :math:`z_{n+1} = F(z_n)`, exposing the
  low-dimensional dynamics underneath the strange attractor.
- **Poincaré first return** — the value of an observable at successive
  crossings of a surface of section, i.e. the section's one-dimensional return
  map for that component.

Both reveal whether the asymptotic motion is effectively one-dimensional: a
tight, single-valued curve means a (noisy) 1-D map governs the dynamics; a
filled cloud means it does not.

References
----------
Lorenz, E. N. (1963). *Deterministic nonperiodic flow.*
J. Atmos. Sci. 20, 130--141.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tsdynamics.families import Trajectory

from .._result import AnalysisResult
from .poincare import _seeded_ic, poincare_section

__all__ = ["ReturnMap", "return_map"]

_KINDS = ("max", "min", "poincare")


@dataclass(frozen=True)
class ReturnMap(AnalysisResult):
    """
    Result of :func:`return_map`.

    An :class:`~tsdynamics.analysis._result.AnalysisResult`, so it carries
    ``.meta`` / ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam.  The recorded
    observable values are :attr:`values`; the return map itself is the pair
    (:attr:`current`, :attr:`successor`) = :math:`(v_n, v_{n+1})`.  Iterate for
    ``(current, successor)`` pairs, or use :meth:`flat` for the scatter-ready
    arrays.
    """

    current: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)  # v_n
    successor: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)  # v_{n+1}
    values: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)  # observable
    times: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)  # times
    observable: int = 0  # which state component was recorded
    kind: str = "max"  # "max" | "min" | "poincare"

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        return iter(zip(self.current, self.successor, strict=True))

    def __len__(self) -> int:
        return int(self.current.size)

    def flat(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the scatter-plot arrays ``(current, successor)``."""
        return self.current, self.successor

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe this return map as a backend-agnostic :class:`PlotSpec`.

        Builds a ``RETURN_MAP`` scatter of :math:`(v_n, v_{n+1})` with the
        diagonal :math:`v_{n+1} = v_n` drawn as a reference line (its fixed-point
        locus).  The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec
        never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"return_map"``).  ``None`` uses
            ``RETURN_MAP``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.RETURN_MAP
        cur = np.asarray(self.current, dtype=float)
        suc = np.asarray(self.successor, dtype=float)
        layers = [Layer(PlotKind.SCATTER, {"x": cur, "y": suc}, label=r"$v_{n+1}$ vs $v_n$")]
        if cur.size:
            both = np.concatenate([cur, suc])
            diag = np.array([float(both.min()), float(both.max())])
            layers.append(Layer(PlotKind.LINE, {"x": diag, "y": diag}, label="$v_{n+1}=v_n$"))
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            aspect="equal",
            title=f"{self.kind} return map",
            x=Axis(label=r"$v_n$"),
            y=Axis(label=r"$v_{n+1}$"),
            layers=layers,
            meta=dict(self.meta),
        )

    def cobweb(self, kind: str | None = None) -> Any:
        r"""Describe the return map's cobweb (staircase) as a :class:`PlotSpec`.

        The cobweb diagram traces the iteration :math:`v_{n+1} = F(v_n)` as a
        staircase: from a point on the diagonal it steps vertically to the return
        curve, then horizontally back to the diagonal, and repeats.  This emits a
        ``COBWEB`` spec carrying the scatter of the return points
        :math:`(v_n, v_{n+1})`, the diagonal :math:`v_{n+1} = v_n`, and the
        staircase ``LINE`` itself built from the recorded sequence.  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls
        a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``COBWEB``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.COBWEB
        cur = np.asarray(self.current, dtype=float)
        suc = np.asarray(self.successor, dtype=float)
        layers = [
            Layer(PlotKind.SCATTER, {"x": cur, "y": suc}, label=r"$v_{n+1}$ vs $v_n$"),
        ]
        if cur.size:
            both = np.concatenate([cur, suc])
            diag = np.array([float(both.min()), float(both.max())])
            layers.append(Layer(PlotKind.LINE, {"x": diag, "y": diag}, label="$v_{n+1}=v_n$"))
            stair_x, stair_y = _cobweb_path(cur, suc)
            layers.append(
                Layer(
                    PlotKind.LINE,
                    {"x": stair_x, "y": stair_y},
                    label="cobweb",
                    style={"lw": 0.8, "alpha": 0.8},
                )
            )
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            aspect="equal",
            title=f"{self.kind} cobweb",
            x=Axis(label=r"$v_n$"),
            y=Axis(label=r"$v_{n+1}$"),
            layers=layers,
            legend=Legend() if len(layers) > 1 else None,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        return (
            f"ReturnMap(kind={self.kind!r}, observable={self.observable}, "
            f"{self.values.size} values)"
        )


def _cobweb_path(current: np.ndarray, successor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Build the staircase polyline of the iteration from the return pairs.

    For each return pair :math:`(v_n, v_{n+1})` the cobweb steps vertically from
    the diagonal point :math:`(v_n, v_n)` up to the curve :math:`(v_n, v_{n+1})`,
    then horizontally to the next diagonal point :math:`(v_{n+1}, v_{n+1})`.
    Returns the concatenated ``(x, y)`` vertices of that polyline.
    """
    cur = np.asarray(current, dtype=float)
    suc = np.asarray(successor, dtype=float)
    xs: list[float] = []
    ys: list[float] = []
    for vn, vn1 in zip(cur, suc, strict=True):
        xs.extend((vn, vn))  # vertical: (vn, vn) -> (vn, vn1)
        ys.extend((vn, vn1))
        xs.append(vn1)  # horizontal: (vn, vn1) -> (vn1, vn1)
        ys.append(vn1)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def return_map(
    system: Any,
    component: int | str = 0,
    *,
    method: str = "max",
    plane: tuple[Any, ...] | None = None,
    direction: int = +1,
    n: int = 2000,
    final_time: float = 200.0,
    dt: float = 0.01,
    transient: float = 0.0,
    skip_crossings: int = 0,
    ic: Any | None = None,
    seed: int | None = None,
    **integrate_kwargs: Any,
) -> ReturnMap:
    r"""
    First-return map of a recurring observable.

    Records a sequence of scalar values :math:`v_0, v_1, \dots` from the motion
    and pairs each with its successor, giving the one-dimensional map
    :math:`v_{n+1} = F(v_n)` that organises the dynamics.

    Parameters
    ----------
    system : System, Trajectory, or array-like
        What to read the observable from.  A continuous
        :class:`~tsdynamics.families.ContinuousSystem` is integrated first; a
        :class:`~tsdynamics.data.Trajectory` is read directly; a 1-D array is
        treated as the observable series itself (``method`` must be ``"max"`` or
        ``"min"``).  (Trajectory / array inputs are the ``data`` overload.)
    component : int or str, default 0
        Which state component to record (names allowed when the system /
        trajectory declares ``variables``).  Ignored when ``system`` is a raw
        1-D series.
    method : {"max", "min", "poincare"}, default "max"
        ``"max"`` / ``"min"`` record successive local maxima / minima of the
        observable (the Lorenz construction); ``"poincare"`` records the
        observable at successive section crossings (needs ``plane``).
    plane : tuple, optional
        ``(i, c)`` or ``(normal, offset)`` — the section for ``method="poincare"``
        (see :func:`~tsdynamics.analysis.orbits.poincare_section`).
    direction : {+1, -1, 0}, default +1
        Crossing-direction filter (``method="poincare"`` only).
    n : int, default 2000
        Number of section crossings to collect when integrating a system in
        ``method="poincare"`` mode.
    final_time, dt : float
        Integration horizon and detection / output step used when ``system`` is
        a flow.  In extremum mode ``dt`` only needs to resolve the peaks; the
        recorded value is sharpened by parabolic interpolation, so a coarse grid
        still gives accurate extrema.  In ``method="poincare"`` mode only ``dt``
        is used (the section is marched until ``n`` crossings); ``final_time``,
        ``ic``, ``transient`` and ``**integrate_kwargs`` apply to extremum mode
        and are ignored.
    transient : float, default 0.0
        Elapsed **time** discarded before recording extrema (``method="max"`` /
        ``"min"``, system input).
    skip_crossings : int, default 0
        Number of leading **crossings** discarded before recording
        (``method="poincare"``, system input).
    ic : array-like, optional
        Initial state when ``system`` is a flow.
    seed : int, optional
        Seed for the random initial condition when the system has none; makes
        the map reproducible.
    **integrate_kwargs
        Forwarded to ``system.integrate`` (extremum mode, system input).

    Returns
    -------
    ReturnMap
        The recorded ``values`` and the paired ``(current, successor)`` arrays.

    Examples
    --------
    >>> rm = return_map(Lorenz(), "z", method="max", final_time=400.0, transient=40.0)
    >>> x, y = rm.flat()       # the cusp map z_n -> z_{n+1}
    >>> rm = return_map(Rossler(), 0, method="poincare", plane=(0, 0.0), n=400)
    """
    method = method.lower()
    if method not in _KINDS:
        raise ValueError(f"method must be one of {_KINDS}, got {method!r}.")

    if method == "poincare":
        values, times, obs_idx = _poincare_observable(
            system, component, plane, direction, n, skip_crossings, dt, seed
        )
    else:
        values, times, obs_idx = _extremum_observable(
            system, component, method, final_time, dt, transient, ic, seed, integrate_kwargs
        )

    current = values[:-1]
    successor = values[1:]
    meta: dict[str, Any] = {"kind": method, "observable": obs_idx, "n": int(values.size)}
    if plane is not None:
        meta["plane"] = plane
    src_name = getattr(type(system), "__name__", None)
    if not isinstance(system, np.ndarray | list | tuple):
        meta["source"] = src_name
    return ReturnMap(
        current=current,
        successor=successor,
        values=values,
        times=times,
        observable=obs_idx,
        kind=method,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# observable extraction
# ---------------------------------------------------------------------------


def _observable_index(obj: Any, component: int | str) -> int:
    """Resolve ``component`` to a column index using the object's ``variables``."""
    if isinstance(component, str):
        names = getattr(obj, "variables", None)
        if not names:
            raise ValueError(
                "a named component needs the system / trajectory to declare `variables`"
            )
        try:
            return list(names).index(component)
        except ValueError:
            raise ValueError(f"unknown component {component!r}; declared: {tuple(names)}") from None
    return int(component)


def _extremum_observable(
    system: Any,
    component: int | str,
    method: str,
    final_time: float,
    dt: float,
    transient: float,
    ic: Any | None,
    seed: int | None,
    integrate_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract the observable series (+ times) for extremum mode, from any input type."""
    if isinstance(system, Trajectory):
        traj = system.after(transient) if transient else system
        idx = _observable_index(traj, component)
        return *_local_extrema(traj.y[:, idx], traj.t, method), idx
    if hasattr(system, "is_discrete"):  # a System
        if system.is_discrete:
            raise TypeError(
                "extremum return maps need a continuous flow; for a map, iterate and "
                "pass the series, or use orbit_diagram."
            )
        idx = _observable_index(system, component)
        seeded = _seeded_ic(system, ic, seed)
        run_ic = seeded if seeded is not None else ic
        traj = system.integrate(final_time=final_time, dt=dt, ic=run_ic, **integrate_kwargs)
        if transient:
            traj = traj.after(transient)
        return *_local_extrema(traj.y[:, idx], traj.t, method), idx
    # raw 1-D series
    series = np.asarray(system, dtype=float)
    if series.ndim != 1:
        raise ValueError(
            f"a raw-series input must be 1-D (got shape {series.shape}); pass a Trajectory "
            f"or System to select a component."
        )
    return *_local_extrema(series, None, method), 0


def _poincare_observable(
    system: Any,
    component: int | str,
    plane: tuple[Any, ...] | None,
    direction: int,
    n: int,
    skip_crossings: int,
    dt: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract the observable series (+ times) for Poincaré-return mode."""
    if plane is None:
        raise ValueError("method='poincare' needs a `plane=(i, c)`.")
    if isinstance(system, Trajectory):
        # the system path discards `skip_crossings` crossings inside
        # poincare_section; the data path has them all, so drop the first
        # `skip_crossings` here to match.
        section = poincare_section(system, plane, direction=direction)
        idx = _observable_index(section, component)
        skip = int(skip_crossings)
        return section.y[skip:, idx], section.t[skip:], idx
    if hasattr(system, "is_discrete"):
        section = poincare_section(
            system,
            plane,
            direction=direction,
            n=n,
            skip_crossings=int(skip_crossings),
            dt=dt,
            seed=seed,
        )
        idx = _observable_index(section, component)
        return section.y[:, idx], section.t, idx
    raise TypeError("method='poincare' needs a System or Trajectory input, not a raw series.")


def _local_extrema(series: Any, times: Any | None, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Strict interior local extrema of ``series``, sharpened by parabolic fit.

    A sample ``i`` is a maximum when ``s[i-1] < s[i] > s[i+1]`` (minimum when
    the inequalities flip).  The recorded value and time are refined to the
    vertex of the parabola through the three samples, so a coarse sampling step
    still yields a sub-sample-accurate extremum (peak interpolation; e.g.
    Smith, *Spectral Audio Signal Processing*).
    """
    s = np.asarray(series, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"extremum series must be 1-D, got shape {s.shape}.")
    sign = 1.0 if kind == "max" else -1.0
    v = sign * s
    if v.size < 3:
        return np.empty(0), np.empty(0)

    interior = v[1:-1]
    is_peak = (interior > v[:-2]) & (interior > v[2:])
    idx = np.nonzero(is_peak)[0] + 1
    if idx.size == 0:
        return np.empty(0), np.empty(0)

    y0 = v[idx]
    ym = v[idx - 1]
    yp = v[idx + 1]
    denom = ym - 2.0 * y0 + yp
    # vertex offset in samples, in [-1/2, 1/2]; flat (denom==0) -> no shift
    delta = np.where(denom != 0.0, 0.5 * (ym - yp) / denom, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    peak = sign * (y0 - 0.25 * (ym - yp) * delta)

    if times is None:
        peak_t = idx.astype(float) + delta
    else:
        t = np.asarray(times, dtype=float)
        # scale the offset by the spacing on the side it points to — correct on a
        # non-uniform grid, and equal to dt on the uniform output grid.
        spacing = np.where(delta >= 0.0, t[idx + 1] - t[idx], t[idx] - t[idx - 1])
        peak_t = t[idx] + delta * spacing
    return peak, peak_t


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
