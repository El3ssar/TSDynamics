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
from .._result_json import _sig
from .poincare import _seeded_ic, poincare_section

__all__ = ["ReturnMap", "return_map"]

_KINDS = ("max", "min", "poincare")


@dataclass(frozen=True)
class ReturnMap(AnalysisResult):
    """
    Result of :func:`return_map`.

    An :class:`~tsdynamics.analysis._result.AnalysisResult`, so it carries
    ``.meta`` / the readout ``repr`` / ``.to_dict()`` / the ``.plot`` seam.  The recorded
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

    def __getitem__(self, key: Any) -> Any:
        """Return the ``(current, successor)`` pair at position ``key`` — what iteration yields.

        It was sized and iterable but **not** subscriptable, so ``rm[0]`` raised
        ``TypeError`` while ``len(rm)`` and ``for a, b in rm`` both worked, and
        ``np.asarray(rm)`` degenerated to a 0-d object array.
        """
        return list(zip(self.current, self.successor, strict=True))[key]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the map as the ``(n, 2)`` scatter it draws: :math:`(v_n, v_{n+1})`."""
        arr = np.column_stack(
            [np.asarray(self.current, dtype=float), np.asarray(self.successor, dtype=float)]
        )
        return arr.astype(dtype, copy=bool(copy)) if dtype is not None else arr

    def flat(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the scatter-plot arrays ``(current, successor)``."""
        return self.current, self.successor

    def __plot_spec__(self, kind: str | None = None) -> Any:
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
        from .. import _plotbuilder as pb

        xlabel, ylabel, legend_label = self._axis_labels()
        cur = np.asarray(self.current, dtype=float)
        suc = np.asarray(self.successor, dtype=float)
        layers = [pb.scatter(cur, suc, label=legend_label)]
        if cur.size:
            layers.append(pb.diagonal(cur, suc))
        return pb.spec(
            kind,
            "return_map",
            layers=layers,
            aspect="equal",
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{self.kind} return map of {self._observable_label()}",
            meta=self.meta,
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
        from .. import _plotbuilder as pb

        xlabel, ylabel, legend_label = self._axis_labels()
        cur = np.asarray(self.current, dtype=float)
        suc = np.asarray(self.successor, dtype=float)
        layers = [pb.scatter(cur, suc, label=legend_label)]
        if cur.size:
            layers.append(pb.diagonal(cur, suc))
            stair_x, stair_y = _cobweb_path(cur, suc)
            layers.append(
                pb.line(stair_x, stair_y, label="cobweb", style={"lw": 0.8, "alpha": 0.8})
            )
        return pb.spec(
            kind,
            "cobweb",
            layers=layers,
            aspect="equal",
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{self.kind} cobweb of {self._observable_label()}",
            legend=len(layers) > 1,
            meta=self.meta,
        )

    def _observable_label(self) -> str:
        """Return the recorded component's declared name, or its index."""
        variables = self.meta.get("variables") if self.meta else None
        if variables is not None:
            names = tuple(variables)
            if 0 <= int(self.observable) < len(names):
                return str(names[int(self.observable)])
        return f"component {int(self.observable)}"

    def _axis_labels(self) -> tuple[str, str, str]:
        r"""Return ``(x, y, legend)`` labels naming the observable this map is OF.

        The axes read :math:`v_n` / :math:`v_{n+1}` for every input, so
        ``return_map(traj, components="z")`` — the Lorenz z-maxima cusp, the
        textbook example — drew a figure captioned about a ``v`` that appears
        nowhere in the system.  The repr already names the channel (``"85 returns
        of z"``), so the picture and the sentence beside it disagreed about what
        was measured.  Here the subscripts are genuinely mathematical, so the
        name is set in mathtext rather than plain (the rule
        :func:`tsdynamics.analysis._plotbuilder.axis_labels` follows for a label
        that *is* a name).
        """
        v = self._observable_label()
        # "component 2" is a sentence, not a symbol; subscripting it reads badly,
        # so the indexed fallback keeps the generic symbol it always had.
        sym = v if " " not in v else "v"
        return f"${sym}_n$", f"${sym}_{{n+1}}$", f"${sym}_{{n+1}}$ vs ${sym}_n$"

    def _answer(self) -> str:
        """Return how many returns were collected, of which observable."""
        n = int(self.current.size)
        label = self._observable_label()
        which = {"max": "successive maxima", "min": "successive minima"}.get(
            self.kind, "successive crossings"
        )
        return f"{n} returns of {label} · {which}"

    def _details(self) -> tuple[str, ...]:
        """Return the span of the recorded observable."""
        v = np.asarray(self.values, dtype=float)
        if not v.size:
            return ()
        return (f"{self._observable_label()} ∈ [{_sig(v.min(), 4)}, {_sig(v.max(), 4)}]",)


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
    components: int | str = 0,
    *,
    kind: str = "max",
    plane: tuple[Any, ...] | None = None,
    direction: int = +1,
    n: int = 2000,
    final_time: float = 200.0,
    dt: float = 0.01,
    transient: float = 0.0,
    skip_crossings: int = 0,
    ic: Any | None = None,
    seed: int | None = 0,
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
        treated as the observable series itself (``kind`` must be ``"max"`` or
        ``"min"``).  (Trajectory / array inputs are the ``data`` overload.)
    components : int or str, default 0
        Which state component to record (names allowed when the system /
        trajectory declares ``variables``).  Ignored when ``system`` is a raw
        1-D series.
    kind : {"max", "min", "poincare"}, default "max"
        ``"max"`` / ``"min"`` record successive local maxima / minima of the
        observable (the Lorenz construction); ``"poincare"`` records the
        observable at successive section crossings (needs ``plane``).
    plane : tuple, optional
        ``(i, c)`` or ``(normal, offset)`` — the section for ``kind="poincare"``
        (see :func:`~tsdynamics.analysis.orbits.poincare_section`).
    direction : {+1, -1, 0}, default +1
        Crossing-direction filter (``kind="poincare"`` only).
    n : int, default 2000
        Number of section crossings to collect when integrating a system in
        ``kind="poincare"`` mode.
    final_time : float, default 200.0
        Integration horizon, in **time units**, when ``system`` is a flow
        (extremum mode only).
    dt : float, default 0.01
        Step, in **time units**, when ``system`` is a flow.  In extremum mode it
        is the *output sampling* step and only needs to resolve the peaks — the
        recorded value is sharpened by parabolic interpolation, so a coarse grid
        still gives accurate extrema.  In ``kind="poincare"`` mode it is the
        crossing-**detection** step of the section march.  Ignored for a
        ``Trajectory`` / a bare series, which are already sampled.
        In ``kind="poincare"`` mode ``final_time``, ``ic``, ``transient`` and
        ``**integrate_kwargs`` do not apply and are ignored (the section is
        marched until ``n`` crossings).
    transient : float, default 0.0
        Dynamics discarded before recording extrema, in **time units** — the unit
        ``final_time`` and ``run(transient=)`` use (``kind="max"`` / ``"min"``,
        system input).  The *section* transient is the separate
        ``skip_crossings``, because a count of crossings is a different quantity
        from an elapsed time.
    skip_crossings : int, default 0
        Number of leading **crossings** discarded before recording
        (``kind="poincare"``, system input).
    ic : array-like, optional
        Initial state when ``system`` is a flow.
    seed : int, default 0
        Seed for the random initial condition when the system has none, so the
        map is reproducible.  Pass ``seed=None`` for an unseeded draw.

        .. versionchanged:: 6.0
            Was ``None`` — the same call returned a different map each run.
    **integrate_kwargs
        Forwarded to ``system.run`` (extremum mode, system input).

    Returns
    -------
    ReturnMap
        The recorded ``values`` and the paired ``(current, successor)`` arrays.

    Examples
    --------
    >>> rm = return_map(Lorenz(), "z", kind="max", final_time=400.0, transient=40.0)
    >>> x, y = rm.flat()       # the cusp map z_n -> z_{n+1}
    >>> rm = return_map(Rossler(), 0, kind="poincare", plane=(0, 0.0), n=400)
    """
    kind = kind.lower()
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}.")

    if kind == "poincare":
        values, times, obs_idx = _poincare_observable(
            system, components, plane, direction, n, skip_crossings, dt, seed
        )
    else:
        values, times, obs_idx = _extremum_observable(
            system, components, kind, final_time, dt, transient, ic, seed, integrate_kwargs
        )

    current = values[:-1]
    successor = values[1:]
    meta: dict[str, Any] = {"kind": kind, "observable": obs_idx, "n": int(values.size)}
    if plane is not None:
        meta["plane"] = plane
    names = tuple(getattr(system, "variables", ()) or ())
    if names:
        # The repr says "26 returns of z", not "of component 2" (S3 inbox d).
        meta["variables"] = names
    src_name = getattr(type(system), "__name__", None)
    if not isinstance(system, np.ndarray | list | tuple):
        meta["source"] = src_name
    return ReturnMap(
        current=current,
        successor=successor,
        values=values,
        times=times,
        observable=obs_idx,
        kind=kind,
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
    if hasattr(system, "family"):  # a System
        if system.family == "map" or getattr(system, "_is_discrete", False):
            raise TypeError(
                "extremum return maps need a continuous flow; for a map, iterate and "
                "pass the series, or use orbit_diagram."
            )
        idx = _observable_index(system, component)
        seeded = _seeded_ic(system, ic, seed)
        run_ic = seeded if seeded is not None else ic
        traj = system.run(final_time=final_time, dt=dt, ic=run_ic, **integrate_kwargs)
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
        raise ValueError("kind='poincare' needs a `plane=(i, c)`.")
    if isinstance(system, Trajectory):
        # the system path discards `skip_crossings` crossings inside
        # poincare_section; the data path has them all, so drop the first
        # `skip_crossings` here to match.
        section = poincare_section(system, plane, direction=direction)
        idx = _observable_index(section, component)
        skip = int(skip_crossings)
        return section.y[skip:, idx], section.t[skip:], idx
    if hasattr(system, "family"):
        section = poincare_section(
            system,
            plane,
            direction=direction,
            crossings=n,
            skip_crossings=int(skip_crossings),
            dt=dt,
            seed=seed,
        )
        idx = _observable_index(section, component)
        return section.y[:, idx], section.t, idx
    raise TypeError("kind='poincare' needs a System or Trajectory input, not a raw series.")


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
