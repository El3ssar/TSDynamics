"""Orbit diagrams — asymptotic states swept across a parameter."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .._result import AnalysisResult
from .poincare import _seeded_ic

__all__ = ["OrbitDiagram", "orbit_diagram"]


@dataclass(frozen=True)
class OrbitDiagram(AnalysisResult):
    """
    Result of :func:`orbit_diagram`.

    An :class:`~tsdynamics.analysis._result.AnalysisResult`, so it carries
    ``.meta`` / ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam.  Iterate to
    get ``(value, points)`` pairs, or use :meth:`flat` for the scatter-ready
    arrays.
    """

    param: str = ""
    values: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)  # (V,)
    points: list[np.ndarray] = field(default_factory=list, compare=False)  # per value (n, k)
    components: tuple[int, ...] = ()

    def __iter__(self) -> Iterator[tuple[Any, np.ndarray]]:
        return iter(zip(self.values, self.points, strict=True))

    def __len__(self) -> int:
        return len(self.values)

    def flat(self, component: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Flatten to scatter-plot arrays ``(x, y)``.

        ``x`` repeats each parameter value once per recorded point; ``y`` is
        the chosen recorded component.
        """
        x = np.concatenate(
            [np.full(p.shape[0], v) for v, p in zip(self.values, self.points, strict=True)]
        )
        y = np.concatenate([p[:, component] for p in self.points])
        return x, y

    def periods(
        self, *, component: int = 0, max_period: int = 16, rtol: float = 0.01
    ) -> np.ndarray:
        """
        Return the detected period at each swept parameter value.

        Counts the distinct asymptotic branches in the recorded orbit — the
        period of a periodic window — by clustering the points of one component
        with a scale-free gap test: a new branch starts where the sorted-value
        gap exceeds ``rtol`` times the orbit's range.

        Parameters
        ----------
        component : int, default 0
            Which recorded component to count branches in.
        max_period : int, default 16
            Periods above this are reported as ``0`` (treated as aperiodic /
            chaotic — too many branches to resolve as a cycle).
        rtol : float, default 0.01
            Relative gap (fraction of the per-value range) separating branches.

        Returns
        -------
        numpy.ndarray of int
            One entry per parameter value: the period ``1, 2, 4, …``, ``0`` for
            aperiodic, or ``-1`` where the sweep recorded no points (diverged).
        """
        out = np.empty(len(self.values), dtype=int)
        for k, pts in enumerate(self.points):
            if pts.shape[0] == 0:
                out[k] = -1
                continue
            p = _count_branches(pts[:, component], rtol)
            out[k] = p if p <= max_period else 0
        return out

    def bifurcation_points(
        self, *, component: int = 0, max_period: int = 16, rtol: float = 0.01
    ) -> np.ndarray:
        """
        Parameter values where the detected period changes.

        Locates the boundaries of the period-doubling cascade (and other
        bifurcations) as the midpoints between consecutive swept values across
        which :meth:`periods` differs.  Transitions touching a diverged value
        (``-1``) are skipped.

        Returns
        -------
        numpy.ndarray of float
            Estimated bifurcation parameter values, in sweep order.  Their
            resolution is the spacing of ``values``.
        """
        p = self.periods(component=component, max_period=max_period, rtol=rtol)
        changed = (p[:-1] != p[1:]) & (p[:-1] != -1) & (p[1:] != -1)
        (i,) = np.nonzero(changed)
        return cast(np.ndarray, 0.5 * (self.values[i] + self.values[i + 1]))

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe this orbit diagram as a backend-agnostic :class:`PlotSpec`.

        Builds an ``ORBIT_DIAGRAM`` scatter of the asymptotic state (first
        recorded component) against the swept parameter — the bifurcation diagram
        — via :meth:`flat`.  The cascade onsets that :meth:`bifurcation_points`
        detects are carried as ``"vline"``
        :class:`~tsdynamics.viz.spec.Annotation` reference lines (each labelled
        with the period it opens onto), so a renderer draws the period-doubling
        boundaries over the diagram.  The :mod:`tsdynamics.viz.spec` import is
        lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"bifurcation"``).  ``None`` uses
            ``ORBIT_DIAGRAM``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Annotation, Axis, Layer, PlotKind, PlotSpec

        x, y = self.flat()
        spec_kind = PlotKind(kind) if kind is not None else PlotKind.ORBIT_DIAGRAM
        annotations: list[Annotation] = []
        if len(self.values) > 1:
            onsets = self.bifurcation_points()
            periods = self.periods()
            for onset in np.asarray(onsets, dtype=float).ravel():
                # Label the line with the period the cascade opens *onto* (the
                # period just to the right of the onset).
                j = int(np.searchsorted(self.values, onset))
                p = int(periods[j]) if 0 <= j < periods.size else 0
                text = f"period {p}" if p > 0 else "bifurcation"
                annotations.append(Annotation(kind="vline", x=float(onset), text=text))
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title="orbit diagram",
            x=Axis(label=self.param),
            y=Axis(label="asymptotic state"),
            layers=[Layer(PlotKind.SCATTER, {"x": x, "y": y}, style={"s": 1.0})],
            annotations=annotations,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        return (
            f"OrbitDiagram({self.param!r}, {len(self.values)} values, "
            f"{self.points[0].shape[0] if self.points else 0} points/value)"
        )


def _count_branches(col: np.ndarray, rtol: float) -> int:
    """
    Distinct branches in ``col`` — clusters separated by a gap > ``rtol``·range.

    The scale-relative *negligible-spread* guard (``span <= rtol·scale``) is what
    keeps this honest for flows: a periodic-orbit branch recorded from a Poincaré
    map differs only by integration noise, so its whole spread is tiny relative to
    its magnitude and must collapse to one branch — without it the relative gap
    test would shatter a single noisy branch into many.  A converged map orbit
    has round-off-small within-branch spread and trips the same guard, correctly
    reading a period-1 window as one branch.  Non-finite values are dropped.
    """
    s = np.asarray(col, dtype=float)
    s = np.sort(s[np.isfinite(s)])
    if s.size <= 1:
        return int(s.size)
    span = s[-1] - s[0]
    scale = max(abs(s[0]), abs(s[-1]), 1.0)
    if span <= rtol * scale:  # spread negligible vs magnitude → a single branch
        return 1
    return 1 + int(np.count_nonzero(np.diff(s) > rtol * span))


def _record_via_iterate(
    current: Any, start: Any, transient: int, n: int, idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Record one parameter value on the Rust engine, in a single ``iterate`` call.

    Resolve the initial condition explicitly and iterate ``transient + n`` steps
    in one FFI round-trip, then slice off the transient.  ``max_retries=1`` with an
    explicit ``ic`` makes a divergent value raise ``RuntimeError`` (no random-IC
    retry) — the caller turns that into an empty record set, exactly as the
    per-step path does.  Returns the recorded points and the final state (the
    carry-over initial condition for the next value).
    """
    ic_resolved = current.resolve_ic(start)
    traj = current.iterate(steps=transient + n, ic=ic_resolved, max_retries=1)
    y = traj.y
    rec = np.asarray(y[transient : transient + n][:, idx], dtype=float)
    # A zero-length run (``transient + n == 0``) yields no rows; the carry-over
    # final state is then just the resolved initial condition — matching the old
    # step-loop's ``current.state()`` after taking no steps.
    last = y[-1].copy() if y.shape[0] else np.asarray(ic_resolved, dtype=float).reshape(current.dim)
    return rec, last


def _record_via_step(
    current: Any, start: Any, transient: int, n: int, idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Record one parameter value via the per-step protocol path (a ``step()`` loop).

    The fallback for flow wrappers (``PoincareMap`` / ``StroboscopicMap``), maps
    whose ``_step`` will not lower to the engine IR, and wheel-free environments.
    Returns the recorded points and the final state.
    """
    current.reinit(start)
    for _ in range(transient):
        current.step()
    rec = np.empty((n, len(idx)))
    for i in range(n):
        rec[i] = current.step()[idx]
    return rec, current.state()


def orbit_diagram(
    system: Any,
    param: str,
    values: Any,
    *,
    n: int = 200,
    transient: int = 500,
    carry_state: bool = True,
    component: int | str | tuple[Any, ...] = 0,
    ic: Any | None = None,
    seed: int | None = None,
) -> OrbitDiagram:
    """
    Sweep a parameter and record the asymptotic orbit at each value.

    Works on anything discrete: a :class:`~tsdynamics.families.DiscreteMap`
    directly, or a flow wrapped in a
    :class:`~tsdynamics.derived.PoincareMap` /
    :class:`~tsdynamics.derived.StroboscopicMap` — in which case this *is*
    the bifurcation diagram of the flow.  ODE parameter changes reuse the
    compiled module (control parameters), so flow sweeps stay cheap; DDE
    sweeps recompile per value (their structure depends on all parameters).

    Parameters
    ----------
    system : System (discrete)
        The system to sweep.  Never mutated — each value gets a fresh
        ``with_params`` copy.
    param : str
        Parameter name to sweep.
    values : iterable of float
        Parameter values, in sweep order.
    n : int
        Points recorded per parameter value.
    transient : int
        Steps discarded before recording, at every value.
    carry_state : bool
        Start each value from the previous value's final state (follows the
        attractor branch; the classic way to draw clean diagrams).  When
        False, every value starts from ``ic`` / the system default.
    component : int, str, or tuple
        Which state component(s) to record (names allowed when the system
        declares ``variables``).
    ic : array-like, optional
        Initial state for the first value (and every value when
        ``carry_state=False``).
    seed : int, optional
        Seed for the random initial condition when the system has none; makes
        the diagram reproducible.

    Examples
    --------
    >>> od = orbit_diagram(Logistic(), "r", np.linspace(2.5, 4.0, 600), n=120)
    >>> x, y = od.flat()
    >>> # bifurcation diagram of a flow:
    >>> od = orbit_diagram(PoincareMap(Rossler(), (1, 0.0)), "c", np.linspace(2, 6, 80))
    """
    if not system.is_discrete:
        raise TypeError(
            "orbit_diagram needs a discrete-time view: a DiscreteMap, or a flow wrapped "
            "in PoincareMap / StroboscopicMap."
        )

    comp = (component,) if isinstance(component, int | str) else tuple(component)
    # Resolve names via the *instance* (not ``type(sys)``): a derived wrapper
    # exposes ``variables`` as a property, so ``type(sys).variables`` returns the
    # descriptor object (truthy) and short-circuits — breaking named components
    # over a PoincareMap/StroboscopicMap.  Instance lookup returns the ClassVar
    # for families and the resolved names for wrappers alike.
    names = getattr(system, "variables", None)
    idx: list[int] = []
    for c in comp:
        if isinstance(c, str):
            if names is None:
                raise ValueError("named components need the system to declare `variables`")
            idx.append(names.index(c))
        else:
            idx.append(int(c))

    resolved_ic = _seeded_ic(system, ic, seed)
    if resolved_ic is not None:
        ic = resolved_ic

    import warnings

    from tsdynamics.engine.compile import TapeCompileError
    from tsdynamics.engine.run import EngineNotAvailableError
    from tsdynamics.families import DiscreteMap

    values_arr = np.asarray(list(values), dtype=float)
    points: list[np.ndarray] = []
    state: np.ndarray | None = None

    # A genuine DiscreteMap iterates on the Rust engine: the whole transient +
    # record run for one parameter value becomes a single engine call (one FFI
    # round-trip) rather than ``transient + n`` Python ``step()`` calls — the
    # per-value bottleneck this stream removes (Logistic 600×120: ~9× faster).
    # Flow wrappers (PoincareMap / StroboscopicMap) have no ``iterate`` and keep
    # the per-step protocol path; a map whose ``_step`` will not lower to the IR,
    # or a wheel-free environment with no compiled engine, falls back to it too
    # (the decision is taken once and latched for the rest of the sweep).  Both
    # paths iterate the *same* lowered ``_step``, so a periodic window converges to
    # the identical cycle and — where the engine and NumPy agree bit-for-bit (e.g.
    # the logistic map) — the whole diagram is byte-identical; a chaotic window
    # records the same attractor (the engine ``iterate`` being the canonical map
    # runner).
    use_engine = isinstance(system, DiscreteMap)

    for v in values_arr:
        current = system.with_params(**{param: v})
        start = state if (carry_state and state is not None) else ic
        try:
            if use_engine:
                try:
                    rec, last = _record_via_iterate(current, start, transient, n, idx)
                except (TapeCompileError, EngineNotAvailableError):
                    # This map cannot run on the engine (a non-lowerable step, or
                    # no compiled wheel): fall back to the protocol path for this
                    # value and the rest of the sweep.
                    use_engine = False
                    rec, last = _record_via_step(current, start, transient, n, idx)
            else:
                rec, last = _record_via_step(current, start, transient, n, idx)
        except RuntimeError as exc:
            # One divergent value must not discard the whole sweep: record an
            # empty point set and restart the next value from `ic`.
            warnings.warn(
                f"orbit_diagram: {param}={v:g} diverged ({exc}); recording an "
                f"empty set for this value.",
                RuntimeWarning,
                stacklevel=2,
            )
            points.append(np.empty((0, len(idx))))
            state = None
            continue
        points.append(rec)
        if carry_state:
            state = last

    meta = {
        "system": type(system).__name__,
        "param": param,
        "n": n,
        "transient": transient,
        "carry_state": carry_state,
        "components": tuple(idx),
    }
    return OrbitDiagram(
        param=param, values=values_arr, points=points, components=tuple(idx), meta=meta
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
