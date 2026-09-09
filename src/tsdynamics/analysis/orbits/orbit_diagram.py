"""Orbit diagrams — asymptotic states swept across a parameter."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .._result import AnalysisResult
from .poincare import _seeded_ic

__all__ = ["OrbitDiagram", "bifurcation_diagram", "orbit_diagram"]

#: Sampling step for the flow path's integration (time units).  Only the *output*
#: grid — the adaptive solver picks its own internal step (see CLAUDE.md, "Dense
#: output and ``max_step``") — so this bounds how finely a peak is resolved before
#: the parabolic sharpening below, not the accuracy of the orbit itself.
_FLOW_DT = 0.01

#: First integration horizon tried per parameter value on the flow path, and the
#: ceiling it doubles up to before giving up on collecting ``transient + n`` peaks.
_FLOW_FINAL_TIME = 100.0
_FLOW_MAX_TIME = 1.0e4

#: Relative spread below which the tail of a flow is read as *converged to an
#: equilibrium* — the asymptotic "orbit" is then a single point, which is the
#: correct entry for the fixed-point branch of a bifurcation diagram.
_STATIONARY_RTOL = 1.0e-6


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
        gap exceeds ``rtol`` times the orbit's range.  A finite period ``p >= 2``
        is only reported when the recorded iterate sequence actually *revisits*
        its values cyclically (``v[i] ≈ v[i + p]`` to ``rtol``); a chaotic band
        whose finite-sample points merely cluster into ``p`` bins fails this
        repeat test and is reported as aperiodic (``0``).

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
            col = pts[:, component]
            p = _count_branches(col, rtol)
            if p > max_period or (p >= 2 and not _is_cyclic(col, p, rtol)):
                # Either too many branches to resolve as a cycle, or the branch
                # count is a finite-sample clustering of a chaotic band that does
                # not actually revisit its values cyclically — aperiodic.
                out[k] = 0
            else:
                out[k] = p
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

        Parameters
        ----------
        component : int, default 0
            Which recorded component to count branches in.
        max_period : int, default 16
            Periods above this are treated as aperiodic when detecting changes.
        rtol : float, default 0.01
            Relative gap separating branches in :meth:`periods`.

        Returns
        -------
        numpy.ndarray of float
            Estimated bifurcation parameter values, in sweep order.  Their
            resolution is the spacing of ``values``.

        References
        ----------
        Feigenbaum, M. J. (1978). Quantitative universality for a class of
        nonlinear transformations. *Journal of Statistical Physics*, 19, 25--52.
        """
        p = self.periods(component=component, max_period=max_period, rtol=rtol)
        return self._bifurcation_points_from_periods(p)

    def _bifurcation_points_from_periods(self, periods: np.ndarray) -> np.ndarray:
        """Midpoints between consecutive values whose ``periods`` differ (no reseed).

        The core of :meth:`bifurcation_points`, factored so a caller that has
        already computed ``periods`` (e.g. :meth:`to_plot_spec`) reuses it instead
        of recomputing the period sweep.  Transitions touching a diverged value
        (``-1``) are skipped.
        """
        changed = (periods[:-1] != periods[1:]) & (periods[:-1] != -1) & (periods[1:] != -1)
        (i,) = np.nonzero(changed)
        return cast(np.ndarray, 0.5 * (self.values[i] + self.values[i + 1]))

    def to_plot_spec(self, kind: str | None = None, *, annotate: bool = False) -> Any:
        """Describe this orbit diagram as a backend-agnostic :class:`PlotSpec`.

        Builds an ``ORBIT_DIAGRAM`` scatter of the asymptotic state (first
        recorded component) against the swept parameter — the classic
        bifurcation diagram — via :meth:`flat`.

        **The default plot is clean** (just the scatter of asymptotic states, the
        textbook bifurcation picture).  A chaotic period-doubling cascade contains
        *dozens* of onsets, so drawing them all as labelled vertical reference
        lines smears the figure into an illegible pile of overlapping text.  Pass
        ``annotate=True`` to overlay the :meth:`bifurcation_points` onsets as
        ``"vline"`` :class:`~tsdynamics.viz.spec.Annotation` reference lines (each
        labelled with the period it opens onto) — best kept for a short, low-period
        sweep where the labels don't collide.  The :mod:`tsdynamics.viz.spec`
        import is lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"bifurcation"``).  ``None`` uses
            ``ORBIT_DIAGRAM``.
        annotate : bool, default False
            Overlay the detected period-doubling onsets as labelled vertical
            reference lines.  Off by default so the default ``.plot()`` is a clean
            bifurcation scatter; the :meth:`periods` / :meth:`bifurcation_points`
            quantifiers are unaffected either way.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        x, y = self.flat()
        annotations: list[Any] = []
        if annotate and len(self.values) > 1:
            # Compute the period sweep once and feed it to *both* the onset
            # detection and the per-line period label (instead of recomputing
            # ``periods()`` inside ``bifurcation_points()`` and again here).
            periods = self.periods()
            onsets = self._bifurcation_points_from_periods(periods)
            for onset in np.asarray(onsets, dtype=float).ravel():
                # Label the line with the period the cascade opens *onto* (the
                # period just to the right of the onset).
                j = int(np.searchsorted(self.values, onset))
                p = int(periods[j]) if 0 <= j < periods.size else 0
                label = f"period {p}" if p > 0 else "bifurcation"
                annotations.append(pb.vline(float(onset), text=label))
        # Name the discrete view on the figure itself: a section always has to be
        # chosen, and a diagram that does not say which one it used is not
        # reproducible from the picture.
        section = self.meta.get("section")
        return pb.spec(
            kind,
            "orbit_diagram",
            layers=[pb.scatter(x, y, style={"markersize": 1.0})],
            xlabel=self.param,
            ylabel=str(section) if section else "asymptotic state",
            title=f"bifurcation diagram — {section}" if section else "bifurcation diagram",
            annotations=annotations,
            meta=self.meta,
        )

    def __repr__(self) -> str:
        # Columns are ragged on the flow path — a value that settled on an
        # equilibrium records one point where a chaotic one records ``n`` — so
        # quoting the FIRST column's size described the whole diagram as
        # "1 points/value" when 39 000 points were in it.  Show the range.
        sizes = [int(p.shape[0]) for p in self.points]
        lo, hi = (min(sizes), max(sizes)) if sizes else (0, 0)
        per = f"{lo}" if lo == hi else f"{lo}-{hi}"
        return f"OrbitDiagram({self.param!r}, {len(self.values)} values, {per} points/value)"


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

    The guard's ``scale`` is the *centered* dispersion — ``max(|s − mean|, 1.0)``,
    measured about the orbit's mean rather than its raw endpoints — so the
    collapse threshold tracks the within-orbit spread (a noise floor) and not the
    orbit's DC offset.  Anchoring on the raw magnitude conflated the offset scale
    with the noise scale, collapsing a genuine multi-branch orbit that merely
    lives far from the origin (e.g. branches at ``{100.0, 100.5}`` read as one
    branch); centering decouples the two.  The ``1.0`` floor preserves the
    integration-noise collapse for orbits near the origin (a tiny spread on a
    period-1 flow branch still collapses unchanged).
    """
    s = np.asarray(col, dtype=float)
    s = np.sort(s[np.isfinite(s)])
    if s.size <= 1:
        return int(s.size)
    span = s[-1] - s[0]
    # Centered dispersion (offset-free): the noise floor scales with the orbit's
    # spread about its mean, not with its distance from the origin.
    mean = float(s.mean())
    scale = max(abs(s[0] - mean), abs(s[-1] - mean), 1.0)
    if span <= rtol * scale:  # spread negligible vs dispersion → a single branch
        return 1
    return 1 + int(np.count_nonzero(np.diff(s) > rtol * span))


def _is_cyclic(col: np.ndarray, p: int, rtol: float) -> bool:
    """
    Whether the iterate sequence ``col`` revisits its values with period ``p``.

    A genuine period-``p`` window obeys ``v[i] ≈ v[i + p]`` for every recorded
    sample (the orbit cyclically returns to the same ``p`` values), to a
    tolerance of ``rtol`` times the orbit's range.  A chaotic band whose
    finite-sample points happen to cluster into ``p`` gap-separated bins violates
    this — its successive iterates wander within the band rather than repeating —
    so the cyclic test distinguishes a true periodic window from a spurious
    finite period read off the cluster count alone.  Non-finite values make the
    sequence non-cyclic (a diverged/NaN run is not a clean cycle).
    """
    v = np.asarray(col, dtype=float)
    if v.shape[0] <= p:
        return False
    finite = v[np.isfinite(v)]
    if finite.size != v.size or finite.size == 0:
        return False
    tol = rtol * max(float(finite.max() - finite.min()), 1.0)
    return bool(np.all(np.abs(v[:-p] - v[p:]) <= tol))


def _sweep_via_kernel(
    system: Any,
    param: str,
    values_arr: np.ndarray,
    *,
    transient: int,
    n: int,
    carry_state: bool,
    ic: Any,
    idx: list[int],
) -> list[np.ndarray]:
    """Run the *whole* map sweep in one engine call (stream perf/param-sweep-kernel).

    Lowers the map once keeping ``param`` as the tape's single runtime ``Param``
    (:func:`tsdynamics.engine.compile.lower_map_sweep_cached`), then drives the
    Rust sweep kernel (:func:`tsdynamics.engine.run.map_param_sweep`) over every
    value — one FFI round-trip for the entire diagram instead of one ``iterate``
    call per value (the WS-MAPITER path).  The per-iterate numerics are
    byte-for-byte the per-value ``iterate`` path, so the diagram is byte-identical
    where the engine and NumPy agree bit-for-bit (the logistic map) and the same
    attractor for a chaotic map.

    Returns the per-value ``points`` list (an empty ``(0, k)`` array for a value
    that diverged, with a :class:`RuntimeWarning` per such value — exactly the
    per-value path's contract).

    Raises
    ------
    NotImplementedError, BackendError
        If the map's ``_step`` will not lower (``TapeCompileError``) or the
        compiled engine is unavailable (``EngineNotAvailableError``).  The caller
        catches these public bases and falls back to the per-value/per-step path.
    """
    import warnings

    from tsdynamics.engine.compile import lower_map_sweep_cached
    from tsdynamics.engine.run import map_param_sweep

    tape = lower_map_sweep_cached(system, param)
    # The sweep tape has exactly the swept parameter as its single runtime input
    # (control_names == [param]); the base vector's one slot is overwritten per
    # value by the kernel, so its initial value is irrelevant.
    base_params = np.zeros(1, dtype=np.float64)
    ic_resolved = np.asarray(system.resolve_ic(ic), dtype=np.float64).reshape(system.dim)
    components = np.asarray(idx, dtype=np.int64)

    points_flat, status = map_param_sweep(
        tape.to_arrays(),
        base_params,
        0,
        values_arr,
        ic_resolved,
        components,
        transient,
        n,
        carry_state=carry_state,
        jit=False,
    )
    # ``points_flat`` is (n_values * n, k); split into one (n, k) block per value,
    # dropping a diverged value's (zero) block to an empty set + warning — exactly
    # the per-value path's divergence contract.
    points: list[np.ndarray] = []
    block = points_flat.reshape(len(values_arr), n, len(idx))
    for k, v in enumerate(values_arr):
        if status[k] != 0:
            warnings.warn(
                f"orbit_diagram: {param}={v:g} diverged; recording an empty set for this value.",
                RuntimeWarning,
                stacklevel=3,
            )
            points.append(np.empty((0, len(idx))))
        else:
            points.append(np.array(block[k], dtype=float))
    return points


def _record_via_step(
    current: Any, start: Any, transient: int, n: int, idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Record one parameter value via the per-step protocol path (a ``step()`` loop).

    The fallback for flow wrappers (``StroboscopicMap``, a degenerate ``n == 0``
    ``PoincareMap``), maps whose ``_step`` will not lower to the engine IR, and
    wheel-free environments.  Returns the recorded points and the final state.
    """
    current.reinit(start)
    for _ in range(transient):
        current.step()
    rec = np.empty((n, len(idx)))
    for i in range(n):
        rec[i] = current.step()[idx]
    return rec, current.state()


def _record_via_trajectory(
    current: Any, start: Any, transient: int, n: int, idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Record one ``PoincareMap`` value through :meth:`PoincareMap.trajectory`.

    ``trajectory`` collects the whole section in **one** engine call (the wired
    Rust event march, stream WS-CROSSKERNEL) rather than re-entering the flow
    integrator once per detection ``dt``, which is what the ``step()`` loop above
    does — the documented "``orbit_diagram`` over a ``PoincareMap`` is *not*
    accelerated" gap.  Measured on a Rössler ``c``-sweep (40 values, 80 crossings
    each) this is **75x** faster: 64.8 s -> 0.86 s, 1620 -> 21.5 ms per value.

    The semantics map exactly: ``trajectory(n, transient=transient)`` discards
    ``transient`` crossings and returns the next ``n``, which is precisely what the
    ``step()`` loop records, and it leaves ``state()`` at the last collected
    crossing — the same value the loop carries into the next parameter value.
    ``trajectory`` itself decides whether the engine march applies (a DDE, a stiff
    ``_default_method``, ``backend="reference"``, or an absent wheel keep the
    Python loop), and its Python fallback is the *same* ``_advance_to_crossing``
    the ``step()`` loop drives — byte-identical.  So this needs no eligibility
    check of its own.

    Where the engine march *does* apply it is faster but **not** identical: it is
    the fixed-step ``rk4`` kernel at the detection ``dt``, where the ``step()``
    loop drove the flow's adaptive default (see the ``orbit_diagram`` Notes).  Two
    consequences a caller should know about:

    * **Accuracy.**  Measured on a Rössler period-1 window at the default
      ``dt=0.01``, the recorded crossing carries ~1.2e-8 absolute error against a
      converged reference where the adaptive loop carried ~1.6e-10; both converge
      as ``O(dt⁴)``, so ``PoincareMap(..., dt=...)`` buys the difference back.
    * **Stiffness.**  ``rk4`` has a bounded stability region (``|λh| ≲ 2.785``), so
      a parameter value that makes the flow stiff *relative to* ``dt`` blows the
      march up where the adaptive loop simply shrank its step.  On the Rössler
      ``c``-sweep at ``dt=0.01`` that threshold is ``c ≈ 200``.  The failure is
      loud, not silent — the resulting :class:`ConvergenceError` is caught by the
      sweep, which records an empty point set and warns for that value.
    """
    current.reinit(start)
    section = current.trajectory(n, transient=transient)
    return np.asarray(section.y, dtype=float)[:, idx], current.state()


def _peak_records(y: np.ndarray, primary: int, comps: list[int]) -> np.ndarray:
    """Record every strict local **maximum** of component ``primary`` of a flow.

    Successive maxima of one coordinate are the classic discrete view of a flow —
    the *next-amplitude* (peak) map, the section Lorenz (1963) used to expose the
    one-dimensional dynamics inside the attractor, and the view every textbook
    bifurcation diagram of a flow is drawn from.  Unlike a fixed Poincaré plane it
    needs no parameter-dependent constant, so it keeps crossing as the attractor
    moves across the sweep.

    A sample ``i`` is a maximum when ``v[i-1] < v[i] > v[i+1]``.  Every recorded
    component is then read at the **same** refined position: the vertex of the
    parabola through the three samples of ``primary``, evaluated on each
    component's own quadratic through the same three samples.  For ``primary``
    itself that is exactly the peak-interpolation formula
    :func:`~tsdynamics.analysis.orbits.return_map._local_extrema` uses, so a
    coarse output grid still yields a sub-sample-accurate branch.
    """
    v = np.asarray(y[:, primary], dtype=float)
    if v.size < 3:
        return np.empty((0, len(comps)))
    interior = v[1:-1]
    k = np.nonzero((interior > v[:-2]) & (interior > v[2:]))[0] + 1
    if k.size == 0:
        return np.empty((0, len(comps)))
    ym, y0, yp = v[k - 1], v[k], v[k + 1]
    denom = ym - 2.0 * y0 + yp
    d = np.where(denom != 0.0, 0.5 * (ym - yp) / denom, 0.0)
    d = np.clip(d, -0.5, 0.5)
    out = np.empty((k.size, len(comps)))
    for j, c in enumerate(comps):
        a, b, cc = y[k - 1, c], y[k, c], y[k + 1, c]
        out[:, j] = b + 0.5 * d * (cc - a) + 0.5 * d * d * (a - 2.0 * b + cc)
    return out


def _is_stationary(y: np.ndarray) -> bool:
    """Whether the tail of ``y`` has stopped moving (converged to an equilibrium).

    Compares the spread of the last tenth of the run against the spread of the
    whole run: a trajectory that has settled onto a fixed point has a tail spread
    that is a vanishing fraction of the transient it came in on.  The ``1.0``
    floor keeps a run that never moved at all (started *at* the equilibrium) on
    the stationary side.
    """
    if y.shape[0] < 10:
        return False
    tail = y[-max(2, y.shape[0] // 10) :]
    spread = float(np.max(tail, axis=0).max() - np.min(tail, axis=0).min())
    scale = max(float(np.max(y, axis=0).max() - np.min(y, axis=0).min()), 1.0)
    return bool(np.isfinite(spread) and spread <= _STATIONARY_RTOL * scale)


def _short_column(
    rec: np.ndarray, transient: int, n: int, idx: list[int], max_time: float, dim: int
) -> np.ndarray:
    """Return the column for a flow value that ran out of ``max_time``, and warn.

    Three outcomes, each with a message naming the numbers the caller passed and
    the line that fixes it:

    * enough peaks survived the transient — record them (a partial column);
    * fewer peaks than ``transient`` — record the **last** ones anyway rather
      than nothing, and say that the transient was not fully discarded;
    * no peaks at all — the recorded component never turns over, so the peak map
      is the wrong view of this flow: name a section or another component.
    """
    import warnings

    found = int(rec.shape[0])
    if found > transient:
        warnings.warn(
            f"bifurcation_diagram: only {found - transient} of {n} peaks were found within "
            f"max_time={max_time:g}, so this value's column is short. To fill it:\n"
            f"    ts.bifurcation_diagram(system, param, values, max_time={max_time * 10:g})",
            RuntimeWarning,
            stacklevel=4,
        )
        return rec[transient:]
    if found:
        keep = rec[-min(n, found) :]
        warnings.warn(
            f"bifurcation_diagram: only {found} peaks were found within "
            f"max_time={max_time:g}, fewer than transient={transient}, so this value's column "
            f"is the last {keep.shape[0]} of them and its transient is NOT fully discarded "
            "(transient/n count peaks here, not iterates — a slow oscillator makes far fewer "
            "of them than a map does). Either:\n"
            f"    ts.bifurcation_diagram(system, param, values, transient={max(found // 4, 1)},"
            f" n={max(found // 2, 1)})\n"
            f"    ts.bifurcation_diagram(system, param, values, max_time={max_time * 10:g})",
            RuntimeWarning,
            stacklevel=4,
        )
        return keep
    # Only offer ``component=`` when the state actually has another component to
    # offer: a scalar flow (a 1-D DDE) would be told to type an index that does
    # not exist, which is worse than no suggestion at all.
    other = next((c for c in range(dim) if c != idx[0]), None)
    lines = ["    ts.bifurcation_diagram(system, param, values, section=('z', 27.0, 'up'))"]
    if other is not None:
        lines.append(f"    ts.bifurcation_diagram(system, param, values, component={other})")
    warnings.warn(
        f"bifurcation_diagram: component {idx[0]} of this flow has no maximum within "
        f"max_time={max_time:g}, so the successive-maxima view records nothing for this value "
        "(it is monotone, or already at rest but still drifting). Read the flow through a "
        "section, or through a component that oscillates:\n" + "\n".join(lines),
        RuntimeWarning,
        stacklevel=4,
    )
    return np.empty((0, len(idx)))


def _record_via_peaks(
    current: Any,
    start: Any,
    transient: int,
    n: int,
    idx: list[int],
    *,
    dt: float,
    final_time: float,
    max_time: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Record one parameter value of a **flow** as successive maxima (the peak map).

    Integrates, collects ``transient + n`` maxima of the first recorded component
    and returns the last ``n`` of them.  The horizon doubles (up to ``max_time``)
    until enough peaks are found; the horizon that worked is returned so the next
    parameter value starts from it instead of re-discovering it.

    A flow that has **converged to an equilibrium** produces no peaks at all — its
    asymptotic orbit is one point, which is exactly what the fixed-point branch of
    a bifurcation diagram should show, so that single state is recorded rather
    than an empty column.

    **When ``max_time`` runs out** the column is whatever the flow did produce,
    never silently nothing.  ``transient + n`` counts *peaks*, and the defaults
    are sized for a map's cheap iterates: a slow oscillator (or a DDE, whose
    peaks are a delay apart) can easily produce fewer than ``transient`` peaks in
    ``max_time``, and dropping the first ``transient`` of those then leaves an
    empty set — a blank figure whose only clue was one ``RuntimeWarning``.  So a
    short run keeps its **last** peaks (the most asymptotic ones available) and
    says so, with the numbers the caller actually passed.  A run with *no* peaks
    at all is a different story — the recorded component never turns over — and
    that one says to name a section or another component.
    """
    need = transient + n
    horizon = float(final_time)
    while True:
        traj = current.integrate(final_time=horizon, dt=dt, ic=start)
        y = np.asarray(traj.y, dtype=float)
        if _is_stationary(y):
            # Settled on an equilibrium: the asymptotic set is the single state.
            return y[-1][idx][None, :], y[-1], horizon
        rec = _peak_records(y, idx[0], idx)
        if rec.shape[0] >= need:
            return rec[transient:need], y[-1], horizon
        if horizon >= max_time:
            return _short_column(rec, transient, n, idx, max_time, y.shape[1]), y[-1], horizon
        # Grow the horizon by the *measured* peak rate rather than blindly
        # doubling: one short pilot run tells us how much time a peak costs, so
        # the second attempt already lands (and the horizon that worked is
        # returned, so the rest of the sweep pays for exactly one integration).
        grow = (need + 5) / max(rec.shape[0], 1) if rec.shape[0] else 4.0
        horizon = min(horizon * max(grow, 1.5), max_time)


def _discrete_view(system: Any, section: Any, component_label: str, param: str) -> tuple[Any, str]:
    """Resolve ``system`` to a discrete-time view, returning ``(view, description)``.

    A genuine discrete view passes straight through.  A **flow** is turned into
    one — the whole point of :func:`bifurcation_diagram`, whose canonical use is a
    flow — either through the section the caller asked for or through the peak map
    (signalled by a ``None`` view, which routes to :func:`_record_via_peaks`).
    The description is recorded in the result's ``meta["section"]`` and printed on
    the figure, so the choice is never made silently.
    """
    from tsdynamics.derived.poincare import PoincareMap
    from tsdynamics.derived.stroboscopic import StroboscopicMap
    from tsdynamics.errors import InvalidInputError, InvalidParameterError

    if not hasattr(system, "is_discrete"):
        raise InvalidInputError(
            f"bifurcation_diagram needs a dynamical system as its first argument, got "
            f"{type(system).__name__}. Pass a system (or a discrete view of one), e.g.\n"
            "    ts.bifurcation_diagram(ts.systems.Lorenz(), 'rho', "
            "np.linspace(0.0, 50.0, 200))"
        )
    if system.is_discrete:
        if section is not None:
            raise InvalidParameterError(
                f"section= chooses how to slice a *flow*, but {type(system).__name__} is "
                "already a discrete-time view, so there is nothing to slice. Drop section=:\n"
                f"    ts.bifurcation_diagram(system, {param!r}, values)"
            )
        if isinstance(system, PoincareMap):
            return system, f"Poincaré section {system.plane}"
        if isinstance(system, StroboscopicMap):
            return system, "stroboscopic sampling"
        return system, "map iterates"

    from tsdynamics.families import StochasticSystem

    if isinstance(system, StochasticSystem):
        raise InvalidInputError(
            f"bifurcation_diagram needs a deterministic system: {type(system).__name__} is "
            "stochastic (an SDE), so its 'asymptotic orbit' is a different sample path on "
            "every run. Sweep a deterministic model instead, e.g.\n"
            "    ts.bifurcation_diagram(ts.systems.Lorenz(), 'rho', "
            "np.linspace(0.0, 50.0, 200))"
        )
    if section is not None:
        return PoincareMap(system, section), f"Poincaré section {section}"
    return None, f"successive maxima of {component_label}"


def bifurcation_diagram(
    system: Any,
    param: str,
    values: Any,
    *,
    n: int = 200,
    transient: int = 500,
    carry_state: bool = True,
    component: int | str | tuple[Any, ...] = 0,
    section: Any | None = None,
    ic: Any | None = None,
    seed: int | None = None,
    dt: float = _FLOW_DT,
    max_time: float = _FLOW_MAX_TIME,
) -> OrbitDiagram:
    """
    Sweep a parameter and record the asymptotic orbit at each value.

    Pass **any** system — this is the one-liner::

        ts.bifurcation_diagram(ts.systems.Lorenz(), "rho", np.linspace(0.0, 50.0, 200))

    A :class:`~tsdynamics.families.DiscreteMap` is swept directly.  A **flow** is
    reduced to a discrete view automatically, because a bifurcation diagram of a
    flow is the common case and a section always has to be chosen:

    - by default, **successive maxima of the recorded component** — the
      next-amplitude (peak) map, the view Lorenz (1963) used and the one every
      textbook bifurcation diagram of a flow is drawn from.  It needs no
      parameter-dependent constant, so it keeps producing points as the attractor
      moves across the sweep, and a value whose flow has settled on an
      **equilibrium** records that single state (the fixed-point branch);
    - or the section you name with ``section=`` (any
      :class:`~tsdynamics.derived.PoincareMap` ``plane`` spelling, e.g.
      ``section=("z", 27.0, "up")``).

    **The choice is never silent**: it is recorded in ``meta["section"]`` and
    printed under the figure's title.  A flow already wrapped in a
    :class:`~tsdynamics.derived.PoincareMap` /
    :class:`~tsdynamics.derived.StroboscopicMap` is used as given.

    ``orbit_diagram`` is the same function under its other name (the map-centric
    spelling); both are exported, and both accept everything described here.

    ODE parameter changes reuse the cached lowered tape (control parameters), so
    flow sweeps stay cheap; DDE sweeps re-lower per value (their structure depends
    on all parameters).

    Notes
    -----
    **How each value is run.**  A genuine :class:`~tsdynamics.families.DiscreteMap`
    sweeps the *whole* parameter array in one Rust kernel call.  A
    :class:`~tsdynamics.derived.PoincareMap` collects each value's section in one
    engine call through :meth:`~tsdynamics.derived.PoincareMap.trajectory` (the
    wired Rust event march) — ~70x faster than the per-``dt`` ``step()`` loop it
    replaces.  Everything else (``StroboscopicMap``, a map whose ``_step`` will not
    lower, a wheel-free environment) drives the per-step protocol loop.

    The Poincaré fast path marches the fixed-step ``rk4`` kernel at the map's
    detection ``dt`` (see :mod:`tsdynamics.derived._crossings`), which is the same
    discretisation :func:`~tsdynamics.analysis.poincare_section` and
    :func:`~tsdynamics.analysis.return_map` already use — so an orbit diagram and
    a section of the same flow are now consistent with each other.  It is a
    *different* discretisation from the flow's adaptive default that the old
    ``step()`` loop used.  Measured over a 40-value Rössler ``c``-cascade at the
    default ``dt=0.01``: in a periodic window the branch values shift by ~5e-8
    (up to ~3e-3 for a value sitting right at a window edge, where a tiny shift
    in the orbit is amplified), and on a chaotic band the two are the same
    attractor rather than the same points (support bounds agree to ~6e-2).  The
    detected branch structure — ``OrbitDiagram.periods()`` and
    ``bifurcation_points()`` — is unchanged, at both a short and a long
    ``transient``.

    ``rk4`` is fixed-step, so its stability region is bounded (``|λh| ≲ 2.785``).
    A sweep that runs into a regime where the flow is stiff *relative to* the
    detection ``dt`` therefore records an empty point set with a
    ``RuntimeWarning`` for those values, where the old adaptive ``step()`` loop
    would have shrunk its step and carried on (on the Rössler ``c``-sweep at
    ``dt=0.01`` this starts at ``c ≈ 200``).  Pass a smaller ``dt`` to
    :class:`~tsdynamics.derived.PoincareMap` when sweeping into a fast-timescale
    regime.

    **The flow path.**  With no ``section=``, each parameter value is integrated
    (output step ``dt``, horizon grown from 100 time units up to ``max_time``)
    until ``transient + n`` maxima of the recorded component have been collected;
    the last ``n`` are the column.  ``transient`` and ``n`` therefore count
    **peaks**, exactly as they count iterates for a map and crossings for a
    section.  Each peak is sharpened by the same parabolic interpolation
    :func:`~tsdynamics.analysis.return_map` uses, so a coarse ``dt`` still gives a
    crisp period-1 branch.

    Parameters
    ----------
    system : System
        The system to sweep — a map, a flow, or a discrete view of a flow.  Never
        mutated: each value gets a fresh ``with_params`` copy.
    param : str
        Parameter name to sweep.
    values : iterable of float
        Parameter values, in sweep order.
    n : int
        Points recorded per parameter value (map iterates / section crossings /
        peaks, according to the view).
    transient : int
        Points discarded before recording, at every value.
    carry_state : bool
        Start each value from the previous value's final state (follows the
        attractor branch; the classic way to draw clean diagrams).  When
        False, every value starts from ``ic`` / the system default.
    component : int, str, or tuple
        Which state component(s) to record (names allowed when the system
        declares ``variables``).  On the flow path the **first** of them also
        defines the peak map.
    section : tuple, optional
        Slice a flow with this Poincaré section instead of the default peak map.
        Any :class:`~tsdynamics.derived.PoincareMap` ``plane`` spelling:
        ``("z", 27.0, "up")``, ``("y", 0.0)``, or ``(normal, offset)``.  Invalid
        for a system that is already discrete.
    ic : array-like, optional
        Initial state for the first value (and every value when
        ``carry_state=False``).
    seed : int, optional
        Seed for the random initial condition when the system has none; makes
        the diagram reproducible.
    dt : float, default 0.01
        Output sampling step of the flow path's integration (time units).  Ignored
        for a map / an already-discrete view.
    max_time : float, default 1e4
        Ceiling on the flow path's integration horizon per parameter value.

    Returns
    -------
    OrbitDiagram
        The swept ``values`` and the per-value recorded ``points``.  ``meta``
        records the discrete view that was used under ``"section"``.  A value
        whose orbit diverged carries an empty point set (and emits a
        :class:`RuntimeWarning`).

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If ``system`` is not a dynamical system, or is stochastic (an SDE has no
        single asymptotic orbit).
    tsdynamics.errors.InvalidParameterError
        If ``section=`` is given for a system that is already discrete.
    ValueError
        If a named ``component`` is requested but the system does not declare
        ``variables``.

    Warns
    -----
    RuntimeWarning
        When a parameter value diverges (that value records an empty set and the
        sweep continues), or when the flow path exhausts ``max_time`` before
        collecting ``n`` peaks.

    References
    ----------
    May, R. M. (1976). Simple mathematical models with very complicated
    dynamics. *Nature*, 261, 459--467.

    Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the
    Atmospheric Sciences*, 20, 130--141.  (Successive maxima as the discrete view
    of a flow.)

    Examples
    --------
    >>> # a flow, straight up — the section is chosen for you and reported:
    >>> od = bifurcation_diagram(Lorenz(), "rho", np.linspace(0.0, 50.0, 200))
    >>> od.meta["section"]
    'successive maxima of x'
    >>> # ... or name the section yourself:
    >>> od = bifurcation_diagram(Rossler(), "c", np.linspace(2, 6, 80),
    ...                          section=("y", 0.0, "up"))
    >>> # a map:
    >>> od = bifurcation_diagram(Logistic(), "r", np.linspace(2.5, 4.0, 600), n=120)
    >>> x, y = od.flat()
    """
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

    label = names[idx[0]] if names is not None else f"component {idx[0]}"
    # Reduce whatever was passed to a discrete-time view.  A flow with no explicit
    # ``section=`` comes back as ``None`` — the peak map, which is not a wrapper
    # object but a way of *reading* the flow, so it is driven directly below.
    resolved, section_label = _discrete_view(system, section, label, param)
    is_peak_map = resolved is None
    view = system if is_peak_map else resolved

    resolved_ic = _seeded_ic(system, ic, seed)
    if resolved_ic is not None:
        ic = resolved_ic

    import warnings

    from tsdynamics.derived.poincare import PoincareMap
    from tsdynamics.errors import BackendError
    from tsdynamics.families import DiscreteMap

    values_arr = np.asarray(list(values), dtype=float)
    points: list[np.ndarray] = []
    state: np.ndarray | None = None
    horizon = _FLOW_FINAL_TIME

    def _meta() -> dict[str, Any]:
        return {
            "system": type(system).__name__,
            "param": param,
            "n": n,
            "transient": transient,
            "carry_state": carry_state,
            "components": tuple(idx),
            # The discrete view this diagram was read through — recorded so an
            # auto-chosen section is never a silent choice (it is also printed
            # under the figure's title by ``to_plot_spec``).
            "section": section_label,
            "section_auto": is_peak_map,
        }

    # A genuine DiscreteMap sweeps the WHOLE parameter array in a single engine
    # call (stream perf/param-sweep-kernel): the map is lowered once keeping the
    # swept parameter as the tape's single runtime input, and the Rust kernel
    # varies it per value — one FFI round-trip for the entire diagram, instead of
    # the WS-MAPITER path's one ``iterate`` call per value (a 1000-value sweep was
    # ~1000 round-trips; ~410 ms → a few ms).  The per-iterate numerics are
    # byte-for-byte the per-value ``iterate`` path, so the diagram is
    # byte-identical where the engine and NumPy agree bit-for-bit (the logistic
    # map) and the same attractor for a chaotic map.  Flow wrappers (PoincareMap /
    # StroboscopicMap) have no ``_step`` to lower; a map whose ``_step`` will not
    # lower to the IR (``TapeCompileError`` → ``NotImplementedError``) or a
    # wheel-free environment (``EngineNotAvailableError`` → ``BackendError``) fall
    # back to the per-value/per-step protocol loop below — the same answer.
    if isinstance(view, DiscreteMap):
        try:
            points = _sweep_via_kernel(
                view,
                param,
                values_arr,
                transient=transient,
                n=n,
                carry_state=carry_state,
                ic=ic,
                idx=idx,
            )
            return OrbitDiagram(
                param=param,
                values=values_arr,
                points=points,
                components=tuple(idx),
                meta=_meta(),
            )
        except (NotImplementedError, BackendError):
            # The map cannot run on the engine sweep (a non-lowerable ``_step`` or
            # no compiled wheel) — catch the PUBLIC bases (not the engine-internal
            # leaf types) and fall back to the per-value/per-step loop below.
            points = []

    # A ``PoincareMap`` collects each value's section in ONE engine call through
    # ``trajectory`` (the wired Rust event march) instead of re-entering the flow
    # integrator per detection ``dt`` — the flagship "bifurcation diagram of a flow"
    # path, previously ~70x slower than necessary (the gap CLAUDE.md and
    # ``derived/_crossings`` both call out).  ``trajectory`` owns the eligibility
    # decision and falls back to the very ``_advance_to_crossing`` loop
    # ``_record_via_step`` drives, so the answer is unchanged where the fast path
    # declines.  A degenerate ``n == 0`` keeps the step loop: ``trajectory`` records
    # nothing, so it cannot leave ``state()`` at the last *discarded* transient
    # crossing the way the step loop does, and ``carry_state`` would drift.
    use_trajectory = isinstance(view, PoincareMap) and n > 0

    # The per-value protocol path: flow wrappers, the raw-flow peak map, and the
    # engine-sweep fallback.
    for v in values_arr:
        current = view.with_params(**{param: v})
        start = state if (carry_state and state is not None) else ic
        try:
            if is_peak_map:
                # A raw flow: read it as the successive-maxima (peak) map.  The
                # horizon that satisfied the previous value seeds the next one, so
                # only the first value pays for discovering it.
                rec, last, horizon = _record_via_peaks(
                    current, start, transient, n, idx, dt=dt, final_time=horizon, max_time=max_time
                )
            else:
                # Flow wrappers (PoincareMap / StroboscopicMap) and the
                # engine-sweep fallback (a non-lowerable map / wheel-free env)
                # drive the per-step protocol loop — byte-identical to the engine
                # path on a lowerable map.
                record = _record_via_trajectory if use_trajectory else _record_via_step
                rec, last = record(current, start, transient, n, idx)
        except RuntimeError as exc:
            # One divergent value must not discard the whole sweep: record an
            # empty point set and restart the next value from `ic`.
            warnings.warn(
                f"bifurcation_diagram: {param}={v:g} diverged ({exc}); recording an "
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

    return OrbitDiagram(
        param=param, values=values_arr, points=points, components=tuple(idx), meta=_meta()
    )


#: The map-centric spelling of the same sweep.  One implementation, two names —
#: so no error message can ever name a function the caller did not type, and
#: ``ts.orbit_diagram`` / ``ts.bifurcation_diagram`` cannot drift apart.
orbit_diagram = bifurcation_diagram


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
