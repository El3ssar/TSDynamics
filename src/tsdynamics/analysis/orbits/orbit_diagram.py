"""Orbit diagrams — asymptotic states swept across a parameter."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .._common import reject_data
from .._result import AnalysisResult
from .._result_json import _sig
from .poincare import _seeded_ic

__all__ = ["OrbitDiagram", "orbit_diagram"]

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

#: Above this many recorded points the repr stops re-deriving the cascade
#: summary.  ``periods()`` clusters every column, so it is O(total points), and a
#: repr is typed at a prompt.  Measured: 2 400 points -> 1.7 ms, 40 000 -> 6.1 ms,
#: 100 000 -> 15.1 ms, 200 000 -> 29.5 ms, so the cap buys a worst case of ~30 ms
#: — under the threshold where a prompt feels slow.  A publication sweep above it
#: simply omits the two summary lines; ``periods()`` / ``bifurcation_points()``
#: are still there to call.
_REPR_MAX_POINTS = 200_000

#: Bifurcation values shown in the repr before it elides with ``…``.
_REPR_MAX_BIFURCATIONS = 4


@dataclass(frozen=True)
class OrbitDiagram(AnalysisResult):
    """
    Result of :func:`orbit_diagram`.

    An :class:`~tsdynamics.analysis._result.AnalysisResult`, so it carries
    ``.meta`` / the readout ``repr`` / ``.to_dict()`` / the ``.plot`` seam.  Iterate to
    get ``(value, points)`` pairs, or use :meth:`flat` for the scatter-ready
    arrays.

    Attributes
    ----------
    param : str
        Name of the swept control parameter.
    values : ndarray
        **The swept parameter axis** — see the note below.
    points : list of ndarray
        The asymptotic orbit recorded at each value, ``points[k]`` of shape
        ``(n, k_components)``; one entry per entry of :attr:`values`.

    .. note::
        **``values`` is the parameter axis here, not the answer.**  The name
        means three different things across the result classes — on a Lyapunov
        spectrum or an embedding it is the measurement, on this class and on
        :class:`~tsdynamics.analysis.results.ContinuationResult` it is the
        *control parameter* that was swept, and on
        :class:`~tsdynamics.analysis.results.ReturnMap` it is the observable
        series.  Always read the field's own line below before using it.
    """

    param: str = ""
    #: **The swept PARAMETER values, not the measured points** — the horizontal
    #: axis of the diagram, one entry per sweep step, in sweep order.  What was
    #: *measured* at each of them is :attr:`points` (and ``flat()`` pairs the
    #: two into the scatter the picture draws).
    #:
    #: The generic name is kept deliberately: it is what
    #: :class:`~tsdynamics.analysis.results.ContinuationResult` calls the same
    #: axis, so a sweep reads the same way whichever verb produced it — but it
    #: is the one name in the result layer that means something different from
    #: class to class, so it is spelled out at every site.
    values: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)  # (V,)
    points: list[np.ndarray] = field(default_factory=list, compare=False)  # per value (n, k)
    components: tuple[int, ...] = ()
    #: Per value: did the flow settle on an **equilibrium** rather than an orbit?
    #:
    #: This is the one thing a bifurcation diagram of a flow could not say.  A
    #: fixed point and a period-1 limit cycle both record ONE branch, so the
    #: readout called both "period 1" — measured on Chua over alpha in [6, 11],
    #: four fifths of the sweep is the equilibrium branch and the summary read
    #: ``periods seen: 1``, which is indistinguishable from a broken sweep.  The
    #: peak-map path already detects it (the trajectory tail stops moving); it
    #: simply had nowhere to put the answer.
    equilibria: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool), compare=False)

    def __iter__(self) -> Iterator[tuple[Any, np.ndarray]]:
        return iter(zip(self.values, self.points, strict=True))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: Any) -> Any:
        """Return the ``(value, points)`` pair at position ``key`` — what iteration yields.

        It was sized and iterable but **not** subscriptable, so ``od[0]`` raised
        ``TypeError`` while ``len(od)`` and ``for v, pts in od`` both worked, and
        ``np.asarray(od)`` degenerated to a 0-d object array.  Indexing and
        iteration now agree, item for item.
        """
        return list(zip(self.values, self.points, strict=True))[key]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return the diagram as the ``(N, 2)`` scatter it draws: parameter, point.

        The long form of :meth:`flat` for the first recorded component — the
        picture itself, rather than the 0-d *object* array numpy used to build.
        """
        if not self.points:
            return np.empty((0, 2), dtype=float)
        x, y = self.flat()
        arr = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        return arr.astype(dtype, copy=bool(copy)) if dtype is not None else arr

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
        self,
        *,
        component: int = 0,
        max_period: int = 16,
        rtol: float = 0.01,
        labelled: bool = False,
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
        labelled : bool, default False
            Return a structured array carrying the **period on each side** of
            every transition, so a genuine period-doubling can be told from a
            band split inside the chaotic regime.

            Measured on the logistic map, the plain array holds 20 values of
            which the first (``r = 3.000``) is the period-doubling and the other
            19 are branch splittings inside the chaotic band — successive gaps
            give ratios like 11.7, nothing near Feigenbaum's 4.669 — with nothing
            to say which is which.  With ``labelled=True`` the rows carry
            ``value`` / ``before`` / ``after`` / ``kind``, and
            ``rows["kind"] == "period-doubling"`` selects the cascade.

            .. versionadded:: 6.0

        Returns
        -------
        numpy.ndarray
            ``labelled=False`` (the default): the estimated bifurcation parameter
            values, in sweep order, resolved to the spacing of ``values``.
            ``labelled=True``: a structured array with fields ``value`` (float),
            ``before`` / ``after`` (int periods, ``0`` = aperiodic) and ``kind``
            (``"period-doubling"`` / ``"period-halving"`` / ``"onset of chaos"``
            / ``"band split"`` / ``"periodic window"`` / ``"period change"``).

        Examples
        --------
        >>> import numpy as np, tsdynamics as ts
        >>> od = ts.analysis.orbit_diagram(
        ...     ts.systems.Logistic(), "r", np.linspace(2.8, 3.6, 120)
        ... )
        >>> rows = od.bifurcation_points(labelled=True)
        >>> float(rows["value"][rows["kind"] == "period-doubling"][0])  # doctest: +SKIP
        3.0033613445378155

        References
        ----------
        Feigenbaum, M. J. (1978). Quantitative universality for a class of
        nonlinear transformations. *Journal of Statistical Physics*, 19, 25--52.
        """
        p = self.periods(component=component, max_period=max_period, rtol=rtol)
        if not labelled:
            return self._bifurcation_points_from_periods(p)
        return self._labelled_bifurcations(p)

    #: The transition kinds :meth:`bifurcation_points` names, longest field first
    #: so the structured array's ``U`` width is right.
    _BIFURCATION_KINDS = (
        "period-doubling",
        "period-halving",
        "onset of chaos",
        "periodic window",
        "band split",
        "period change",
    )

    def _labelled_bifurcations(self, periods: np.ndarray) -> np.ndarray:
        """Build the structured array :meth:`bifurcation_points` returns when labelled."""
        changed = (periods[:-1] != periods[1:]) & (periods[:-1] != -1) & (periods[1:] != -1)
        (i,) = np.nonzero(changed)
        values = 0.5 * (np.asarray(self.values)[i] + np.asarray(self.values)[i + 1])
        before = periods[i].astype(int)
        after = periods[i + 1].astype(int)
        width = max(len(k) for k in self._BIFURCATION_KINDS)
        rows = np.empty(
            i.size, dtype=[("value", float), ("before", int), ("after", int), ("kind", f"U{width}")]
        )
        rows["value"] = values
        rows["before"] = before
        rows["after"] = after
        for j, (b, a) in enumerate(zip(before, after, strict=True)):
            if b > 0 and a == 2 * b:
                kind = "period-doubling"
            elif a > 0 and b == 2 * a:
                kind = "period-halving"
            elif b > 0 and a <= 0:
                kind = "onset of chaos"
            elif b <= 0 and a > 0:
                kind = "periodic window"
            elif b <= 0 and a <= 0:
                kind = "band split"
            else:
                kind = "period change"
            rows["kind"][j] = kind
        return rows

    def _bifurcation_points_from_periods(self, periods: np.ndarray) -> np.ndarray:
        """Midpoints between consecutive values whose ``periods`` differ (no reseed).

        The core of :meth:`bifurcation_points`, factored so a caller that has
        already computed ``periods`` (e.g. :meth:`__plot_spec__`) reuses it instead
        of recomputing the period sweep.  Transitions touching a diverged value
        (``-1``) are skipped.
        """
        changed = (periods[:-1] != periods[1:]) & (periods[:-1] != -1) & (periods[1:] != -1)
        (i,) = np.nonzero(changed)
        return cast(np.ndarray, 0.5 * (self.values[i] + self.values[i + 1]))

    def __plot_spec__(self, kind: str | None = None, *, annotate: bool = False) -> Any:
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
        # reproducible from the picture.  It belongs in the TITLE, though — the y
        # axis is a picture of the OBSERVABLE, and labelling it "Poincaré section
        # (1, 0.0)" named the slicing plane instead, identically for every choice
        # of ``components=``.  The section stays one line up, where it says how
        # the diagram was read without claiming to say what was measured.
        section = self.meta.get("section")
        observable = self.meta.get("observable")
        ylabel = str(observable) if observable else "asymptotic state"
        return pb.spec(
            kind,
            "orbit_diagram",
            layers=[pb.scatter(x, y, style={"markersize": 1.0})],
            xlabel=self.param,
            ylabel=ylabel,
            title=f"bifurcation diagram — {section}" if section else "bifurcation diagram",
            annotations=annotations,
            meta=self.meta,
        )

    def _answer(self) -> str:
        """Return the swept range and the size of the diagram.

        Columns are ragged on the flow path — a value that settled on an
        equilibrium records one point where a chaotic one records ``n`` — so
        quoting the FIRST column's size described the whole diagram as
        "1 points/value" when 39 000 points were in it.  Show the range.
        """
        sizes = [int(p.shape[0]) for p in self.points]
        lo, hi = (min(sizes), max(sizes)) if sizes else (0, 0)
        per = f"{lo}" if lo == hi else f"{lo}–{hi}"
        v = np.asarray(self.values, dtype=float)
        span = f"{self.param} ∈ [{_sig(v[0], 4)}, {_sig(v[-1], 4)}] · " if v.size else ""
        return f"{span}{len(self.values)} values × {per} points"

    def _where(self) -> str:
        """Render the parameter span the equilibrium branch covers, when contiguous."""
        settled = np.asarray(self.equilibria, dtype=bool)
        v = np.asarray(self.values, dtype=float)
        if settled.size != v.size or not settled.any():
            return ""
        idx = np.flatnonzero(settled)
        contiguous = int(idx[-1] - idx[0] + 1) == idx.size
        if not contiguous:
            return ""
        return f" ({self.param} ∈ [{_sig(v[idx[0]], 4)}, {_sig(v[idx[-1]], 4)}])"

    def _details(self) -> tuple[str, ...]:
        """Return what the cascade did: the periods seen and where they changed.

        This is the *answer* a bifurcation diagram is computed for, and it is
        two method calls away from a reader who does not know the method names.
        Both are recomputed here, so a very large diagram is skipped rather than
        making a repr slow (:data:`_REPR_MAX_POINTS`), and a failure to cluster
        is silent — a repr must never raise.
        """
        total = sum(int(p.shape[0]) for p in self.points)
        if not self.points or total > _REPR_MAX_POINTS:
            return ()
        try:
            periods = np.asarray(self.periods())
            cuts = np.asarray(self.bifurcation_points(), dtype=float)
        except Exception:  # pragma: no cover - a repr must never raise
            return ()
        lines = []
        settled = np.asarray(self.equilibria, dtype=bool)
        if settled.size == periods.size and settled.any():
            # Name the fixed-point branch.  "periods seen: 1" over a sweep that
            # is four fifths equilibrium is the one place this library said
            # something ambiguous about the PHYSICS, and it reads exactly like a
            # broken sweep: a reader who cannot do the Routh-Hurwitz by hand
            # files a bug and goes back to writing the loop themselves.
            lines.append(f"settled on an EQUILIBRIUM at {int(settled.sum())} values{self._where()}")
            periods = periods[~settled]
        seen = sorted({int(p) for p in periods if p > 0})
        aperiodic = int((periods <= 0).sum())
        if seen:
            tail = f" · {aperiodic} aperiodic values" if aperiodic else ""
            lines.append(f"periods seen: {', '.join(str(p) for p in seen)}{tail}")
        if cuts.size:
            shown = ", ".join(_sig(c, 4) for c in cuts[:_REPR_MAX_BIFURCATIONS])
            more = " …" if cuts.size > _REPR_MAX_BIFURCATIONS else ""
            lines.append(f"bifurcations at {self.param} = {shown}{more}")
        return tuple(lines)


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
    ic_resolved = np.asarray(system._resolve_ic(ic), dtype=np.float64).reshape(system.dim)
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
    # The start state is named AT the call.  Since v6 ``run`` is always a fresh
    # integration and reinitialises first (§3.1), so a preceding
    # ``current.reinit(start)`` was silently discarded — and with it the sweep's
    # carry-state contract, which is what continues one parameter value's
    # attractor into the next.
    section = current.run(n, transient=transient, ic=start)
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

    The spread is **per component, over time** (``ptp`` down the time axis, then
    the widest component).  It used to be
    ``max(tail, axis=0).max() - min(tail, axis=0).min()``, which is the range
    ACROSS components — so a system resting at ``(1.5, 0, -1.5)`` measured a
    "spread" of 3.0 and was never called stationary.  Nothing detected an
    equilibrium at all: measured on Chua over alpha in [6, 11], the entire
    fixed-point branch was recorded as the peaks of the decaying spiral
    approaching it and summarised as ``periods seen: 1``.
    """
    if y.shape[0] < 10:
        return False
    tail = y[-max(2, y.shape[0] // 10) :]
    spread = float(np.ptp(tail, axis=0).max())
    scale = max(float(np.ptp(y, axis=0).max()), 1.0)
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
            f"orbit_diagram: only {found - transient} of {n} peaks were found within "
            f"max_time={max_time:g}, so this value's column is short. To fill it:\n"
            f"    ts.analysis.orbit_diagram(system, param, values, max_time={max_time * 10:g})",
            RuntimeWarning,
            stacklevel=4,
        )
        return rec[transient:]
    if found:
        keep = rec[-min(n, found) :]
        warnings.warn(
            f"orbit_diagram: only {found} peaks were found within "
            f"max_time={max_time:g}, fewer than transient={transient}, so this value's column "
            f"is the last {keep.shape[0]} of them and its transient is NOT fully discarded "
            "(transient/n count peaks here, not iterates — a slow oscillator makes far fewer "
            "of them than a map does). Either:\n"
            f"    ts.analysis.orbit_diagram(system, param, values, transient={max(found // 4, 1)},"
            f" points_per_value={max(found // 2, 1)})\n"
            f"    ts.analysis.orbit_diagram(system, param, values, max_time={max_time * 10:g})",
            RuntimeWarning,
            stacklevel=4,
        )
        return keep
    # Only offer ``components=`` when the state actually has another component to
    # offer: a scalar flow (a 1-D DDE) would be told to type an index that does
    # not exist, which is worse than no suggestion at all.
    other = next((c for c in range(dim) if c != idx[0]), None)
    lines = ["    ts.analysis.orbit_diagram(system, param, values, section=('z', 27.0, 'up'))"]
    if other is not None:
        lines.append(f"    ts.analysis.orbit_diagram(system, param, values, components={other})")
    warnings.warn(
        f"orbit_diagram: component {idx[0]} of this flow has no maximum within "
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
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Record one column, plus whether the flow SETTLED — the peak map of a flow.

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
        traj = current.run(final_time=horizon, dt=dt, ic=start)
        y = np.asarray(traj.y, dtype=float)
        if _is_stationary(y):
            # Settled on an equilibrium: the asymptotic set is the single state.
            # The FLAG is the point — a fixed point and a period-1 limit cycle
            # both record one branch, and only this path can tell them apart.
            return y[-1][idx][None, :], y[-1], horizon, True
        rec = _peak_records(y, idx[0], idx)
        if rec.shape[0] >= need:
            return rec[transient:need], y[-1], horizon, False
        if horizon >= max_time:
            return (
                _short_column(rec, transient, n, idx, max_time, y.shape[1]),
                y[-1],
                horizon,
                False,
            )
        # Grow the horizon by the *measured* peak rate rather than blindly
        # doubling: one short pilot run tells us how much time a peak costs, so
        # the second attempt already lands (and the horizon that worked is
        # returned, so the rest of the sweep pays for exactly one integration).
        grow = (need + 5) / max(rec.shape[0], 1) if rec.shape[0] else 4.0
        horizon = min(horizon * max(grow, 1.5), max_time)


def _discrete_view(system: Any, section: Any, component_label: str, param: str) -> tuple[Any, str]:
    """Resolve ``system`` to a discrete-time view, returning ``(view, description)``.

    A genuine discrete view passes straight through.  A **flow** is turned into
    one — the whole point of :func:`orbit_diagram`, whose canonical use is a
    flow — either through the section the caller asked for or through the peak map
    (signalled by a ``None`` view, which routes to :func:`_record_via_peaks`).
    The description is recorded in the result's ``meta["section"]`` and printed on
    the figure, so the choice is never made silently.
    """
    from tsdynamics.derived.poincare import PoincareMap
    from tsdynamics.derived.stroboscopic import StroboscopicMap
    from tsdynamics.errors import InvalidInputError, InvalidParameterError

    # v6: ``is_discrete`` left the public surface (``family`` replaced it), but a
    # DERIVED wrapper is the case ``family`` cannot answer yet — a PoincareMap of a
    # flow reports ``family="ode"`` while being a genuinely discrete view.  The
    # private ``_is_discrete`` is correct on all five wrappers and every family.
    if not hasattr(system, "_is_discrete"):
        raise InvalidInputError(
            f"orbit_diagram needs a dynamical system as its first argument, got "
            f"{type(system).__name__}. Pass a system (or a discrete view of one), e.g.\n"
            "    ts.analysis.orbit_diagram(ts.systems.Lorenz(), 'rho', "
            "np.linspace(0.0, 50.0, 200))"
        )
    if system._is_discrete:
        if section is not None:
            raise InvalidParameterError(
                f"section= chooses how to slice a *flow*, but {type(system).__name__} is "
                "already a discrete-time view, so there is nothing to slice. Drop section=:\n"
                f"    ts.analysis.orbit_diagram(system, {param!r}, values)"
            )
        if isinstance(system, PoincareMap):
            return system, f"Poincaré section {system.plane}"
        if isinstance(system, StroboscopicMap):
            return system, "stroboscopic sampling"
        return system, "map iterates"

    from tsdynamics.families import StochasticSystem

    if isinstance(system, StochasticSystem):
        raise InvalidInputError(
            f"orbit_diagram needs a deterministic system: {type(system).__name__} is "
            "stochastic (an SDE), so its 'asymptotic orbit' is a different sample path on "
            "every run. Sweep a deterministic model instead, e.g.\n"
            "    ts.analysis.orbit_diagram(ts.systems.Lorenz(), 'rho', "
            "np.linspace(0.0, 50.0, 200))"
        )
    if section is not None:
        return PoincareMap(system, section), f"Poincaré section {section}"
    return None, f"successive maxima of {component_label}"


def orbit_diagram(
    system: Any,
    param: str,
    values: Any,
    *,
    points_per_value: int = 200,
    transient: int = 500,
    carry_state: bool = True,
    components: int | str | tuple[Any, ...] = 0,
    section: Any | None = None,
    ic: Any | None = None,
    seed: int | None = 0,
    dt: float = _FLOW_DT,
    max_time: float = _FLOW_MAX_TIME,
) -> OrbitDiagram:
    """
    Sweep a parameter and record the asymptotic orbit at each value.

    Pass **any** system — this is the one-liner::

        ts.analysis.orbit_diagram(ts.systems.Lorenz(), "rho", np.linspace(0.0, 50.0, 200))

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

    ``orbit_diagram`` is the **one** spelling: ``bifurcation_diagram`` was the
    same object under a second name and is gone (a shared implementation can name
    only one of its spellings in a message, so half of all callers were answered
    about a function they had never typed).  The *picture* is still called a
    bifurcation diagram — ``PlotKind.BIFURCATION_DIAGRAM`` is untouched.

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
    points_per_value : int
        Points recorded per parameter value (map iterates / section crossings /
        peaks, according to the view) — the *height* of one vertical stripe of
        the diagram.

        .. versionchanged:: 6.0
            Named ``n`` before v6.  ``n`` was carrying four different meanings
            across the public surface (map iterations, points kept per parameter
            value, number of crossings, Monte-Carlo sample count); this is one of
            the two that misled, so it says what it counts.
    transient : int
        Points discarded before recording, at every value — a **count**, in the
        unit the sweep advances in: map iterations for a map, crossings for a
        Poincaré/stroboscopic view, recorded peaks for the flow peak map.  (The
        subject of an orbit diagram is always a discrete view, so this word never
        means time here, unlike ``run(transient=)`` on a flow.)
    carry_state : bool
        Start each value from the previous value's final state (follows the
        attractor branch; the classic way to draw clean diagrams).  When
        False, every value starts from ``ic`` / the system default.
    components : int, str, or tuple
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
    seed : int, default 0
        Seed for the random initial condition when the system has none, so the
        diagram is reproducible.  Pass ``seed=None`` for an unseeded draw.

        .. versionchanged:: 6.0
            Was ``None``: a system with no declared initial condition drew a
            fresh random one on every call, so the same line of code drew a
            different diagram each time it ran.
    dt : float, default 0.01
        Output sampling step of the flow path's integration, in **time units**.
        Ignored for a map / an already-discrete view, which iterate.
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
        If a named ``components`` is requested but the system does not declare
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
    >>> od = orbit_diagram(Lorenz(), "rho", np.linspace(0.0, 50.0, 200))
    >>> od.meta["section"]
    'successive maxima of x'
    >>> # ... or name the section yourself:
    >>> od = orbit_diagram(Rossler(), "c", np.linspace(2, 6, 80),
    ...                          section=("y", 0.0, "up"))
    >>> # a map:
    >>> od = orbit_diagram(Logistic(), "r", np.linspace(2.5, 4.0, 600), points_per_value=120)
    >>> x, y = od.flat()
    """
    # Measured data first, through the shared guard.  A ``Trajectory`` carries a
    # ``_is_discrete`` flag, so it walked straight past the system check below and
    # failed ~300 lines later on ``view.with_params`` — an internal name, in an
    # AttributeError, about an object the caller never reached for.
    reject_data(system, analysis="orbit_diagram")
    comp = (components,) if isinstance(components, int | str) else tuple(components)
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

    # One short local alias: the sweep machinery below (and the helpers it calls
    # positionally) reads better with the terse name, while the *signature* says
    # what the number counts.
    n = points_per_value
    values_arr = np.asarray(list(values), dtype=float)
    points: list[np.ndarray] = []
    state: np.ndarray | None = None
    horizon = _FLOW_FINAL_TIME

    def _meta() -> dict[str, Any]:
        return {
            "system": type(system).__name__,
            "param": param,
            "points_per_value": n,
            "transient": transient,
            "carry_state": carry_state,
            "components": tuple(idx),
            # The NAME of the recorded observable, not merely its index: it is
            # what the figure's y axis is a picture of.  Without it the axis was
            # labelled with the *section description* — identical for
            # ``components=0`` and ``components='z'``, so the one label that had
            # to distinguish them could not.
            "observable": label,
            # The discrete view this diagram was read through — recorded so an
            # auto-chosen section is never a silent choice (it is also printed
            # under the figure's title by ``__plot_spec__``).
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
    equilibria: list[bool] = []
    for v in values_arr:
        current = view.with_params(**{param: v})
        start = state if (carry_state and state is not None) else ic
        try:
            if is_peak_map:
                # A raw flow: read it as the successive-maxima (peak) map.  The
                # horizon that satisfied the previous value seeds the next one, so
                # only the first value pays for discovering it.
                rec, last, horizon, settled = _record_via_peaks(
                    current, start, transient, n, idx, dt=dt, final_time=horizon, max_time=max_time
                )
                equilibria.append(settled)
            else:
                # Flow wrappers (PoincareMap / StroboscopicMap) and the
                # engine-sweep fallback (a non-lowerable map / wheel-free env)
                # drive the per-step protocol loop — byte-identical to the engine
                # path on a lowerable map.
                record = _record_via_trajectory if use_trajectory else _record_via_step
                rec, last = record(current, start, transient, n, idx)
                # A discrete view cannot reach an equilibrium silently: a flow
                # that settles stops crossing its section, which raises below.
                equilibria.append(False)
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
            if len(equilibria) < len(points):
                equilibria.append(False)
            state = None
            continue
        points.append(rec)
        if carry_state:
            state = last

    return OrbitDiagram(
        param=param,
        values=values_arr,
        points=points,
        components=tuple(idx),
        equilibria=np.asarray(equilibria, dtype=bool),
        meta=_meta(),
    )


# ``bifurcation_diagram`` used to be an alias of ``orbit_diagram`` — the same
# object under two names.  It was deleted in v6 under "one concept, one
# spelling": a shared implementation can only ever name ONE of its spellings in
# a ``TypeError``, so half of all callers were sent to look up a function they
# had never typed.  ``ts.bifurcation_diagram`` now raises an ``AttributeError``
# naming ``orbit_diagram`` (see ``_RENAMED_IN_V6``).  The *picture* is still
# called a bifurcation diagram — ``PlotKind.BIFURCATION_DIAGRAM`` is a different
# concept and is untouched.


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
