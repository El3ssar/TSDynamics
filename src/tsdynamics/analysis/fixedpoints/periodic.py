r"""
Periodic orbits of maps and flows, and period estimation from a signal.

- :func:`periodic_orbits` — period-``p`` orbits of a
  :class:`~tsdynamics.families.DiscreteMap`.  A period-``p`` orbit is a fixed
  point of the ``p``-fold composition :math:`f^{p}`, so the Schmelcher--Diakonos
  / Davidchack--Lai stabilising-transformation root finder is run on
  :math:`g(x) = f^{p}(x) - x`.  Orbits whose *minimal* period properly divides
  ``p`` are filtered out (``prime=True``) and cyclic shifts of one orbit are
  merged.
  For a :class:`~tsdynamics.families.ContinuousSystem` the same verb finds the
  limit cycle by single shooting: Newton on
  the unknowns ``(x0, T)`` solving :math:`\varphi_T(x_0) - x_0 = 0` with an
  orthogonality phase condition, using the monodromy matrix from the variational
  equations.  Stability is read from the Floquet multipliers.
- :func:`estimate_period` — the dominant period of a sampled signal
  (autocorrelation or spectral peak), used to seed shooting or to characterise a
  limit cycle.

References
----------
Schmelcher & Diakonos (1997), *Phys. Rev. Lett.* 78, 4733.
Davidchack & Lai (1999), *Phys. Rev. E* 60, 6172.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from tsdynamics.errors import ConvergenceError, InvalidInputError, remedy
from tsdynamics.families import ContinuousSystem, DiscreteMap

from .._common import reject_data, reject_system
from .._result import AnalysisResult, CollectionResult, ScalarResult
from .._result_json import _sig, _state
from . import _common as _c
from .fixed import _build_seeds, _eigenvalue_plane_spec, _stabilising_matrices

__all__ = [
    "OrbitSet",
    "PeriodicOrbit",
    "estimate_period",
    "period_diagnostic",
    "periodic_orbits",
]

# The two seeding recipes every shooting failure points at.  They are kept here,
# spelled as complete statements, because an error that says "seed it better"
# without handing over the lines to type is not a remedy -- and because a line
# naming a `traj` the reader never defined is not runnable either.
#
# They are complementary, which is why the escalation below offers the second
# only after the first has already failed: the autocorrelation period works when
# the signal has one dominant frequency (Rossler closes at T=17.5158), while the
# near-return scan works when it does not (Lorenz T=3.82025, Chua T=1.6671 --
# both of which the autocorrelation guess sends to an equilibrium instead).
_SEED_FROM_ATTRACTOR = (
    "traj = system.run(final_time=200.0, dt=0.01)",
    "ts.analysis.periodic_orbits(system, float(ts.analysis.estimate_period(traj)), ic=traj.y[-1])",
)
_SEED_FROM_NEAR_RETURN = (
    "traj = system.run(final_time=200.0, dt=0.01)",
    "lag = int(np.argmin([np.linalg.norm(traj.y[m:] - traj.y[:-m], axis=1).min()",
    "                     for m in range(50, 500)])) + 50   # the closest near-return",
    "k = int(np.linalg.norm(traj.y[lag:] - traj.y[:-lag], axis=1).argmin())",
    "ts.analysis.periodic_orbits(system, lag * 0.01, ic=traj.y[k])",
)


@dataclass(frozen=True)
class PeriodicOrbit(AnalysisResult):
    r"""A periodic orbit with its Floquet/multiplier stability data.

    Attributes
    ----------
    points : ndarray
        The orbit, shape ``(n_points, dim)``: the ``period`` distinct points of a
        map cycle, or a dense sampling along one period of a flow cycle.
    period : int or float
        The (minimal) period — an integer iteration count for a map, the time
        ``T`` for a flow.
    multipliers : ndarray
        Stability multipliers: eigenvalues of :math:`Df^{p}` at an orbit point
        (map) or the Floquet multipliers (eigenvalues of the monodromy matrix) of
        the cycle (flow).  A flow always carries one trivial multiplier ``≈ 1``
        along the flow direction.
    stable : bool
        ``True`` iff every *non-trivial* multiplier lies inside the unit circle.
    continuous : bool
        ``True`` for a flow cycle, ``False`` for a map cycle.
    residual : float
        Closure residual ``‖f^p(x) − x‖`` (map) or ``‖φ_T(x0) − x0‖`` (flow).
    """

    points: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    period: int | float = 0
    multipliers: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    stable: bool = False
    continuous: bool = False
    residual: float = 0.0

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe this periodic orbit as a backend-agnostic :class:`PlotSpec`.

        Builds a phase portrait of the orbit :attr:`points`: a closed ``LINE3D``
        loop for a 3-D-or-higher flow cycle (first three coordinates), a 2-D
        ``LINE`` (flow) / ``SCATTER`` (the discrete map cycle's distinct points)
        for two coordinates, and a 1-D index plot for a scalar map.  A flow loop
        is drawn as a line (the continuous cycle); a map orbit as markers (its
        ``period`` distinct points).  The :mod:`tsdynamics.viz.spec` import is
        lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
            value).  ``None`` picks ``PHASE_PORTRAIT_3D`` / ``PHASE_PORTRAIT_2D``
            / ``TIME_SERIES`` from the orbit's dimensionality.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        pts = np.atleast_2d(np.asarray(self.points, dtype=float))
        dim = pts.shape[1] if pts.ndim == 2 and pts.size else 1
        per = f"T = {self.period:.4g}" if self.continuous else f"p = {int(self.period)}"
        title = f"{'stable' if self.stable else 'unstable'} orbit ({per})"
        # A flow loop is a continuous line; a map orbit is its distinct points.
        mark2d = pb.line if self.continuous else pb.scatter

        if dim >= 3:
            loop = (
                pb.line3d(pts[:, 0], pts[:, 1], pts[:, 2], label="orbit")
                if self.continuous
                else pb.markers(pts[:, 0], pts[:, 1], z=pts[:, 2], label="orbit")
            )
            return pb.spec(
                kind,
                "phase_portrait_3d",
                layers=[loop],
                aspect="equal",
                xlabel="$x_0$",
                ylabel="$x_1$",
                zlabel="$x_2$",
                title=title,
            )
        if dim == 2:
            return pb.spec(
                kind,
                "phase_portrait_2d",
                layers=[mark2d(pts[:, 0], pts[:, 1], label="orbit")],
                aspect="equal",
                xlabel="$x_0$",
                ylabel="$x_1$",
                title=title,
            )
        y = pts[:, 0] if pts.ndim == 2 else np.ravel(pts).astype(float)
        return pb.spec(
            kind,
            "time_series",
            layers=[mark2d(np.arange(y.size, dtype=float), y, label="orbit")],
            xlabel="index",
            ylabel="$x$",
            title=title,
        )

    def eigenvalue_plane(self, kind: str | None = None) -> Any:
        r"""Describe the multiplier spectrum as an :class:`EIGENVALUE_PLANE` spec.

        Plots the stability multipliers in the complex plane against the unit
        circle (a map's eigenvalues of :math:`Df^{p}`, or a flow's Floquet
        multipliers, both judged by ``|μ| < 1``).  For a **flow** the trivial
        multiplier ``≈ 1`` along the flow direction is split into its own
        distinctly-marked layer — located here for the plot by ``argmin|μ − 1|``
        (a presentation heuristic; the stability flag itself uses the more robust
        eigenvector-alignment test).  The :mod:`tsdynamics.viz.spec` import is
        lazy.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``EIGENVALUE_PLANE``.

        Returns
        -------
        PlotSpec
        """
        mu = np.asarray(self.multipliers).ravel().astype(complex)
        trivial = int(np.argmin(np.abs(mu - 1.0))) if (self.continuous and mu.size) else None
        per = f"T = {self.period:.4g}" if self.continuous else f"p = {int(self.period)}"
        title = f"Floquet multipliers ({per})" if self.continuous else f"multipliers ({per})"
        return _eigenvalue_plane_spec(
            mu,
            continuous=False,  # multipliers live on the unit-circle convention (maps + flows)
            title=title,
            meta=dict(self.meta) if self.meta else {},
            kind=kind,
            trivial_index=trivial,
        )

    def _period_label(self) -> str:
        """Return ``period <p>`` for a map orbit or ``T = <t>`` for a flow's cycle."""
        return f"T = {self.period:.6g}" if self.continuous else f"period {int(self.period)}"

    def _gauge(self) -> str:
        """Return the leading multiplier — the reading stability is decided from."""
        mu = np.asarray(self.multipliers)
        if not mu.size:
            return "no multipliers"
        return f"|μ|max = {_sig(float(np.abs(mu).max()), 4)}"

    def _answer(self) -> str:
        """Return the period and the point the orbit starts from."""
        x0 = np.asarray(self.points, dtype=float)
        start = f"  x0 = {_state(x0[0])}" if x0.size else ""
        return f"{self._period_label()}{start}"

    def _interpretation(self) -> str | None:
        """Name the stability, and the multiplier it was decided from."""
        return ("stable" if self.stable else "unstable") + f"  {self._gauge()}"

    def _context(self) -> str | None:
        """Return the subject and how many points the orbit is stored as."""
        bits = [b for b in (self._system_label(),) if b]
        bits.append(f"{len(self.points)} points")
        return ", ".join(bits)

    def _as_item(self) -> str:
        """Return the compact one-line form used inside an orbit set's list."""
        x0 = np.asarray(self.points, dtype=float)
        start = f"  x0 = {_state(x0[0])}" if x0.size else ""
        stability = "stable" if self.stable else "unstable"
        return f"{self._period_label()}  {stability}  {self._gauge()}{start}"


@dataclass(frozen=True, eq=False)
class OrbitSet(CollectionResult):
    """The set of periodic orbits found, behaving like a ``list``.

    A :class:`~tsdynamics.analysis._result.CollectionResult`: iterate it, index it
    (``orbits[0]`` is a :class:`PeriodicOrbit`), take its ``len``, and read
    :attr:`stable` / :attr:`unstable` sublists — while it carries ``.meta`` /
    the readout ``repr`` / ``.to_frame()`` / the ``.plot`` seam.
    """

    @property
    def stable(self) -> list[PeriodicOrbit]:
        """The stable orbits in the set."""
        return [o for o in self.items if o.stable]

    @property
    def unstable(self) -> list[PeriodicOrbit]:
        """The unstable orbits in the set."""
        return [o for o in self.items if not o.stable]

    def _noun(self) -> str:
        """Return ``orbit`` — what one item of this collection is."""
        return "orbit"

    def _answer(self) -> str:
        """Return the count, the shared period when there is one, and the split."""
        if not self.items:
            return "none found"
        periods = {o.period for o in self.items}
        shared = f" of period {int(next(iter(periods)))}" if len(periods) == 1 else ""
        n_stable = len(self.stable)
        plural = "s" if len(self.items) != 1 else ""
        return (
            f"{len(self.items)} {self._noun()}{plural}{shared} · "
            f"{n_stable} stable, {len(self.items) - n_stable} unstable"
        )

    def _derived(self) -> dict[str, Any]:
        """Export the stable/unstable split the repr reports."""
        return {"n_stable": len(self.stable), "n_unstable": len(self.unstable)}

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the whole orbit set as one backend-agnostic phase portrait.

        Overlays every orbit's points in one spec — one labelled layer per orbit
        (a ``SCATTER`` of a map cycle's distinct points, a ``LINE`` of a flow
        loop) — so the family of period-``p`` orbits is drawn together.  The
        canvas dimensionality follows the orbits' state dimension (3-D, 2-D, or a
        1-D index plot for a scalar map), and a :class:`~tsdynamics.viz.spec.Legend`
        is attached when more than one orbit is present.  An empty set yields a
        valid (layer-less) ``PHASE_PORTRAIT_2D``.  The :mod:`tsdynamics.viz.spec`
        import is lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
            value).  ``None`` picks the kind from the orbits' dimensionality.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Layer

        from .. import _plotbuilder as pb

        orbits = list(self.items)
        if not orbits:
            return pb.spec(
                kind, "phase_portrait_2d", layers=[], title="periodic orbits (none found)"
            )

        # Build each orbit's own spec (it owns the per-dim layering logic) and
        # gather the layers, relabelling each by its period/stability.
        sub = [(o, o.to_plot_spec()) for o in orbits]
        ndim = max(int(s.ndim) for _, s in sub)
        layers: list[Layer] = []
        for o, s in sub:
            per = f"T={o.period:.4g}" if o.continuous else f"p={int(o.period)}"
            label = f"{'stable' if o.stable else 'unstable'} {per}"
            for lyr in s.layers:
                layers.append(Layer(lyr.kind, dict(lyr.data), label=label, style=dict(lyr.style)))

        return pb.spec(
            kind,
            "phase_portrait_3d" if ndim == 3 else "phase_portrait_2d",
            layers=layers,
            aspect="equal",
            xlabel="$x_0$",
            ylabel="$x_1$",
            zlabel="$x_2$" if ndim == 3 else None,
            title=f"periodic orbits ({len(orbits)} found)",
            legend=len(orbits) > 1,
        )


# ── periodic orbits of maps ───────────────────────────────────────────────────


def periodic_orbits(
    system: Any,
    period: int | float | None = None,
    *,
    region: Any = None,
    n_seeds: int = 300,
    method: str = "newton",
    lam: float = 0.05,
    beta: float = 1.0,
    tol: float | None = None,
    max_iter: int | None = None,
    dedup_tol: float = 1e-6,
    prime: bool = True,
    max_c: int | None = None,
    seed: int | None = None,
    ic: Any = None,
    transient: float = 0.0,
    steps_per_period: int = 2000,
    n_points: int = 400,
    min_amplitude: float = 1e-6,
) -> OrbitSet:
    r"""
    Periodic orbits of a map, or a flow's limit cycle.

    Solves :math:`f^{p}(x) = x` by multi-start root finding, recovers each orbit
    by forward iteration, filters orbits whose minimal period properly divides
    ``period`` (``prime=True``), and merges the cyclic shifts of one orbit.

    The default is plain **Newton** on :math:`f^{p}`.  It was Davidchack--Lai
    until v6, on the reasoning that the stabilising transformations reach
    unstable orbits Newton misses; measured, that is not what happens at the
    periods users actually explore.  On the logistic map at ``r = 4``, where the
    prime-cycle counts are known exactly (2, 1, 2, 3, 6, 9, 18 for
    :math:`p = 1 \ldots 7`), Newton at the default ``n_seeds`` recovers **all of
    them**, and so do ``"sd"`` and ``"dl"`` — at up to 19x the cost.  On
    Henon(1.4, 0.3) the two agree on the count at every :math:`p \le 7` while
    Newton is **112-323x** faster (``p=7``: 0.09 s against 29.0 s), because
    ``"dl"`` runs every seed against each of the :math:`2^d d!` stabilising
    matrices and each residual is a ``p``-fold monodromy sweep.  The cheap axis
    for completeness is ``n_seeds``, not the transformation: raise it first, and
    reach for ``"dl"`` when a high period still comes up short.

    Parameters
    ----------
    system : DiscreteMap
        The map.
    period : int
        The period ``p`` (``p=1`` returns the fixed points as one-point orbits).
    region, n_seeds, dedup_tol, seed
        Seeding controls (see :func:`~tsdynamics.analysis.fixedpoints.fixed_points`).
    method : {"newton", "sd", "dl"}
        Root finder.  ``"newton"`` (default) = Newton on ``f^p``;
        ``"sd"`` = Schmelcher--Diakonos; ``"dl"`` = Davidchack--Lai.  The two
        stabilising transformations cost :math:`2^d d!` root-finding passes per
        seed (384 matrices at ``dim=4``) — see above for the measured trade.
    lam, beta, max_c
        Stabilising-transformation controls (see ``fixed_points``).
    tol : float
        Residual tolerance ``‖f^p(x) − x‖``.
    max_iter : int
        Iterations per seed/matrix.
    prime : bool
        Keep only orbits of *minimal* period ``p`` (drop divisor-period orbits).

    Returns
    -------
    OrbitSet
        A list-like ``CollectionResult`` of :class:`PeriodicOrbit`, sorted by
        the orbit's lexicographically smallest point.

    Raises
    ------
    InvalidInputError
        If ``system`` is measured data, or is not a
        a :class:`~tsdynamics.families.DiscreteMap`, when a period is named.  A ``TypeError`` subclass, so ``except TypeError`` keeps working.
    ValueError
        If ``period < 1`` or ``method`` is not ``"newton"``/``"sd"``/``"dl"``.

    Examples
    --------
    >>> periodic_orbits(Logistic(params={"r": 3.2}), 2)   # the stable 2-cycle
    >>> periodic_orbits(Logistic(params={"r": 3.83}), 3)  # stable node + saddle
    """
    reject_data(system, analysis="periodic_orbits")
    if not isinstance(system, DiscreteMap):
        # A flow's periodic orbit is a closed curve with a REAL period, so it is
        # found by single shooting on ``(x0, T)`` rather than by rooting f^p.
        # One verb, one return type: v6 absorbed the old ``periodic_orbit``
        # (singular) here, and ``period`` is the period guess for a flow.
        return _flow_periodic_orbits(
            system,
            period_guess=None if period is None else float(period),
            ic=ic,
            transient=transient,
            steps_per_period=steps_per_period,
            tol=1e-10 if tol is None else float(tol),
            max_iter=50 if max_iter is None else int(max_iter),
            n_points=n_points,
            min_amplitude=min_amplitude,
            seed=seed,
        )
    if period is None:
        raise InvalidInputError(
            "periodic_orbits needs the period p to look for on a map — a map's "
            "orbits are the fixed points of f^p, one root problem per p."
            + remedy(
                "ts.analysis.periodic_orbits(system, 2)   # the 2-cycles",
                lead="Name the period:",
            )
        )
    tol = 1e-12 if tol is None else float(tol)
    max_iter = 200 if max_iter is None else int(max_iter)
    period = int(period)
    if period < 1:
        raise ValueError("period must be a positive integer.")
    method = method.lower()
    if method not in ("newton", "sd", "dl"):
        raise ValueError(f"method must be 'newton', 'sd', or 'dl', got {method!r}.")

    dim = int(system.dim)  # type: ignore[arg-type]  # dim resolved at construction
    rng = np.random.default_rng(seed)
    step, jac = _c.map_fns(system)
    eye = np.eye(dim)

    # Both the residual ``g(x) = f^p(x) - x`` and its Jacobian ``Df^p - I`` come
    # from the *same* p-fold orbit + monodromy sweep.  ``converge_root`` evaluates
    # them at the identical iterate ``x`` within one step (e.g. DL calls both), so
    # cache the last ``(x_p, M)`` keyed on the input vector and share it between
    # the two closures — one ``map_orbit_monodromy`` per iterate instead of two.
    # The cache is per-iterate (single-slot), so the value is byte-identical to
    # recomputing.
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def _orbit_monodromy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = x.tobytes()
        hit = cache.get(key)
        if hit is None:
            x_p, m, _ = _c.map_orbit_monodromy(step, jac, x, period, dim)
            cache.clear()  # single-slot: only the live iterate is ever reused
            cache[key] = (x_p, m)
            return x_p, m
        return hit

    def residual(x: np.ndarray) -> np.ndarray:
        return cast("np.ndarray", _orbit_monodromy(x)[0] - x)

    def jac_resid(x: np.ndarray) -> np.ndarray:
        return _orbit_monodromy(x)[1] - eye

    # One burn-in orbit serves both the automatic box and the on-orbit seeds.
    orbit = _c.sample_orbit_box(system, dim, rng=rng) if region is None else None
    lo, hi = (
        _c.hull_box(orbit, dim, _c.HULL_PAD)
        if orbit is not None
        else _c.resolve_box(system, region, dim, rng)
    )
    seeds = _build_seeds(dim, lo, hi, n_seeds, rng, orbit=orbit)
    c_mats = _stabilising_matrices(method, dim, max_c)

    # Do not box-clip: an unstable orbit may sit outside the attractor's hull; the
    # closure-residual, prime-period and distinctness filters reject spurious roots.
    roots = _c.solve_roots(
        residual,
        jac_resid,
        dim,
        seeds,
        method=method,
        c_mats=c_mats,
        lam=lam,
        beta=beta,
        tol=tol,
        max_iter=max_iter,
        dedup_tol=dedup_tol,
        bounds=None,
    )

    # The minimal-period closure test is deliberately looser than the root tol:
    # re-iterating f^d from a root accurate only to `tol` accumulates round-off, so
    # a too-tight check would mistake a true divisor-period orbit for a prime one.
    orbit_tol = max(tol * 1e4, 1e-8)
    divisors = [d for d in range(1, period) if period % d == 0]
    reps: list[np.ndarray] = []
    orbits: list[PeriodicOrbit] = []
    for r in roots:
        m, points = _minimal_period(step, r, period, dim, divisors, orbit_tol)
        if prime and m != period:
            continue
        rep = points[np.lexsort(points.T[::-1])][0]  # lexicographically smallest point
        if any(np.linalg.norm(rep - q) < dedup_tol for q in reps):
            continue
        reps.append(rep)
        x_end, monodromy, _ = _c.map_orbit_monodromy(step, jac, rep, m, dim)
        eig = np.linalg.eigvals(monodromy)
        closure = float(np.linalg.norm(x_end - rep))
        orbits.append(
            PeriodicOrbit(
                points=points,
                period=int(m),
                multipliers=eig,
                stable=bool(np.all(np.abs(eig) < 1.0)),
                continuous=False,
                residual=closure,
            )
        )
    orbits.sort(key=lambda o: tuple(np.asarray(o.points)[0]))
    return OrbitSet(
        items=tuple(orbits),
        meta=AnalysisResult.build_meta(
            system, analysis="periodic_orbits", period=int(period), method=method
        ),
    )


def _minimal_period(
    step: Any, x: np.ndarray, period: int, dim: int, divisors: list[int], tol: float
) -> tuple[int, np.ndarray]:
    """Return ``(m, orbit_points)`` where ``m`` is the minimal period of ``x``.

    Iterates the map up to ``period`` times; the minimal period is the smallest
    divisor ``d`` of ``period`` with ``f^d(x) ≈ x`` (else ``period`` itself).
    ``orbit_points`` holds the ``m`` distinct points.
    """
    pts = np.empty((period, dim))
    cur = np.asarray(x, dtype=float).ravel().copy()
    for k in range(period):
        pts[k] = cur
        cur = step(cur)
    for d in divisors:  # ascending, so the first hit is the minimal period
        if np.linalg.norm(pts[d] - pts[0]) < tol:
            return d, pts[:d]
    return period, pts


# ── periodic orbits of flows (single shooting) ────────────────────────────────


def _flow_periodic_orbits(
    system: Any,
    *,
    ic: Any | None = None,
    period_guess: float | None = None,
    steps_per_period: int = 2000,
    transient: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 50,
    n_points: int = 400,
    min_amplitude: float = 1e-6,
    seed: int | None = None,
) -> OrbitSet:
    r"""
    Find a limit cycle of an autonomous flow by single shooting.

    Newton iterates the unknowns ``(x0, T)`` to solve ``φ_T(x0) − x0 = 0`` with an
    orthogonality phase condition ``f(x0)·δx = 0`` (which removes the trivial
    time-shift degeneracy).  The monodromy matrix ``M = dφ_T/dx0`` comes from
    integrating the variational equation alongside the state; the Floquet
    multipliers ``eig(M)`` give stability (one trivial multiplier ``≈ 1``).

    Parameters
    ----------
    system : ContinuousSystem
        An autonomous flow.
    ic : array-like, optional
        Initial guess for a point on the orbit (default: the system's IC, after a
        short burn-in if it is not already near the cycle).
    period_guess : float, optional
        Initial guess for the period ``T``.  If omitted, it is estimated from a
        burn-in trajectory via :func:`estimate_period`.
    steps_per_period : int
        Fixed RK4 sub-steps used to integrate one period (state + monodromy).
    transient : float
        Time to forward-integrate ``ic`` before shooting (default ``0``).  A few
        periods of burn-in lands a guess near a *stable* limit cycle, widening the
        Newton basin; leave it ``0`` when targeting an unstable orbit from a
        precise guess.
    tol : float
        Convergence tolerance on ``‖φ_T(x0) − x0‖``.
    max_iter : int
        Maximum Newton iterations.
    n_points : int
        Number of points sampled along the converged orbit.
    min_amplitude : float
        Minimum orbit extent (bounding-box diagonal) for the result to count as a
        genuine cycle.  Shooting can collapse onto the trivial solution
        ``x0 = equilibrium`` (any ``T``, residual ``0``) — common for a centre or
        from a poor guess — which is rejected below this threshold.
    seed : int, optional
        Seed for the burn-in IC, when ``ic`` is not given.  The random fallback is
        drawn from a *local* :class:`numpy.random.Generator` seeded with this
        value, so a given ``seed`` is reproducible regardless of the global
        ``numpy.random`` state; ``seed=None`` (the default) is non-deterministic.

    Returns
    -------
    OrbitSet
        One :class:`PeriodicOrbit` with ``continuous=True``, ``period`` the
        converged ``T`` and ``multipliers`` the Floquet multipliers.

    Raises
    ------
    InvalidInputError
        If ``system`` is measured data rather than a model.
    NotImplementedError
        If ``system`` is not a continuous flow.
    ValueError
        If ``period_guess`` (or the auto-estimated period) is not positive.
    ConvergenceError
        If the Newton iteration does not converge, or it collapses onto an
        equilibrium (the target may be a centre — a non-isolated orbit — rather
        than a hyperbolic cycle); seed from a point on the cycle.  A
        ``RuntimeError`` subclass, so ``except RuntimeError`` keeps working.

    Examples
    --------
    >>> periodic_orbits(VanDerPol(params={"mu": 1.0}), 6.6, ic=[2.0, 0.0])
    """
    reject_data(system, analysis="periodic_orbits")
    if not isinstance(system, ContinuousSystem):
        raise NotImplementedError(
            f"periodic_orbits shoots for a closed *flow* trajectory (x0, T), and "
            f"{type(system).__name__} has no continuous time — a map's periodic orbit "
            f"is a finite cycle of integer period."
            + remedy(
                "ts.analysis.periodic_orbits(system, 2)",
                lead="Use the map routine (plural), with the period you want:",
            )
        )
    dim = int(system.dim)  # type: ignore[arg-type]  # dim resolved at construction
    rhs, jac = _c.flow_fns(system)
    rng = np.random.default_rng(seed)

    x0 = (
        np.asarray(_c._orbit_start_ic(system, dim, rng), dtype=float).ravel()
        if ic is None
        else np.asarray(ic, dtype=float).ravel()
    )
    t_period = float(period_guess) if period_guess is not None else _guess_period(system, x0, dim)
    if t_period <= 0.0:
        raise ValueError("period_guess must be positive.")

    if transient > 0.0:  # land near a stable cycle to widen the Newton basin
        n_burn = max(1, int(round(transient / 0.01)))
        x0 = _c.flow_state(rhs, x0, float(transient), n_burn)

    eye = np.eye(dim)
    converged = False
    r_norm = float(np.linalg.norm(_c.flow_state(rhs, x0, t_period, steps_per_period) - x0))
    for _ in range(max_iter):
        x_end, monodromy = _c.flow_monodromy(rhs, jac, x0, t_period, steps_per_period)
        r = x_end - x0
        r_norm = float(np.linalg.norm(r))
        if r_norm < tol:
            converged = True
            break
        f0 = rhs(x0, 0.0)
        f_end = rhs(x_end, 0.0)  # = dφ_T/dT at the orbit
        # Bordered (d+1) Newton system:  [[M - I, f_end], [f0^T, 0]] δ = -[r, 0]
        amat = np.zeros((dim + 1, dim + 1))
        amat[:dim, :dim] = monodromy - eye
        amat[:dim, dim] = f_end
        amat[dim, :dim] = f0
        rhs_vec = np.concatenate([-r, [0.0]])
        try:
            delta = np.linalg.solve(amat, rhs_vec)
        except np.linalg.LinAlgError as exc:
            raise ConvergenceError(
                "periodic_orbits: the shooting Jacobian is singular, so the Newton step "
                "is undefined — the target is a centre (a continuum of orbits, none "
                "isolated) or the phase condition is degenerate at this point."
                + remedy(
                    *_SEED_FROM_ATTRACTOR,
                    lead="Seed from a different point on the orbit:",
                )
            ) from exc
        if not np.all(np.isfinite(delta)):
            raise ConvergenceError(
                "periodic_orbits: the Newton step is non-finite — the shooting "
                "trajectory blew up before it closed."
                + remedy(
                    *_SEED_FROM_ATTRACTOR,
                    lead="Seed from a point on the attractor and a period near the truth:",
                )
            )
        # Backtracking line search: take the largest fraction of the Newton step
        # that keeps T > 0 and strictly reduces the closure residual (shooting has
        # a small basin, so an undamped step can overshoot to T <= 0 or diverge).
        alpha, accepted = 1.0, False
        for _ls in range(30):
            x_try = x0 + alpha * delta[:dim]
            t_try = t_period + alpha * float(delta[dim])
            if t_try > 0.0:
                end_try = _c.flow_state(rhs, x_try, t_try, steps_per_period)
                r_try = float(np.linalg.norm(end_try - x_try))
                if np.isfinite(r_try) and r_try < r_norm:
                    x0, t_period, r_norm, accepted = x_try, t_try, r_try, True
                    break
            alpha *= 0.5
        if not accepted:
            break  # no productive step — report the converged-or-not state below

    x_end, monodromy = _c.flow_monodromy(rhs, jac, x0, t_period, steps_per_period)
    residual = float(np.linalg.norm(x_end - x0))
    if not converged and residual >= tol:
        # Escalate rather than loop.  A caller who passed no seed has not yet
        # tried the attractor recipe, so offer it; a caller who *did* seed has
        # already run that line, and repeating it back would be an error message
        # telling them to type what they just typed.  For them the guess is the
        # sensitive unknown, so point at the near-return scan instead.
        seeded = ic is not None and period_guess is not None
        lead, lines = (
            (
                "The period is what shooting is most sensitive to, so take it from "
                "the trajectory's closest near-return rather than from a guess:",
                _SEED_FROM_NEAR_RETURN,
            )
            if seeded
            else (
                "Seed it from the attractor and estimate the period from it:",
                _SEED_FROM_ATTRACTOR,
            )
        )
        raise ConvergenceError(
            f"periodic_orbits: Newton did not converge (closure residual {residual:.3e} "
            f"≥ tol {tol:.1e}), so (x0, T) is not a closed orbit. Shooting has a small "
            f"basin: it needs a starting point already close to the cycle"
            + (f", and period_guess={period_guess:g} did not put it there." if seeded else ".")
            + remedy(*lines, lead=lead)
        )

    points = _sample_cycle(rhs, x0, t_period, n_points)
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if extent < min_amplitude:
        raise ConvergenceError(
            f"periodic_orbits: shooting collapsed onto an equilibrium (orbit extent "
            f"{extent:.2e} < {min_amplitude:.1e}). Newton walked to a fixed point "
            f"because the starting guess was not near a cycle — or the system has no "
            f"isolated cycle to find (a centre is a continuum of orbits, so shooting "
            f"has nothing to converge *to*)."
            + remedy(
                "traj = system.run(final_time=200.0, dt=0.01)",
                "ts.analysis.periodic_orbits(system, "
                "float(ts.analysis.estimate_period(traj)), ic=traj.y[-1])",
                lead="Seed it from a point that is actually on the cycle:",
            )
        )

    multipliers, eigenvectors = np.linalg.eig(monodromy)
    stable = _flow_orbit_stable(multipliers, eigenvectors, rhs(x0, 0.0))
    meta = AnalysisResult.build_meta(system, analysis="periodic_orbits", period=float(t_period))
    return OrbitSet(
        items=(
            PeriodicOrbit(
                points=points,
                period=float(t_period),
                multipliers=multipliers,
                stable=stable,
                continuous=True,
                residual=residual,
                meta=meta,
            ),
        ),
        meta=meta,
    )


def _flow_orbit_stable(
    multipliers: np.ndarray, eigenvectors: np.ndarray, flow_dir: np.ndarray
) -> bool:
    """Stable iff every *non-trivial* Floquet multiplier is inside the unit circle.

    The trivial multiplier (``≈ 1``) has eigenvector along the flow direction
    ``f(x0)`` (since ``M f(x0) = f(x0)``), so it is identified by eigenvector
    *alignment* with ``f(x0)`` rather than by ``argmin|μ − 1|`` — the latter
    misfires near a bifurcation, where a non-trivial multiplier can sit closer to
    ``1`` than the trivial one.  Falls back to the ``|μ − 1|`` heuristic only when
    the flow direction is degenerate (``f(x0) ≈ 0``).
    """
    if multipliers.size <= 1:
        return True
    nf = float(np.linalg.norm(flow_dir))
    if nf > 0.0:
        fhat = (np.asarray(flow_dir, dtype=float) / nf).astype(complex)
        cols = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
        trivial = int(np.argmax(np.abs(cols.conj().T @ fhat)))
    else:
        trivial = int(np.argmin(np.abs(multipliers - 1.0)))
    nontrivial = np.delete(multipliers, trivial)
    return bool(np.all(np.abs(nontrivial) < 1.0))


def _sample_cycle(rhs: Any, x0: np.ndarray, period: float, n_points: int) -> np.ndarray:
    """Sample ``n_points`` states along one period of the converged orbit (RK4)."""
    n = max(2, int(n_points))
    h = period / (n - 1)
    pts = np.empty((n, x0.size))
    x, t = x0.copy(), 0.0
    for i in range(n):
        pts[i] = x
        x = _c.rk4_state(rhs, x, t, h)
        t += h
    return pts


def _guess_period(system: Any, x0: np.ndarray, dim: int, t_run: float = 200.0) -> float:
    """Estimate a period from a burn-in trajectory to seed shooting."""
    rhs, _ = _c.flow_fns(system)
    h = 0.01
    n_trans = int(0.5 * t_run / h)
    x, t = x0.copy(), 0.0
    for _ in range(n_trans):  # burn in onto the attractor
        x = _c.rk4_state(rhs, x, t, h)
        t += h
    n = int(0.5 * t_run / h)
    series = np.empty(n)
    var_col = (
        int(np.argmax([_component_variance(rhs, x, h, c) for c in range(dim)])) if dim > 1 else 0
    )
    for i in range(n):
        series[i] = x[var_col]
        x = _c.rk4_state(rhs, x, t, h)
        t += h
    return float(estimate_period(series, dt=h))


def _component_variance(rhs: Any, x0: np.ndarray, h: float, comp: int, n: int = 400) -> float:
    """Rough variance of one component over a short run (to pick a lively channel)."""
    vals = np.empty(n)
    x, t = x0.copy(), 0.0
    for i in range(n):
        vals[i] = x[comp]
        x = _c.rk4_state(rhs, x, t, h)
        t += h
    return float(np.var(vals))


# ── period estimation from a signal ───────────────────────────────────────────


def estimate_period(
    data: Any,
    *,
    dt: float | None = None,
    components: int | str | None = None,
    method: str = "autocorrelation",
    max_delay: int | None = None,
    detrend: bool = True,
) -> ScalarResult:
    r"""
    Estimate the dominant period of a sampled signal.

    Accepts a :class:`~tsdynamics.data.Trajectory` (the sampling step is read from
    its time grid), a 1-D array, or a 2-D array (one row per sample); for
    multi-component input, ``components`` selects the channel (default: the
    highest-variance one).

    Parameters
    ----------
    data : Trajectory or array-like
        The signal.
    dt : float, optional
        Sampling step.  For a bare array it sets the time unit (default ``1.0`` →
        period in samples).  For a Trajectory the step is read from its time grid;
        passing ``dt`` overrides that grid.
    components : int or str, optional
        Which **column** of a multi-component input to read, by index or by
        name (default: the highest-variance one).  It selects a channel, never
        a sample: ``estimate_period(traj, components=0)`` reads ``traj.y[:, 0]``.
    method : {"autocorrelation", "fft"}
        ``"autocorrelation"`` — first autocorrelation peak after the first
        zero-crossing (parabolically refined).  ``"fft"`` — reciprocal of the
        dominant spectral frequency, with the peak bin parabolically refined to
        sub-bin resolution.
    max_delay : int, optional
        Largest lag considered (``"autocorrelation"`` only); default ``len // 2``.
    detrend : bool
        Subtract the mean before estimating (default ``True``).

    Returns
    -------
    ScalarResult
        The estimated period in time units (``dt`` units); ``float(result)``
        returns the number.

    Raises
    ------
    ValueError
        If fewer than 8 samples are given, the signal is constant, the
        autocorrelation is degenerate or has no zero-crossing, the spectrum has
        no dominant frequency, or ``method`` is not ``"autocorrelation"``/
        ``"fft"``.

    Examples
    --------
    >>> estimate_period(VanDerPol().run(final_time=200, dt=0.01))   # ≈ 6.66

    References
    ----------
    Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting
    and Control*. Holden-Day (autocorrelation method).
    """
    y, step = _coerce_signal(data, dt, components)
    if y.size < 8:
        raise ValueError("estimate_period needs at least 8 samples.")
    if detrend:
        y = y - y.mean()
    if np.allclose(y, 0.0):
        raise ValueError("estimate_period: signal is constant (no period).")

    method = method.lower()
    if method == "autocorrelation":
        lag, abscissa, ordinate, curve_label = _autocorr_period_lag(y, max_delay, step)
    elif method == "fft":
        lag, abscissa, ordinate, curve_label = _fft_period_lag(y, step)
    else:
        raise ValueError(f"method must be 'autocorrelation' or 'fft', got {method!r}.")
    # The diagnostic curve (autocorrelation function or power spectrum) rides on
    # ``meta`` so :func:`period_diagnostic` can draw a ``DIAGNOSTIC_CURVE`` of how
    # the estimate was read off, without changing the result *type* (it stays a
    # plain :class:`ScalarResult`).
    return ScalarResult(
        value=float(lag * step),
        meta={
            "analysis": "estimate_period",
            "method": method,
            "period": float(lag * step),
            "curve_abscissa": abscissa,
            "curve_ordinate": ordinate,
            "curve_xlabel": curve_label[0],
            "curve_ylabel": curve_label[1],
        },
    )


def period_diagnostic(data: Any, **kwargs: Any) -> Any:
    r"""Build the period-estimation diagnostic curve as a :class:`PlotSpec`.

    Runs :func:`estimate_period` on ``data`` (forwarding any keyword arguments)
    and renders *how* the period was read off as a ``DIAGNOSTIC_CURVE``: the
    autocorrelation function (``method="autocorrelation"``) or the power spectrum
    (``method="fft"``) as a ``LINE``, with a ``"vline"`` annotation marking the
    detected period (its lag, or its frequency).  The
    :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
    plotting library.

    Parameters
    ----------
    data : Trajectory or array-like
        The signal — passed straight to :func:`estimate_period`.
    **kwargs
        Forwarded to :func:`estimate_period` (``dt`` / ``components`` /
        ``method`` / ``max_delay`` / ``detrend``).

    Returns
    -------
    PlotSpec
        A ``DIAGNOSTIC_CURVE`` of the autocorrelation / spectral diagnostic.

    Examples
    --------
    >>> spec = period_diagnostic(VanDerPol().run(final_time=200, dt=0.01))
    """
    from .. import _plotbuilder as pb

    result = estimate_period(data, **kwargs)
    meta = dict(result.meta)
    abscissa = np.asarray(meta.get("curve_abscissa", np.empty(0)), dtype=float)
    ordinate = np.asarray(meta.get("curve_ordinate", np.empty(0)), dtype=float)
    method = str(meta.get("method", "autocorrelation"))
    xlabel = str(meta.get("curve_xlabel", "lag"))
    ylabel = str(meta.get("curve_ylabel", "value"))
    period = float(result)

    # Mark the detected period: at its lag (autocorrelation) or its frequency 1/T
    # (the spectral peak the FFT picked).
    mark_x = period if method == "autocorrelation" else (1.0 / period if period else 0.0)
    return pb.spec(
        None,
        "diagnostic_curve",
        layers=[pb.line(abscissa, ordinate, label=ylabel)],
        xlabel=xlabel,
        ylabel=ylabel,
        title=f"period estimate ({method})",
        annotations=[pb.vline(mark_x, text=f"period = {period:.4g}")],
        meta={"analysis": "period_diagnostic", "method": method, "period": period},
    )


def _column(data: Any, arr: np.ndarray, components: int | str | None) -> np.ndarray:
    """Select ONE column of an ``(N, dim)`` point set, by index or by name.

    The v6 fix for the axis bug: ``data[component]`` on a
    :class:`~tsdynamics.data.Trajectory` selects a **row** (one state), not a
    channel, so ``estimate_period(traj, component=0)`` measured the period of a
    3-sample signal.  On a 2-component system that raised; on a 10-component
    one it silently returned 0.030 where the truth is 1.58.
    """
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError("estimate_period expects a 1-D series or an (N, dim) point set.")
    if components is None:
        return arr[:, 0] if arr.shape[1] == 1 else arr[:, int(np.argmax(arr.var(0)))]
    if isinstance(components, str):
        names = tuple(getattr(data, "variables", ()) or ())
        if components not in names:
            known = f" This subject's components are: {', '.join(names)}." if names else ""
            raise ValueError(f"no component named {components!r}.{known}")
        index = names.index(components)
    else:
        index = int(components)
    if not -arr.shape[1] <= index < arr.shape[1]:
        raise ValueError(
            f"components={components!r} is out of range: this subject has "
            f"{arr.shape[1]} components."
        )
    return arr[:, index]


def _coerce_signal(
    data: Any, dt: float | None, components: int | str | None
) -> tuple[np.ndarray, float]:
    """Coerce input to ``(1-D float array, sampling step)``.

    Rejects a ``System`` first: ``estimate_period`` reads a *measured signal*,
    and a system handed to it would otherwise die inside ``np.asarray``.
    """
    reject_system(data, analysis="estimate_period")
    if hasattr(data, "t") and hasattr(data, "y"):  # Trajectory (duck-typed)
        t = np.asarray(data.t, dtype=float)
        y = _column(data, np.asarray(data.y, dtype=float), components)
        step = dt if dt is not None else float(np.mean(np.diff(t))) if t.size > 1 else 1.0
        return np.ravel(y), step
    arr = _column(data, np.asarray(data, dtype=float), components)
    return np.ravel(arr), (1.0 if dt is None else float(dt))


def _autocorr_period_lag(
    y: np.ndarray, max_lag: int | None, step: float
) -> tuple[float, np.ndarray, np.ndarray, tuple[str, str]]:
    """Lag of the first autocorrelation peak past the first zero-crossing.

    Returns ``(lag, abscissa, ordinate, (xlabel, ylabel))`` — the lag (in
    samples) plus the autocorrelation diagnostic curve (lag in time units against
    the normalised autocorrelation) for :func:`period_diagnostic`.
    """
    n = y.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(y, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n].real
    if acf[0] <= 0.0:
        raise ValueError("estimate_period: degenerate autocorrelation.")
    acf = acf / acf[0]
    hi = n // 2 if max_lag is None else min(int(max_lag), n - 2)
    abscissa = np.arange(hi, dtype=float) * step
    ordinate = acf[:hi].astype(float)
    curve_label = ("lag", "autocorrelation")
    # first lag where the autocorrelation has come back up after dipping below 0
    zero = next((k for k in range(1, hi) if acf[k] < 0.0), None)
    if zero is None:
        raise ValueError(
            "estimate_period: no autocorrelation zero-crossing — signal may be "
            "aperiodic or too short for its period (try method='fft' or more data)."
        )
    peak = zero + int(np.argmax(acf[zero:hi]))
    if peak <= 0 or peak >= hi - 1:
        return float(peak), abscissa, ordinate, curve_label
    # parabolic sub-sample refinement around the peak
    a, b, c = acf[peak - 1], acf[peak], acf[peak + 1]
    denom = a - 2.0 * b + c
    shift = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    return float(peak + shift), abscissa, ordinate, curve_label


def _fft_period_lag(
    y: np.ndarray, step: float
) -> tuple[float, np.ndarray, np.ndarray, tuple[str, str]]:
    """Lag (in samples) of the dominant spectral frequency.

    Returns ``(lag, abscissa, ordinate, (xlabel, ylabel))`` — the period (in
    samples) plus the power spectrum diagnostic curve (frequency in ``1/step``
    units against spectral power) for :func:`period_diagnostic`.

    The dominant bin is refined to sub-bin resolution by a parabolic fit through
    the peak power and its two neighbours (the same interpolation the
    autocorrelation path uses), so the estimate is not pinned to the coarse
    ``n / k`` grid that a single FFT bin would give.

    A Hann (raised-cosine) taper is applied before the transform to suppress
    spectral leakage on a finite record whose true period is not an integer
    divisor of ``n``; this roughly halves the sub-bin period bias on
    non-commensurate records, at the cost of a slightly broader main lobe
    (Harris 1978, *Proc. IEEE* 66, 51 — the Hann window's leakage trade-off).
    The autocorrelation method (the default) remains preferable for very short
    records.
    """
    n = y.size
    spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    spec[0] = 0.0  # drop the DC component
    freqs = np.fft.rfftfreq(n, d=step)
    k = int(np.argmax(spec))
    if k == 0:
        raise ValueError("estimate_period: no dominant frequency found.")
    k_ref = float(k)
    if 0 < k < spec.size - 1:  # parabolic sub-bin refinement around the peak bin
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2.0 * b + c
        if denom != 0.0:
            k_ref = k + 0.5 * (a - c) / denom
    return float(n / k_ref), freqs.astype(float), spec.astype(float), ("frequency", "power")


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
