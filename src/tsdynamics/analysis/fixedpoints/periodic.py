r"""
Periodic orbits of maps and flows, and period estimation from a signal.

- :func:`periodic_orbits` — period-``p`` orbits of a
  :class:`~tsdynamics.families.DiscreteMap`.  A period-``p`` orbit is a fixed
  point of the ``p``-fold composition :math:`f^{p}`, so the Schmelcher--Diakonos
  / Davidchack--Lai stabilising-transformation root finder is run on
  :math:`g(x) = f^{p}(x) - x`.  Orbits whose *minimal* period properly divides
  ``p`` are filtered out (``prime=True``) and cyclic shifts of one orbit are
  merged.
- :func:`periodic_orbit` — a periodic orbit of a
  :class:`~tsdynamics.families.ContinuousSystem` by single shooting: Newton on
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

from tsdynamics.families import ContinuousSystem, DiscreteMap

from .._result import AnalysisResult, CollectionResult, ScalarResult
from . import _common as _c
from .fixed import _build_seeds, _eigenvalue_plane_spec, _stabilising_matrices

__all__ = [
    "OrbitSet",
    "PeriodicOrbit",
    "estimate_period",
    "period_diagnostic",
    "periodic_orbit",
    "periodic_orbits",
]


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
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        pts = np.atleast_2d(np.asarray(self.points, dtype=float))
        dim = pts.shape[1] if pts.ndim == 2 and pts.size else 1
        per = f"T = {self.period:.4g}" if self.continuous else f"p = {int(self.period)}"
        title = f"{'stable' if self.stable else 'unstable'} orbit ({per})"
        mark = PlotKind.LINE if self.continuous else PlotKind.SCATTER

        if dim >= 3:
            spec_kind = PlotKind(kind) if kind is not None else PlotKind.PHASE_PORTRAIT_3D
            loop = PlotKind.LINE3D if self.continuous else PlotKind.MARKERS
            return PlotSpec(
                kind=spec_kind,
                ndim=3,
                aspect="equal",
                title=title,
                x=Axis(label="$x_0$"),
                y=Axis(label="$x_1$"),
                z=Axis(label="$x_2$"),
                layers=[
                    Layer(loop, {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}, label="orbit")
                ],
            )
        if dim == 2:
            spec_kind = PlotKind(kind) if kind is not None else PlotKind.PHASE_PORTRAIT_2D
            return PlotSpec(
                kind=spec_kind,
                ndim=2,
                aspect="equal",
                title=title,
                x=Axis(label="$x_0$"),
                y=Axis(label="$x_1$"),
                layers=[Layer(mark, {"x": pts[:, 0], "y": pts[:, 1]}, label="orbit")],
            )
        spec_kind = PlotKind(kind) if kind is not None else PlotKind.TIME_SERIES
        y = pts[:, 0] if pts.ndim == 2 else np.ravel(pts).astype(float)
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=title,
            x=Axis(label="index"),
            y=Axis(label="$x$"),
            layers=[Layer(mark, {"x": np.arange(y.size, dtype=float), "y": y}, label="orbit")],
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

    def __repr__(self) -> str:  # noqa: D105
        kind = "stable" if self.stable else "unstable"
        mu = np.abs(self.multipliers).max() if self.multipliers.size else float("nan")
        per = f"T={self.period:.6g}" if self.continuous else f"p={int(self.period)}"
        return f"PeriodicOrbit({per}, {kind}, |μ|max={mu:.4f}, n={len(self.points)})"


@dataclass(frozen=True, eq=False)
class OrbitSet(CollectionResult):
    """The set of periodic orbits found, behaving like a ``list``.

    A :class:`~tsdynamics.analysis._result.CollectionResult`: iterate it, index it
    (``orbits[0]`` is a :class:`PeriodicOrbit`), take its ``len``, and read
    :attr:`stable` / :attr:`unstable` sublists — while it carries ``.meta`` /
    ``.summary()`` / ``.to_frame()`` / the ``.plot`` seam.
    """

    @property
    def stable(self) -> list[PeriodicOrbit]:
        """The stable orbits in the set."""
        return [o for o in self.items if o.stable]

    @property
    def unstable(self) -> list[PeriodicOrbit]:
        """The unstable orbits in the set."""
        return [o for o in self.items if not o.stable]

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
        from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

        orbits = list(self.items)
        if not orbits:
            spec_kind = PlotKind(kind) if kind is not None else PlotKind.PHASE_PORTRAIT_2D
            return PlotSpec(kind=spec_kind, ndim=2, title="periodic orbits (none found)", layers=[])

        # Build each orbit's own spec (it owns the per-dim layering logic) and
        # gather the layers, relabelling each by its period/stability.
        sub = [(o, o.to_plot_spec()) for o in orbits]
        ndim = max(int(s.ndim) for _, s in sub)
        spec_kind = (
            PlotKind(kind)
            if kind is not None
            else PlotKind.PHASE_PORTRAIT_3D
            if ndim == 3
            else PlotKind.PHASE_PORTRAIT_2D
        )
        layers: list[Layer] = []
        for o, s in sub:
            per = f"T={o.period:.4g}" if o.continuous else f"p={int(o.period)}"
            label = f"{'stable' if o.stable else 'unstable'} {per}"
            for lyr in s.layers:
                layers.append(Layer(lyr.kind, dict(lyr.data), label=label, style=dict(lyr.style)))

        z = Axis(label="$x_2$") if ndim == 3 else None
        return PlotSpec(
            kind=spec_kind,
            ndim=3 if ndim == 3 else 2,
            aspect="equal",
            title=f"periodic orbits ({len(orbits)} found)",
            x=Axis(label="$x_0$"),
            y=Axis(label="$x_1$"),
            z=z,
            layers=layers,
            legend=Legend() if len(orbits) > 1 else None,
        )


# ── periodic orbits of maps ───────────────────────────────────────────────────


def periodic_orbits(
    system: Any,
    period: int,
    *,
    region: Any = None,
    n_seeds: int = 300,
    method: str = "dl",
    lam: float = 0.05,
    beta: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 200,
    dedup_tol: float = 1e-6,
    prime: bool = True,
    max_c: int | None = None,
    seed: int | None = None,
) -> OrbitSet:
    r"""
    Find period-``period`` orbits of a discrete map.

    Solves :math:`f^{p}(x) = x` by multi-start root finding (Davidchack--Lai by
    default — the stabilising transformations reach unstable orbits that plain
    Newton misses), recovers each orbit by forward iteration, filters orbits
    whose minimal period properly divides ``period`` (``prime=True``), and merges
    the cyclic shifts of one orbit.

    Parameters
    ----------
    system : DiscreteMap
        The map.
    period : int
        The period ``p`` (``p=1`` returns the fixed points as one-point orbits).
    region, n_seeds, dedup_tol, seed
        Seeding controls (see :func:`~tsdynamics.analysis.fixedpoints.fixed_points`).
    method : {"dl", "sd", "newton"}
        Root finder.  ``"dl"`` (default) = Davidchack--Lai; ``"sd"`` =
        Schmelcher--Diakonos; ``"newton"`` = plain Newton on ``f^p``.
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
    TypeError
        If ``system`` is not a :class:`~tsdynamics.families.DiscreteMap` (use
        :func:`periodic_orbit` for flows).
    ValueError
        If ``period < 1`` or ``method`` is not ``"newton"``/``"sd"``/``"dl"``.

    Examples
    --------
    >>> periodic_orbits(Logistic(params={"r": 3.2}), 2)   # the stable 2-cycle
    >>> periodic_orbits(Logistic(params={"r": 3.83}), 3)  # stable node + saddle
    """
    if not isinstance(system, DiscreteMap):
        raise TypeError("periodic_orbits is for DiscreteMap systems; use periodic_orbit for flows.")
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

    lo, hi = _c.resolve_box(system, region, dim, rng)
    seeds = _build_seeds(system, dim, lo, hi, n_seeds, rng)
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


def periodic_orbit(
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
) -> PeriodicOrbit:
    r"""
    Find a periodic orbit of an autonomous flow by single shooting.

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
    PeriodicOrbit
        With ``continuous=True``, ``period`` the converged ``T`` and ``multipliers``
        the Floquet multipliers.

    Raises
    ------
    NotImplementedError
        If ``system`` is not a continuous flow.
    ValueError
        If ``period_guess`` (or the auto-estimated period) is not positive.
    RuntimeError
        If the Newton iteration does not converge, or it collapses onto an
        equilibrium (the target may be a centre — a non-isolated orbit — rather
        than a hyperbolic cycle); try a better ``ic`` / ``period_guess``.

    Examples
    --------
    >>> periodic_orbit(VanDerPol(params={"mu": 1.0}), ic=[2.0, 0.0], period_guess=6.6)
    """
    if not isinstance(system, ContinuousSystem):
        raise NotImplementedError(
            f"periodic_orbit (shooting) is for continuous flows, not {type(system).__name__}; "
            f"use periodic_orbits for maps."
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
            raise RuntimeError(
                "periodic_orbit: singular shooting Jacobian — the target may be a "
                "centre (non-isolated orbit) or the phase condition is degenerate."
            ) from exc
        if not np.all(np.isfinite(delta)):
            raise RuntimeError("periodic_orbit: non-finite Newton step (diverged).")
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
        raise RuntimeError(
            f"periodic_orbit: Newton did not converge (residual {residual:.3e} ≥ tol {tol:.1e}); "
            f"try a better ic/period_guess or a hyperbolic orbit."
        )

    points = _sample_cycle(rhs, x0, t_period, n_points)
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if extent < min_amplitude:
        raise RuntimeError(
            f"periodic_orbit: shooting collapsed onto an equilibrium (orbit extent "
            f"{extent:.2e} < {min_amplitude:.1e}) — the target may be a centre "
            f"(non-isolated orbit), or seed a point on an actual cycle."
        )

    multipliers, eigenvectors = np.linalg.eig(monodromy)
    stable = _flow_orbit_stable(multipliers, eigenvectors, rhs(x0, 0.0))
    return PeriodicOrbit(
        points=points,
        period=float(t_period),
        multipliers=multipliers,
        stable=stable,
        continuous=True,
        residual=residual,
        meta=AnalysisResult.build_meta(system, analysis="periodic_orbit", period=float(t_period)),
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
    component: int | str | None = None,
    method: str = "autocorrelation",
    max_delay: int | None = None,
    detrend: bool = True,
) -> ScalarResult:
    r"""
    Estimate the dominant period of a sampled signal.

    Accepts a :class:`~tsdynamics.data.Trajectory` (the sampling step is read from
    its time grid), a 1-D array, or a 2-D array (one row per sample); for
    multi-component input, ``component`` selects the channel (default: the
    highest-variance one).

    Parameters
    ----------
    data : Trajectory or array-like
        The signal.
    dt : float, optional
        Sampling step.  For a bare array it sets the time unit (default ``1.0`` →
        period in samples).  For a Trajectory the step is read from its time grid;
        passing ``dt`` overrides that grid.
    component : int or str, optional
        Channel to analyse for multi-component input.
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
    >>> estimate_period(VanDerPol().integrate(final_time=200, dt=0.01))   # ≈ 6.66

    References
    ----------
    Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting
    and Control*. Holden-Day (autocorrelation method).
    """
    y, step = _coerce_signal(data, dt, component)
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
        Forwarded to :func:`estimate_period` (``dt`` / ``component`` / ``method``
        / ``max_delay`` / ``detrend``).

    Returns
    -------
    PlotSpec
        A ``DIAGNOSTIC_CURVE`` of the autocorrelation / spectral diagnostic.

    Examples
    --------
    >>> spec = period_diagnostic(VanDerPol().integrate(final_time=200, dt=0.01))
    """
    from tsdynamics.viz.spec import Annotation, Axis, Layer, PlotKind, PlotSpec

    result = estimate_period(data, **kwargs)
    meta = dict(result.meta)
    abscissa = np.asarray(meta.get("curve_abscissa", np.empty(0)), dtype=float)
    ordinate = np.asarray(meta.get("curve_ordinate", np.empty(0)), dtype=float)
    method = str(meta.get("method", "autocorrelation"))
    xlabel = str(meta.get("curve_xlabel", "lag"))
    ylabel = str(meta.get("curve_ylabel", "value"))
    period = float(result)

    layers = [Layer(PlotKind.LINE, {"x": abscissa, "y": ordinate}, label=ylabel)]
    # Mark the detected period: at its lag (autocorrelation) or its frequency 1/T
    # (the spectral peak the FFT picked).
    mark_x = period if method == "autocorrelation" else (1.0 / period if period else 0.0)
    annotations = [Annotation(kind="vline", x=mark_x, text=f"period = {period:.4g}")]
    return PlotSpec(
        kind=PlotKind.DIAGNOSTIC_CURVE,
        ndim=2,
        title=f"period estimate ({method})",
        x=Axis(label=xlabel),
        y=Axis(label=ylabel),
        layers=layers,
        annotations=annotations,
        meta={"analysis": "period_diagnostic", "method": method, "period": period},
    )


def _coerce_signal(
    data: Any, dt: float | None, component: int | str | None
) -> tuple[np.ndarray, float]:
    """Coerce input to ``(1-D float array, sampling step)``."""
    if hasattr(data, "t") and hasattr(data, "y"):  # Trajectory (duck-typed)
        t = np.asarray(data.t, dtype=float)
        if component is not None:
            y = np.asarray(data[component], dtype=float)
        else:
            ys = np.asarray(data.y, dtype=float)
            y = ys.ravel() if ys.ndim == 1 or ys.shape[1] == 1 else ys[:, int(np.argmax(ys.var(0)))]
        step = dt if dt is not None else float(np.mean(np.diff(t))) if t.size > 1 else 1.0
        return np.ravel(y), step
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        if component is not None:
            arr = arr[:, int(component)]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr[:, int(np.argmax(arr.var(0)))]
    arr = np.ravel(arr)
    if arr.ndim != 1:
        raise ValueError("estimate_period expects a 1-D series after component selection.")
    return arr, (1.0 if dt is None else float(dt))


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
    """
    n = y.size
    spec = np.abs(np.fft.rfft(y)) ** 2
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
