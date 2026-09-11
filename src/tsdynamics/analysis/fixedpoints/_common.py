r"""
Shared plumbing for fixed-point and periodic-orbit detection (stream **A-FP**).

Holds the family-agnostic tangent-dynamics primitives the detectors share:

- ``map_fns`` / ``flow_fns`` — ``(step, jac)`` for a :class:`~tsdynamics.families.DiscreteMap`
  (compiled ``_step`` / ``_jacobian``) and ``(rhs, jac)`` for a
  :class:`~tsdynamics.families.ContinuousSystem` (SymEngine-lambdified numeric
  RHS / Jacobian).  Both stay engine-free, so the detectors run in the fast
  tier with no engine tape lowering.
- ``rk4_state`` / ``rk4_variational`` — the classic and the augmented
  (state ⊕ fundamental matrix) RK4 steps used by the flow shooting/monodromy
  code.
- ``map_orbit_monodromy`` — the orbit and chain-rule Jacobian of the ``p``-fold
  composition :math:`f^{p}` (a period-``p`` orbit is a fixed point of
  :math:`f^{p}`).
- ``signed_permutation_matrices`` — the hyperoctahedral set of orthogonal
  ``{-1,0,1}`` matrices that drives the Schmelcher--Diakonos / Davidchack--Lai
  stabilising transformations.
- small helpers: search-box resolution, root deduplication, a burn-in orbit
  sampler.

The family-agnostic tangent-dynamics primitives (``to_native`` / ``map_fns`` /
``finite_diff_jac`` / ``flow_fns`` / ``rk4_state`` / ``rk4_variational``) are
shared with the A-CHAOS stream and live once in
:mod:`tsdynamics.analysis._tangent`; this module re-exports them under the names
the A-FP detectors use.  Everything else here is A-FP-specific.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

# The family-agnostic tangent-dynamics primitives are shared with the A-CHAOS
# stream (one home in ``analysis._tangent``); re-exported here under the names
# the A-FP detectors already use.
from tsdynamics.analysis._tangent import (
    finite_diff_jac as finite_diff_jac,
)
from tsdynamics.analysis._tangent import (
    flow_fns as flow_fns,
)
from tsdynamics.analysis._tangent import (
    map_fns as map_fns,
)
from tsdynamics.analysis._tangent import (
    rk4_state as rk4_state,
)
from tsdynamics.analysis._tangent import (
    rk4_variational as rk4_variational,
)
from tsdynamics.analysis._tangent import (
    to_native as to_native,
)
from tsdynamics.errors import InvalidInputError, remedy

if TYPE_CHECKING:
    from tsdynamics.families import SystemBase

# ── tangent dynamics: maps ───────────────────────────────────────────────────


def map_orbit_monodromy(
    step: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    period: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Orbit and chain-rule Jacobian of the ``period``-fold composition.

    Returns ``(x_p, M, orbit)`` where ``x_p = f^period(x0)``, the monodromy
    ``M = DF^{period}(x0) = J(x_{p-1}) ... J(x_1) J(x_0)`` (Jacobian of the
    composition), and ``orbit`` is the array of the ``period`` distinct points
    ``[x_0, x_1, ..., x_{p-1}]``.
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    m = np.eye(dim)
    orbit = np.empty((period, dim))
    for k in range(period):
        orbit[k] = x
        m = jac(x) @ m
        x = step(x)
    return x, m, orbit


# ── tangent dynamics: flows ──────────────────────────────────────────────────
# ``flow_fns`` / ``rk4_state`` / ``rk4_variational`` are re-exported from
# ``analysis._tangent`` (the shared home) at the top of this module.


def flow_state(
    rhs: Callable[..., np.ndarray], x0: np.ndarray, period: float, n_steps: int
) -> np.ndarray:
    """Integrate the state only over ``period`` with ``n_steps`` RK4 steps.

    Used by the shooting line search, where the monodromy is not needed to test a
    trial step's closure residual.
    """
    h = period / n_steps
    x = np.asarray(x0, dtype=float).ravel().copy()
    t = 0.0
    for _ in range(n_steps):
        x = rk4_state(rhs, x, t, h)
        t += h
    return x


def flow_monodromy(
    rhs: Callable[..., np.ndarray],
    jac: Callable[..., np.ndarray],
    x0: np.ndarray,
    period: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Integrate state + fundamental matrix over one period.

    Returns ``(x_T, M)`` with ``x_T = phi_T(x0)`` and the monodromy
    ``M = d phi_T / d x0`` (fundamental matrix at ``t = period``, ``M(0) = I``),
    using ``n_steps`` fixed RK4 steps of size ``period / n_steps``.
    """
    dim = x0.size
    h = period / n_steps
    x = np.asarray(x0, dtype=float).ravel().copy()
    m = np.eye(dim)
    t = 0.0
    for _ in range(n_steps):
        x, m = rk4_variational(rhs, jac, x, m, t, h)
        t += h
    return x, m


# ── stabilising transformations (Schmelcher--Diakonos / Davidchack--Lai) ──────


def signed_permutation_matrices(dim: int, max_count: int | None = None) -> list[np.ndarray]:
    r"""Return the hyperoctahedral set of orthogonal ``{-1, 0, 1}`` matrices.

    Each ``C`` places a single ``±1`` in every row and column — a permutation
    ``sigma`` of the axes combined with a sign vector ``s in {±1}^dim``
    (``C[i, sigma(i)] = s[i]``).  There are ``2^dim · dim!`` of them (the
    hyperoctahedral group :math:`B_d`); Schmelcher--Diakonos and Davidchack--Lai
    cycle through this set so that, for any fixed point, at least one ``C`` makes
    the stabilised iteration locally contracting regardless of the point's
    instability type.

    For ``dim >= 4`` the count explodes (``384, 3840, ...``); ``max_count`` caps
    the returned list.  Matrices are generated lazily in priority order —
    identity first, then sign flips on the identity, then the non-identity
    permutations with their sign flips — and generation stops as soon as
    ``max_count`` matrices are collected (the full set is never materialised when
    truncated).  Callers warn when they truncate.

    Parameters
    ----------
    dim : int
        State dimension; the group has ``2^dim · dim!`` members.
    max_count : int, optional
        Cap on the number returned.  ``None`` returns the full set.

    Returns
    -------
    list of ndarray
        Up to ``max_count`` ``(dim, dim)`` signed-permutation matrices, in
        priority order (identity first).
    """
    out: list[np.ndarray] = []
    for c in _iter_signed_permutation_matrices(dim):
        out.append(c)
        if max_count is not None and len(out) >= max_count:
            break
    return out


def _signed_permutation_count(dim: int) -> int:
    """Size of the hyperoctahedral group ``B_d``: ``2^dim · dim!`` (computed, not built)."""
    count = 1 << dim
    for k in range(2, dim + 1):
        count *= k
    return count


def _iter_signed_permutation_matrices(dim: int) -> Iterator[np.ndarray]:
    """Yield the signed-permutation matrices lazily in priority order.

    Order: identity, then pure sign flips on the identity (fewest negatives
    first), then each non-identity permutation with its sign flips — so a caller
    that stops early keeps the most useful matrices.
    """
    eye = np.eye(dim)
    # Identity permutation first, then the remaining permutations in itertools'
    # lexicographic order; within each permutation, signs ordered fewest-negative
    # first so the identity matrix (all +1) leads the whole sequence.
    identity_perm = tuple(range(dim))
    perms = [identity_perm] + [p for p in itertools.permutations(range(dim)) if p != identity_perm]
    sign_vectors = sorted(
        itertools.product((1.0, -1.0), repeat=dim),
        key=lambda s: sum(1 for v in s if v < 0.0),
    )
    for perm in perms:
        perm_mat = eye[list(perm)]
        for signs in sign_vectors:
            yield (np.array(signs)[:, None] * perm_mat).copy()


# ── root finding: Newton / Schmelcher--Diakonos / Davidchack--Lai ─────────────

# The three schemes drive a residual ``g(x)`` (with Jacobian ``G(x) = Dg``) to
# zero.  For map fixed/periodic points ``g(x) = f^p(x) - x`` and ``G = Df^p - I``;
# for flow equilibria ``g(x) = f(x)`` and ``G = J(x)``.  Davidchack & Lai (1999)
# is a Newton step regularised by ``beta*‖g‖*C``; it reduces to plain Newton at
# ``beta = 0`` and recovers Newton's quadratic rate as ``‖g‖ → 0`` (the
# regulariser self-anneals).  Schmelcher & Diakonos (1997) is the explicit-Euler
# step ``x + lambda*C*g`` on the stabilising flow.  Both sweep the *same* ``C``
# list (the truncation-consistent choice — see ``converge_root``).


def converge_root(
    residual: Callable[[np.ndarray], np.ndarray],
    jac_resid: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    method: str,
    c_mat: np.ndarray | None,
    lam: float,
    beta: float,
    tol: float,
    max_iter: int,
    polish_tol: float = 1e-2,
    step_cap: float = 1e6,
) -> np.ndarray | None:
    r"""Run one Newton/SD/DL trajectory from a single seed with a single ``C``.

    Returns the converged root (residual below ``tol``) or ``None`` if the run
    diverged, hit a singular system, or ran out of iterations.  Once the residual
    drops below ``polish_tol`` the iteration switches to a plain Newton step: the
    stabilising transformation ``C`` has by then steered the iterate into the
    root's local basin, so handing off lets the linearly-convergent SD scheme (and
    DL away from the root) finish at Newton's quadratic rate — exactly the
    "polish with a few Newton steps" recommendation (Davidchack & Lai 1999).
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    for _ in range(max_iter):
        g = residual(x)
        if not np.all(np.isfinite(g)):
            return None
        ng = float(np.linalg.norm(g))
        if ng < tol:
            return x
        use_newton = method == "newton" or ng < polish_tol
        try:
            if use_newton:
                dx = np.linalg.solve(jac_resid(x), -g)
            elif method == "sd":
                assert c_mat is not None
                dx = lam * (c_mat @ g)
            elif method == "dl":
                assert c_mat is not None
                # Sweep the *same* ``C`` list SD does: over the full
                # hyperoctahedral set ``{Cᵀ} = {C}`` so the original
                # Davidchack--Lai ``Cᵀ`` and ``C`` are equivalent, but under a
                # ``max_c`` truncation ``Cᵀ`` would visit a different subset than
                # SD's ``C`` — using ``C`` keeps both schemes on the identical
                # (truncated) sweep.
                a = beta * ng * c_mat - jac_resid(x)
                dx = np.linalg.solve(a, g)
            else:  # pragma: no cover - guarded by the public entry points
                raise ValueError(f"unknown root-finding method {method!r}.")
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(dx)) or float(np.linalg.norm(dx)) > step_cap:
            return None
        x = x + dx
    return x if float(np.linalg.norm(residual(x))) < tol else None


def solve_roots(
    residual: Callable[[np.ndarray], np.ndarray],
    jac_resid: Callable[[np.ndarray], np.ndarray],
    dim: int,
    seeds: np.ndarray,
    *,
    method: str,
    c_mats: list[np.ndarray],
    lam: float,
    beta: float,
    tol: float,
    max_iter: int,
    dedup_tol: float,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Multi-start root search; returns deduplicated roots (optionally box-clipped).

    Newton uses each seed once; the stabilising-transformation methods (``sd`` /
    ``dl``) try every ``C`` in ``c_mats`` from each seed, so any orbit whose
    instability type is stabilised by *some* ``C`` is found.
    """
    mats: list[np.ndarray | None] = [None] if method == "newton" else list(c_mats)
    roots: list[np.ndarray] = []
    for c_mat in mats:
        for seed in seeds:
            x = converge_root(
                residual,
                jac_resid,
                seed,
                method=method,
                c_mat=c_mat,
                lam=lam,
                beta=beta,
                tol=tol,
                max_iter=max_iter,
            )
            if x is None:
                continue
            if bounds is not None:
                lo, hi = bounds
                if not np.all((x >= lo - 1e-6) & (x <= hi + 1e-6)):
                    continue
            if any(np.linalg.norm(x - r) < dedup_tol for r in roots):
                continue
            roots.append(x)
    return roots


# ── geometry: search box, deduplication, burn-in orbit ────────────────────────


def dedup_points(points: list[np.ndarray], tol: float) -> list[np.ndarray]:
    """Greedily merge points closer than ``tol`` (keeps the first of each cluster)."""
    kept: list[np.ndarray] = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in kept):
            kept.append(p)
    return kept


#: Samples kept from the burn-in orbit that seeds the automatic search box, and
#: the discarded transient before them.  At the fixed ``h = 0.01`` RK4 step this
#: is **20 time units** of flow after a 5-unit transient (a map takes the same
#: counts in iterations).  The v5 values were ``200`` / ``50`` — 2.0 time units,
#: which for anything slower than Lorenz is a short *arc*, not the attractor:
#: Rossler (period ~6) produced the hull ``lo=[-1.58, -0.14, 0.03]``,
#: ``hi=[1.05, 1.09, 0.04]`` and ``fixed_points`` then found 1 of its 2
#: equilibria at every seed, silently.
#:
#: **This is a fixed budget on purpose; an adaptive one was tried and rejected.**
#: The measurements, so the next reader does not have to redo them:
#:
#: * cost — the burn-in is **0.11 s** for Lorenz (dim 3) and **0.13 s** for
#:   KuramotoSivashinsky (dim 32), against 0.011 s / 0.013 s at the v5 counts.
#:   It is Python-loop bound, not dimension bound (a 10x jump in ``dim`` costs
#:   16 %), so "scale the budget with ``dim``" is not the lever it looks like.
#: * it is not the bottleneck — 25 ``fixed_points`` calls over five catalogue
#:   flows take **11.4 s** at ``200``/``50`` and **10.2 s** at ``2000``/``500``.
#:   The short budget is *slower end to end*: a poorer box costs more in
#:   root-finding than the shorter orbit saves.
#: * "stop when the hull stops growing" is **unsafe** — a chaotic hull grows in
#:   bursts.  Rossler's is unchanged to 0.3 % of a span from sample 100 to 1200,
#:   then grows **0.92 spans** between 1600 and 2000; any patience rule stops in
#:   that quiet stretch, and ``fixed_points(Rossler())`` then finds 1 of its 2
#:   equilibria (measured over seeds 0-4 at ``n <= 1200``).
#: * "stop when the state stops moving" is **not exact** — the natural relative
#:   test floors its scale at 1, so on an orbit converging to the origin it fires
#:   while the state is still travelling: on ``x' = -x, y' = -2y`` it fires at
#:   sample 1803 of 2000 with ``8.5e-11`` of travel still to come, i.e. it moves
#:   the hull for a 9 % saving on the few systems where it fires at all.
#:
#: The honest lever, if this ever does become hot, is to run the burn-in on the
#: Rust engine instead of this pure-Python RK4 loop — which changes the sampled
#: orbit and so every seed, and needs its own validation pass.
ORBIT_SAMPLES = 2000
ORBIT_TRANSIENT = 500

#: Fractional padding of the orbit hull for the *outer* seed box, and the share
#: of ``n_seeds`` spent on it.  The **inner** box is the bare hull (pad ``0``).
#:
#: The two boxes answer the two places a flow's equilibria are found, and the
#: split is what keeps both reachable at one seed budget:
#:
#: * **inside the hull** — Thomas has 27 equilibria, Lorenz 3 and Chua 3, all
#:   within the attractor's own bounding box.  Density here is the binding
#:   constraint, so the bare hull gets the full ``n_seeds``.
#: * **outside it** — Rossler's second equilibrium is at
#:   ``(5.69, -28.47, 28.47)`` while its attractor never leaves ``|y| < 12``.
#:   The padded box reaches those for ``HULL_PAD_FRACTION`` of the budget.
#:
#: **Padding is expensive and buys little, so it is kept small.** A box padded by
#: ``p`` spans has ``(1 + 2p)**dim`` times the volume of the hull — at ``p = 4``
#: that is 729x in 3-D and ``9**32`` in 32-D, i.e. the seeds land nowhere near
#: anything.  Measured over 17 catalogue flows x 3-4 seeds: an extra box at
#: ``p = 4`` (a ``WIDE_PAD`` that used to live here) changed **no** count on any
#: system, including the Rossler case it was introduced for — the long burn-in
#: hull already reaches that equilibrium through ``p = 0.5``.  Spending the same
#: seeds on the bare hull instead takes Thomas from 19/23/19 recovered to
#: **27/27/27** (the rigorous Krawczyk count) and lifts KuramotoSivashinsky
#: (dim 32) off zero, while Lorenz / Rossler / Chua / Halvorsen / Aizawa /
#: RabinovichFabrikant / Dadras / ChenLee / … are unchanged, at 11 % less wall
#: time.
HULL_PAD = 0.5

#: Outer-box seeds as a fraction of ``n_seeds`` (see :data:`HULL_PAD`).  Total
#: seed count is unchanged from the two-box scheme this replaced: ``1.5 *
#: n_seeds`` plus 20 on-orbit points.
HULL_PAD_FRACTION = 0.5


def hull_box(orbit: np.ndarray, dim: int, pad: float) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of ``orbit`` grown by ``pad`` spans on each side.

    Falls back to ``[-2, 2]^dim`` for an empty orbit (one that diverged or could
    not be sampled).  A degenerate axis (span below ``1e-3``, e.g. a coordinate
    that is constant along the orbit) is given unit span so the box is never
    flat.
    """
    if orbit.size == 0:
        return -2.0 * np.ones(dim), 2.0 * np.ones(dim)
    lo, hi = orbit.min(axis=0), orbit.max(axis=0)
    span = np.where(hi - lo < 1e-3, 1.0, hi - lo)
    return lo - pad * span, hi + pad * span


def resolve_box(
    system: SystemBase, region: Any, dim: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the search ``region`` to ``(lo, hi)`` arrays of length ``dim``.

    Reads the region through :func:`tsdynamics.data.as_region` — the library's
    one region grammar, **one ``(lo, hi)`` bound per state component** — so
    ``region=[(-3, 3), (-3, 3)]`` searches the box it looks like it searches.
    A :class:`~tsdynamics.data.Box` / :class:`~tsdynamics.data.Ball` /
    :class:`~tsdynamics.data.Grid` is accepted unchanged, and ``None`` uses the
    burn-in orbit's bounding box padded by :data:`HULL_PAD` (falling back to
    ``[-2, 2]^dim`` if the orbit diverges or cannot be sampled).
    """
    if region is not None:
        from tsdynamics.data import Ball, as_region

        resolved = as_region(region, dim=dim, analysis="fixed_points", system=system)
        if isinstance(resolved, Ball):
            lo = np.asarray(resolved.center - resolved.r, dtype=float)
            hi = np.asarray(resolved.center + resolved.r, dtype=float)
        else:
            lo = np.asarray(resolved.lo, dtype=float)
            hi = np.asarray(resolved.hi, dtype=float)
        if lo.size != dim:
            raise InvalidInputError(
                f"{type(system).__name__} has {dim} state components, so the search "
                f"region needs {dim} per-axis bounds — got {lo.size}."
                + remedy(
                    "ts.analysis.fixed_points(system, region=["
                    + ", ".join(["(-2.0, 2.0)"] * dim)
                    + "])",
                    lead="Pass one (lo, hi) bound per state component:",
                )
            )
        return lo.reshape(dim), hi.reshape(dim)
    return hull_box(sample_orbit_box(system, dim, rng=rng), dim, HULL_PAD)


def _orbit_start_ic(system: SystemBase, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Resolve a starting state for the burn-in orbit from a *local* ``rng``.

    Mirrors :meth:`~tsdynamics.families.SystemBase.resolve_ic`'s priority — an
    explicitly stored ``system.ic`` first, then the class ``default_ic`` — but the
    final random fallback draws from the supplied seeded ``Generator`` instead of
    the process-global ``numpy.random`` state, so a caller's ``seed=`` fully
    determines the sampling regardless of global RNG history (issue #487).
    """
    if system.ic is not None:
        return np.asarray(system.ic, dtype=float).reshape(dim)
    declared = type(system)._default_ic
    if declared is not None:
        return np.asarray(declared, dtype=float).reshape(dim)
    return rng.random(dim)


def sample_orbit_box(
    system: SystemBase,
    dim: int,
    n: int = ORBIT_SAMPLES,
    transient: int = ORBIT_TRANSIENT,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Collect a burn-in orbit to bound an auto search box (backend-free).

    Returns an empty array when the orbit does not settle on a bounded set — it
    went non-finite, raised, **or** is still escaping at the end
    (:func:`_orbit_escapes`) — and callers then fall back to a default box.  The
    starting state is resolved through :func:`_orbit_start_ic`, so its random
    fallback draws from the seeded ``rng`` rather than the process-global
    ``numpy.random`` state (issue #487).
    """
    from tsdynamics.families import ContinuousSystem, DiscreteMap

    try:
        x = np.asarray(_orbit_start_ic(system, dim, rng), dtype=float).ravel()
    except Exception:  # noqa: BLE001
        return np.empty((0, dim))
    pts: list[np.ndarray] = []
    try:
        if isinstance(system, DiscreteMap):
            step, _ = map_fns(system)
            for _ in range(transient):
                x = step(x)
            for _ in range(n):
                x = step(x)
                if not np.all(np.isfinite(x)):
                    # The orbit blew up mid-sample: it demonstrably does not
                    # settle, so the partial hull is an escape trajectory's, not
                    # an attractor's.  Keeping it is how ``JerkCircuit`` produced
                    # a "settled" box of +-1.2e8 from 79 samples (seed 5) and
                    # ``fixed_points`` then missed its single equilibrium.
                    return np.empty((0, dim))
                pts.append(x.copy())
        elif isinstance(system, ContinuousSystem):
            rhs, _ = flow_fns(system)
            h, t = 0.01, 0.0
            for _ in range(transient):
                x = rk4_state(rhs, x, t, h)
                t += h
            for _ in range(n):
                x = rk4_state(rhs, x, t, h)
                t += h
                if not np.all(np.isfinite(x)):
                    return np.empty((0, dim))  # blew up mid-sample; see above
                pts.append(x.copy())
        else:
            return np.empty((0, dim))
    except Exception:  # noqa: BLE001
        return np.empty((0, dim))
    if not pts:
        return np.empty((0, dim))
    orbit = np.array(pts)
    return np.empty((0, dim)) if _orbit_escapes(orbit) else orbit


#: An orbit whose last quarter reaches more than this many times the magnitude of
#: its first quarter is still growing: it has not settled on a bounded set, so its
#: bounding box describes an escape trajectory, not an attractor.  Measured over
#: the catalogue flows, the separation is wide and unambiguous — every bounded
#: system sits at a ratio of 0.55–6.4 (Lorenz 1.3, Thomas 1.0, Rossler 6.3 while
#: its transient is still settling) while Chua from a random off-attractor start
#: sits at 87–109.  The consequence of missing this: a 20-time-unit Chua escape
#: gives the hull ``|x| < 3.5e3``, ``|z| < 7.7e3``, whose seeds are so diffuse
#: that ``fixed_points`` loses the origin equilibrium (3 -> 2) — the shorter v5
#: orbit only avoided that by stopping before the blow-up became visible.
ESCAPE_RATIO = 20.0

#: Absolute magnitude beyond which an orbit is an escape whatever its *growth
#: rate* says.  :data:`ESCAPE_RATIO` compares the orbit's last quarter with its
#: first, so it is blind to a blow-up that finishes inside the **discarded
#: transient**: both quarters are then equally astronomical and the ratio is
#: ``~1``.  ``JerkCircuit`` does exactly that — its ``exp(y / 0.026)`` term
#: detonates within the burn-in, giving a "settled" orbit at ``|state| = 3.5e164``
#: (ratio 1.115, and ``np.linalg.norm`` overflowing to ``inf`` so that even
#: ``inf > 20 * inf`` is ``False``).  The resulting hull spans ``+-2.9e164``, and
#: ``fixed_points(JerkCircuit(), seed=1)`` returned **0** equilibria where the
#: truth is exactly 1 (the origin: ``y = z = 0`` forces ``x = 0``).
#:
#: The threshold has ~10 orders of margin on both sides: over the catalogue
#: flows x 3 seeds the largest *genuine* attractor magnitude is 6.3e2
#: (``WindmiReduced``), with a median of 10.6, while the one real escape is
#: 3.5e164.
ESCAPE_MAGNITUDE = 1e12


def _orbit_escapes(orbit: np.ndarray) -> bool:
    """Whether ``orbit`` failed to settle on a bounded set.

    Two independent tests, because neither alone is sufficient (see
    :data:`ESCAPE_RATIO` and :data:`ESCAPE_MAGNITUDE`): the orbit is still
    *growing* at its end, or it is simply *enormous*.  The magnitude test is
    taken in the overflow-free sup norm, so an orbit that has already run past
    ``sqrt(f64::MAX)`` is still measured rather than collapsing to ``inf``.
    """
    q = orbit.shape[0] // 4
    if q < 1:
        return False
    if not np.all(np.isfinite(orbit)) or float(np.max(np.abs(orbit))) > ESCAPE_MAGNITUDE:
        return True
    mag = np.linalg.norm(orbit, axis=1)
    head = float(mag[:q].max())
    tail = float(mag[-q:].max())
    return tail > ESCAPE_RATIO * max(head, np.finfo(float).tiny)
