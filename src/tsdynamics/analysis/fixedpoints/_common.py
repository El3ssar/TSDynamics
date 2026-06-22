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
from collections.abc import Callable
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
    the returned list (identity-first ordering, then sign flips, then
    permutations) — callers warn when they truncate.
    """
    eye = np.eye(dim)
    mats: list[np.ndarray] = []
    # Order so the most useful matrices come first: identity, then pure sign
    # flips on the identity, then the permutations with their sign flips.
    for perm in itertools.permutations(range(dim)):
        perm_mat = eye[list(perm)]
        for signs in itertools.product((1.0, -1.0), repeat=dim):
            mats.append((np.array(signs)[:, None] * perm_mat).copy())
    # Move the identity to the front (signs all +1 on the identity permutation).
    mats.sort(key=lambda c: (not np.array_equal(c, eye), _signed_perm_rank(c)))
    if max_count is not None and len(mats) > max_count:
        mats = mats[:max_count]
    return mats


def _signed_perm_rank(c: np.ndarray) -> int:
    """Return a stable ordering key: number of negative entries, then displacement."""
    neg = int((c < 0).sum())
    disp = int(np.sum(np.abs(np.argmax(np.abs(c), axis=1) - np.arange(c.shape[0]))))
    return 100 * neg + disp


# ── root finding: Newton / Schmelcher--Diakonos / Davidchack--Lai ─────────────

# The three schemes drive a residual ``g(x)`` (with Jacobian ``G(x) = Dg``) to
# zero.  For map fixed/periodic points ``g(x) = f^p(x) - x`` and ``G = Df^p - I``;
# for flow equilibria ``g(x) = f(x)`` and ``G = J(x)``.  Davidchack & Lai (1999)
# is a Newton step regularised by ``beta*‖g‖*Cᵀ``; it reduces to plain Newton at
# ``beta = 0`` and recovers Newton's quadratic rate as ``‖g‖ → 0`` (the
# regulariser self-anneals).  Schmelcher & Diakonos (1997) is the explicit-Euler
# step ``x + lambda*C*g`` on the stabilising flow.


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
                a = beta * ng * c_mat.T - jac_resid(x)
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


def resolve_box(
    system: SystemBase, region: Any, dim: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the search ``region`` to ``(lo, hi)`` arrays of length ``dim``.

    Accepts a :class:`~tsdynamics.data.Box` / :class:`~tsdynamics.data.Grid`
    (reads its ``lo`` / ``hi``), a ``(lo, hi)`` tuple, or ``None`` — which uses a
    burn-in orbit's bounding box padded by 50 % (falling back to ``[-2, 2]^dim``
    if the orbit diverges or cannot be sampled).
    """
    if region is not None:
        lo_src, hi_src = (region.lo, region.hi) if hasattr(region, "lo") else (region[0], region[1])
        lo = np.asarray(lo_src, dtype=float).reshape(dim)
        hi = np.asarray(hi_src, dtype=float).reshape(dim)
        return lo, hi
    orbit = sample_orbit_box(system, dim)
    if orbit.size:
        lo, hi = orbit.min(axis=0), orbit.max(axis=0)
        span = np.where(hi - lo < 1e-3, 1.0, hi - lo)
        return lo - 0.5 * span, hi + 0.5 * span
    return -2.0 * np.ones(dim), 2.0 * np.ones(dim)


def sample_orbit_box(system: SystemBase, dim: int, n: int = 200, transient: int = 50) -> np.ndarray:
    """Collect a short burn-in orbit to bound an auto search box (backend-free).

    Returns an empty array if the orbit cannot be produced (e.g. it diverges);
    callers fall back to a default box.
    """
    from tsdynamics.families import ContinuousSystem, DiscreteMap

    try:
        x = np.asarray(system.resolve_ic(None), dtype=float).ravel()
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
                    break
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
                    break
                pts.append(x.copy())
        else:
            return np.empty((0, dim))
    except Exception:  # noqa: BLE001
        return np.empty((0, dim))
    return np.array(pts) if pts else np.empty((0, dim))
