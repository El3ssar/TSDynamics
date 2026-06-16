r"""
Shared plumbing for the chaos indicators.

Holds the observable coercion the 0--1 test consumes, the tangent-dynamics
primitives GALI and expansion entropy share (a map Jacobian/step pair and an
RK4 variational step for flows), and the small linear-algebra helpers
(``_orthonormal`` initial frame, the wedge/expansion volumes, a least-squares
line fit).

Only :class:`~tsdynamics.families.DiscreteMap` (via its compiled ``_step`` /
``_jacobian``) and :class:`~tsdynamics.families.ContinuousSystem` (via its
SymEngine-lambdified numeric RHS and Jacobian) carry a finite-dimensional
tangent space here; delay and stochastic systems are rejected by the public
entry points.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tsdynamics.data import Box

# ── observable coercion (0--1 test) ──────────────────────────────────────────


def _as_observable(data: object, component: int | None = None) -> np.ndarray:
    """Coerce a scalar observable to a 1-D ``float`` array.

    Accepts a 1-D array-like, or a :class:`~tsdynamics.data.Trajectory`
    (duck-typed via its ``.y`` attribute).  A multi-component trajectory needs a
    ``component`` index (or a pre-extracted column such as ``traj["x"]``).

    Raises
    ------
    TypeError
        If a live system (rather than a sampled series) is passed.
    ValueError
        If the series is not one-dimensional after component selection, or is
        too short to be informative.
    """
    if hasattr(data, "integrate") or hasattr(data, "_equations") or hasattr(data, "_step"):
        raise TypeError(
            "the 0-1 test consumes a sampled observable, not a live system; "
            "integrate/iterate first and pass one component, e.g. "
            "zero_one_test(sys.integrate(...).component('x'))."
        )
    y = getattr(data, "y", None)
    arr = np.asarray(y if y is not None else data, dtype=float)
    if arr.ndim == 2:
        if component is not None:
            arr = arr[:, int(component)]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            raise ValueError(
                f"observable has {arr.shape[1]} components; pass a 1-D series "
                "(e.g. traj['x']) or component=<index>."
            )
    arr = np.ravel(arr)
    if arr.ndim != 1:
        raise ValueError(f"observable must be one-dimensional, got shape {np.shape(data)}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("observable contains non-finite values (nan/inf).")
    return arr


# ── tangent dynamics: maps ───────────────────────────────────────────────────


def _unwrap(x: np.ndarray, dim: int) -> object:
    """Present the state to a compiled ``_step``/``_jacobian`` in its native form.

    One-dimensional maps are written for a scalar argument (``x = X``);
    higher-dimensional maps unpack an array (``x, y = X[0], X[1]``).
    """
    a = np.asarray(x, dtype=float).ravel()
    return float(a[0]) if dim == 1 else a


def _map_fns(system: object) -> tuple[Callable, Callable]:
    """Return ``(step, jac)`` callables for a discrete map.

    ``step(x) -> ndarray (dim,)`` advances one iteration; ``jac(x) -> ndarray
    (dim, dim)`` is the Jacobian at ``x``.  Both wrap the class's compiled
    ``_step`` / ``_jacobian`` (the same convention
    :class:`~tsdynamics.derived.TangentSystem` uses).  A map without an analytic
    ``_jacobian`` falls back to a forward finite difference of ``_step``.
    """
    cls = type(system)
    dim = int(system.dim)
    step_raw = cls._step
    jac_raw = getattr(cls, "_jacobian", None)
    params = tuple(system.params.as_tuple())

    def step(x: np.ndarray) -> np.ndarray:
        return np.asarray(step_raw(_unwrap(x, dim), *params), dtype=float).ravel()

    if jac_raw is not None:

        def jac(x: np.ndarray) -> np.ndarray:
            j = np.asarray(jac_raw(_unwrap(x, dim), *params), dtype=float)
            return np.atleast_2d(j).reshape(dim, dim)

    else:

        def jac(x: np.ndarray) -> np.ndarray:
            return _finite_diff_jac(step, x, dim)

    return step, jac


def _finite_diff_jac(step: Callable, x: np.ndarray, dim: int, eps: float = 1e-7) -> np.ndarray:
    """Forward finite-difference Jacobian of a map step (fallback only)."""
    x = np.asarray(x, dtype=float).ravel()
    base = step(x)
    out = np.empty((dim, dim))
    for j in range(dim):
        xp = x.copy()
        h = eps * (1.0 + abs(xp[j]))
        xp[j] += h
        out[:, j] = (step(xp) - base) / h
    return out


# ── tangent dynamics: flows ──────────────────────────────────────────────────


def _flow_fns(system: object) -> tuple[Callable, Callable]:
    """Return ``(rhs, jac)`` for a flow: ``rhs(u, t)`` and ``jac(u, t)``.

    Both come from the SymEngine-lambdified numeric forms
    (:meth:`ContinuousSystem._rhs_numeric` / :meth:`ContinuousSystem.jacobian`),
    so the variational integrator is self-contained (no engine tape lowering)
    and runs in the fast tier.  Parameters are captured at call time.
    """
    rhs = system._rhs_numeric()

    def jac(u: np.ndarray, t: float) -> np.ndarray:
        return system.jacobian(u, t)

    return rhs, jac


def _rk4_state(rhs: Callable, x: np.ndarray, t: float, h: float) -> np.ndarray:
    """One classic RK4 step of ``dx/dt = rhs(x, t)`` (state only; for burn-in)."""
    k1 = rhs(x, t)
    k2 = rhs(x + 0.5 * h * k1, t + 0.5 * h)
    k3 = rhs(x + 0.5 * h * k2, t + 0.5 * h)
    k4 = rhs(x + h * k3, t + h)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_variational(
    rhs: Callable, jac: Callable, x: np.ndarray, m: np.ndarray, t: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""One RK4 step of the augmented system ``dx/dt = f``, ``dM/dt = J(x) M``.

    ``M`` holds the tangent vectors as columns (shape ``(dim, ncol)``); the same
    fundamental-matrix flow drives GALI (``ncol = k``) and expansion entropy
    (``ncol = dim``).
    """
    k1x = rhs(x, t)
    k1m = jac(x, t) @ m
    x2, m2, t2 = x + 0.5 * h * k1x, m + 0.5 * h * k1m, t + 0.5 * h
    k2x = rhs(x2, t2)
    k2m = jac(x2, t2) @ m2
    x3, m3 = x + 0.5 * h * k2x, m + 0.5 * h * k2m
    k3x = rhs(x3, t2)
    k3m = jac(x3, t2) @ m3
    x4, m4, t4 = x + h * k3x, m + h * k3m, t + h
    k4x = rhs(x4, t4)
    k4m = jac(x4, t4) @ m4
    x_new = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    m_new = m + (h / 6.0) * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
    return x_new, m_new


# ── linear algebra ───────────────────────────────────────────────────────────


def _orthonormal_frame(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``k`` random orthonormal column vectors in ``R^dim`` (shape ``(dim, k)``)."""
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return np.ascontiguousarray(q[:, :k])


def _normalize_columns(m: np.ndarray) -> np.ndarray:
    """Scale each column of ``m`` to unit Euclidean norm (zero columns left as-is)."""
    norms = np.linalg.norm(m, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    return m / norms


def gali_volume(w: np.ndarray) -> float:
    r"""GALI value of unit-norm deviation columns: ``\prod_i \sigma_i`` of ``w``.

    Equals the norm of the wedge product ``\hat w_1 \wedge \dots \wedge \hat
    w_k`` (the volume of the parallelepiped the unit vectors span): ``1`` when
    orthonormal, ``\to 0`` as they align (Skokos et al. 2007).

    A collapsed frame spans zero volume.  Guard the SVD accordingly: for a
    chaotic orbit the unit columns align to within machine precision (and a
    diverged orbit makes them non-finite), where LAPACK's iterative SVD can fail
    to converge — the volume is ``0`` in either case, so return it rather than
    raise.
    """
    if not np.all(np.isfinite(w)):
        return 0.0
    try:
        s = np.linalg.svd(w, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    return float(np.prod(s))


def expansion_volume(m: np.ndarray) -> float:
    r"""Hunt--Ott ``G(M)``: the product of the singular values of ``M`` exceeding 1.

    The volume growth of the unit ball under ``M`` restricted to its expanding
    directions (``= 1`` when ``M`` is non-expanding).

    The raw fundamental matrix is not renormalised, so a long horizon can
    overflow to non-finite — that is unbounded growth, so report ``+inf`` (the
    ``ln E(t)`` fit masks non-finite samples) rather than let a degenerate SVD
    raise.
    """
    m = np.atleast_2d(m)
    if not np.all(np.isfinite(m)):
        return float("inf")
    try:
        s = np.linalg.svd(m, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("inf")
    s = s[s > 1.0]
    return float(np.prod(s)) if s.size else 1.0


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares line ``y ≈ slope·x + intercept``; return ``(slope, intercept, stderr)``.

    ``stderr`` is the standard error of the slope (``0`` for a two-point fit).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need at least two points for a line fit.")
    xm, ym = x.mean(), y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    if sxx == 0.0:
        raise ValueError("degenerate fit: all abscissae equal.")
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    intercept = float(ym - slope * xm)
    if n > 2:
        resid = y - (slope * x + intercept)
        s2 = float(np.sum(resid**2) / (n - 2))
        stderr = float(np.sqrt(s2 / sxx))
    else:
        stderr = 0.0
    return slope, intercept, stderr


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two equal-length vectors (``0`` if either is constant)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt(float(np.sum(xm * xm)) * float(np.sum(ym * ym)))
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    return float(np.sum(xm * ym) / denom)


def _resolve_region(system: object, region: object, *, margin: float = 0.1) -> Box:
    """Coerce a region argument to a :class:`~tsdynamics.data.Box`.

    Accepts a ``Box``, an ``(lo, hi)`` pair, or ``None`` — in which case the box
    is the (``margin``-expanded) bounding box of a burn-in orbit of ``system``.
    """
    if isinstance(region, Box):
        return region
    if region is not None:
        lo, hi = region
        return Box(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    pts = _sample_orbit_box(system)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = margin * (hi - lo)
    pad = np.where(pad > 0.0, pad, 1.0)
    return Box(lo - pad, hi + pad)


def _sample_orbit_box(system: object, n: int = 2000, transient: int = 500) -> np.ndarray:
    """Collect a burn-in orbit to bound an auto-region (backend-free)."""
    from tsdynamics.families import ContinuousSystem, DiscreteMap

    x = np.asarray(system.resolve_ic(None), dtype=float).ravel()
    if isinstance(system, DiscreteMap):
        step, _ = _map_fns(system)
        for _ in range(transient):
            x = step(x)
        pts = np.empty((n, x.size))
        for i in range(n):
            x = step(x)
            pts[i] = x
        return pts
    if isinstance(system, ContinuousSystem):
        rhs, _ = _flow_fns(system)
        h, t = 0.01, 0.0
        for _ in range(transient):
            x = _rk4_state(rhs, x, t, h)
            t += h
        pts = np.empty((n, x.size))
        for i in range(n):
            x = _rk4_state(rhs, x, t, h)
            t += h
            pts[i] = x
        return pts
    raise TypeError(f"cannot auto-bound a region for {type(system).__name__}.")
