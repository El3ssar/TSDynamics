r"""
Shared plumbing for the chaos indicators.

Holds the observable coercion the 0--1 test consumes and the small
linear-algebra helpers (``_orthonormal`` initial frame, the wedge/expansion
volumes, a least-squares line fit).  The tangent-dynamics primitives GALI and
expansion entropy share (a map Jacobian/step pair and the RK4 variational step
for flows) live once in :mod:`tsdynamics.analysis._tangent` and are re-exported
here under the private names the indicators use.

Only :class:`~tsdynamics.families.DiscreteMap` (via its compiled ``_step`` /
``_jacobian``) and :class:`~tsdynamics.families.ContinuousSystem` (via its
SymEngine-lambdified numeric RHS and Jacobian) carry a finite-dimensional
tangent space here; delay and stochastic systems are rejected by the public
entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

# The family-agnostic tangent-dynamics primitives are shared with the A-FP
# stream (one home in ``analysis._tangent``); re-exported here under the private
# names the A-CHAOS indicators (gali / expansion) consume via ``_c.<name>``.
from tsdynamics.analysis._tangent import flow_fns as _flow_fns  # noqa: F401
from tsdynamics.analysis._tangent import map_fns as _map_fns  # noqa: F401
from tsdynamics.analysis._tangent import rk4_state as _rk4_state  # noqa: F401
from tsdynamics.analysis._tangent import rk4_variational as _rk4_variational  # noqa: F401
from tsdynamics.data import Box

if TYPE_CHECKING:
    import numpy.typing as npt

    from tsdynamics.families import SystemBase

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
        shape = np.shape(cast("npt.ArrayLike", data))
        raise ValueError(f"observable must be one-dimensional, got shape {shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("observable contains non-finite values (nan/inf).")
    return arr


# ── tangent dynamics ─────────────────────────────────────────────────────────
# ``_unwrap`` / ``_map_fns`` / ``_finite_diff_jac`` / ``_flow_fns`` /
# ``_rk4_state`` / ``_rk4_variational`` are re-exported from
# ``analysis._tangent`` (the shared home) at the top of this module.


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


def _resolve_region(system: SystemBase, region: object, *, margin: float = 0.1) -> Box:
    """Coerce a region argument to a :class:`~tsdynamics.data.Box`.

    Accepts a ``Box``, an ``(lo, hi)`` pair, or ``None`` — in which case the box
    is the (``margin``-expanded) bounding box of a burn-in orbit of ``system``.
    """
    if isinstance(region, Box):
        return region
    if region is not None:
        lo, hi = cast("tuple[npt.ArrayLike, npt.ArrayLike]", region)
        return Box(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    pts = _sample_orbit_box(system)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = margin * (hi - lo)
    pad = np.where(pad > 0.0, pad, 1.0)
    return Box(lo - pad, hi + pad)


def _sample_orbit_box(system: SystemBase, n: int = 2000, transient: int = 500) -> np.ndarray:
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
