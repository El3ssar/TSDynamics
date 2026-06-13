"""Fixed points of discrete maps via multi-start Newton iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tsdynamics.base import DiscreteMap

__all__ = ["FixedPoint", "fixed_points"]


@dataclass(frozen=True)
class FixedPoint:
    """A fixed point of a map with its linear stability data."""

    x: np.ndarray
    eigenvalues: np.ndarray
    stable: bool  # all |λ| < 1

    def __repr__(self) -> str:
        kind = "stable" if self.stable else "unstable"
        return f"FixedPoint({np.round(self.x, 6)}, {kind}, |λ|max={np.abs(self.eigenvalues).max():.4f})"


def fixed_points(
    map_sys: Any,
    *,
    box: tuple | None = None,
    n_seeds: int = 200,
    tol: float = 1e-12,
    max_iter: int = 60,
    dedup_tol: float = 1e-6,
    seed: int | None = None,
) -> list[FixedPoint]:
    """
    Find fixed points ``f(x) = x`` of a discrete map by multi-start Newton.

    Seeds are drawn uniformly from ``box`` plus points sampled from a short
    orbit; each runs Newton on ``g(x) = f(x) − x`` with the map's Jacobian.
    Converged roots are deduplicated and classified by the eigenvalues of
    ``J(x*)`` (stable iff every ``|λ| < 1``).

    Parameters
    ----------
    map_sys : DiscreteMap
    box : ((lo...), (hi...)), optional
        Search box; defaults to the orbit's bounding box padded by 50 %,
        or ``[-2, 2]^dim`` if the orbit diverges.
    n_seeds : int
        Random seeds (orbit points are added on top).
    tol : float
        Residual tolerance ``‖f(x) − x‖``.
    max_iter : int
        Newton iterations per seed.
    dedup_tol : float
        Distance below which two roots are considered the same point.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    list[FixedPoint]
        Sorted by first coordinate.

    Examples
    --------
    >>> fps = fixed_points(Henon())
    >>> [fp.x[0] for fp in fps]   # analytic: (-0.7 ± sqrt(0.49 + 5.6)) / 2.8
    """
    if not isinstance(map_sys, DiscreteMap):
        raise TypeError("fixed_points currently supports DiscreteMap systems")

    rng = np.random.default_rng(seed)
    dim = map_sys.dim
    params = map_sys.params.as_tuple()
    step_fn = type(map_sys)._step
    jac_fn = type(map_sys)._jacobian

    def f(x: np.ndarray) -> np.ndarray:
        return np.asarray(step_fn(x, *params), dtype=float).ravel()

    def jac_at(x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(jac_fn(x, *params), dtype=float))

    # Seed pool: random box points + on-orbit points
    import contextlib

    orbit_pts = np.empty((0, dim))
    with contextlib.suppress(Exception):  # divergent maps still get box seeds
        orbit_pts = map_sys.copy().iterate(steps=100, max_retries=15).y[20:]

    if box is None:
        if orbit_pts.size:
            lo, hi = orbit_pts.min(axis=0), orbit_pts.max(axis=0)
            span = np.where(hi - lo < 1e-3, 1.0, hi - lo)
            lo, hi = lo - 0.5 * span, hi + 0.5 * span
        else:
            lo, hi = -2.0 * np.ones(dim), 2.0 * np.ones(dim)
    else:
        lo = np.asarray(box[0], dtype=float).reshape(dim)
        hi = np.asarray(box[1], dtype=float).reshape(dim)

    seeds = rng.uniform(lo, hi, size=(n_seeds, dim))
    if orbit_pts.size:
        seeds = np.vstack([seeds, orbit_pts[:: max(1, len(orbit_pts) // 20)]])

    roots: list[np.ndarray] = []
    eye = np.eye(dim)
    for x in seeds:
        x = x.copy()
        ok = False
        for _ in range(max_iter):
            try:
                gx = f(x) - x
            except Exception:  # noqa: BLE001
                break
            if not np.all(np.isfinite(gx)):
                break
            if np.linalg.norm(gx) < tol:
                ok = True
                break
            A = jac_at(x) - eye
            try:
                dx = np.linalg.solve(A, -gx)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dx)) or np.linalg.norm(dx) > 1e6:
                break
            x = x + dx
        if not ok:
            continue
        if not np.all((x >= lo - 1e-6) & (x <= hi + 1e-6)):
            continue  # outside the search box
        if any(np.linalg.norm(x - r) < dedup_tol for r in roots):
            continue
        roots.append(x)

    out = []
    for r in roots:
        eig = np.linalg.eigvals(jac_at(r))
        out.append(FixedPoint(x=r, eigenvalues=eig, stable=bool(np.all(np.abs(eig) < 1.0))))
    out.sort(key=lambda fp: tuple(fp.x))
    return out
