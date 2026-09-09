"""Curve resampling primitives for the web-export path (stream VIZ-WEB-EXPORT).

A browser is not a plot device with infinite budget: a 1e6-sample attractor
lowered vertex-for-vertex into a three.js payload is a ~26 MB JSON document (~60 MB
unrounded, and ~137 MB under the pre-v6 exporter) that no page should be asked to
load.  Capping the vertex count is therefore not an optimisation, it is a
correctness requirement for the embedding story — but *how* the cap is applied
decides whether the capped curve still looks like the attractor.

**Arc length, never stride.**  A uniform-in-*time* stride (``y[::k]``) keeps every
``k``-th sample, so it spends its budget where the trajectory is *slow* and starves
the fast, tightly-curved turns — precisely the features that carry the geometry.
Measured on the catalogue's fastest attractor (HyperQi) at a 40 000-vertex budget,
the worst-case *sagitta* (the bow of the curve off its local chord, as a fraction
of the bounding-box diagonal) is **0.245** under a stride and **0.005** under an
arc-length resample: a 52x difference between "visibly a chorded polygon" and
"reads as a smooth arc".  So this module resamples uniformly in **space**
(:func:`resample_arclength`), which gives every turn resolution proportional to the
length it occupies, and :func:`max_sagitta_ratio` is the scale-free criterion the
tests assert against.

A **point cloud** (a map's iterate set) is not a curve — its samples have no
chord to bow off, and the drawn object is a *set*, not a swept path.  Its cap is
therefore :func:`uniform_subsample_indices`: a deterministic, seeded, order-
preserving thinning that keeps the cloud's density structure.

The module is a **leaf**: it imports only NumPy (never a plotting library, never
another ``tsdynamics.viz`` module), so the threejs exporter, the docs viewer
generator and any future web front-end share one implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = [
    "DEFAULT_SAGITTA_TARGET",
    "cumulative_arclength",
    "max_sagitta_ratio",
    "resample_arclength",
    "smooth_arclength",
    "uniform_subsample_indices",
]

#: The worst-case sagitta / bounding-box-diagonal a resampled curve should stay
#: under.  The maintainer's readability rule is ``0.01``; ``0.008`` leaves headroom
#: for payload float rounding and browser rasterisation, so a zoomed outer loop
#: still reads as an arc rather than a chorded polygon.
DEFAULT_SAGITTA_TARGET = 0.008


def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    """Cumulative chord length along a polyline.

    Parameters
    ----------
    points : ndarray, shape (m, k)
        The polyline vertices (``k`` is 2 or 3 in practice, but any width works).

    Returns
    -------
    ndarray, shape (m,)
        ``s[0] == 0`` and ``s[i]`` the summed Euclidean chord length up to vertex
        ``i`` — the natural parameterisation both :func:`resample_arclength` and any
        auxiliary-channel interpolation must share.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return np.zeros(len(pts), dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_arclength(
    points: np.ndarray,
    n: int,
    *,
    channels: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Resample a polyline to ``n`` vertices equally spaced in **arc length**.

    Places the ``n`` output vertices at equal cumulative-chord-length intervals, so
    every drawn segment has (approximately) the same *spatial* length: the fast,
    tightly-curved turns get proportionally as many vertices as the slow arcs
    instead of being starved by a uniform-in-time stride.

    Any auxiliary per-vertex ``channels`` (a scalar colour field, a time channel)
    are interpolated **on the same parameterisation**, so ``positions`` and every
    channel stay index-aligned — a channel resampled independently would silently
    de-register from the geometry it colours.

    Parameters
    ----------
    points : ndarray, shape (m, k)
        The polyline to resample.
    n : int
        The target vertex count.  ``n >= m`` (or a degenerate zero-length curve)
        returns the input unchanged, so this function only ever *thins*.
    channels : mapping of str to ndarray, optional
        Per-vertex scalar fields of length ``m`` to carry through the same
        parameterisation.  A channel of the wrong length is dropped.

    Returns
    -------
    (ndarray, dict)
        The resampled ``(n, k)`` vertices and the resampled channels.
    """
    pts = np.asarray(points, dtype=float)
    chans: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=float).reshape(-1) for k, v in (channels or {}).items()
    }
    m = len(pts)
    if pts.ndim != 2 or m < 3 or n < 2 or n >= m:
        return pts, chans

    s = cumulative_arclength(pts)
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        return pts, chans

    u = np.linspace(0.0, total, int(n))
    out = np.stack([np.interp(u, s, pts[:, j]) for j in range(pts.shape[1])], axis=1)
    out_chans = {k: np.interp(u, s, v) for k, v in chans.items() if v.size == m}
    return out, out_chans


def max_sagitta_ratio(points: np.ndarray) -> float:
    """Worst-case sagitta / bounding-box diagonal over a polyline's vertex triples.

    The *sagitta* of a triple ``(p0, p1, p2)`` is the perpendicular distance of the
    middle vertex from the chord ``p0 -> p2`` — the bow of the curve off its local
    chord, i.e. exactly the geometric error a straight-line renderer commits.
    Dividing by the cloud's bounding-box diagonal makes the criterion **scale-free**,
    so one threshold (:data:`DEFAULT_SAGITTA_TARGET`) applies to every attractor
    regardless of its units.

    Works for 2-D (delay embeddings) and 3-D (flows) alike; ``0.0`` for a
    degenerate input.
    """
    y = np.asarray(points, dtype=float)
    if y.ndim != 2 or len(y) < 3:
        return 0.0
    p0, p1, p2 = y[:-2], y[1:-1], y[2:]
    chord = p2 - p0
    clen = np.linalg.norm(chord, axis=1)
    v = p1 - p0
    # |v x chord| / |chord|, computed by hand: np.cross's 2-D-vector form is
    # deprecated in NumPy 2 (and errors under the suite's filterwarnings=error).
    if y.shape[1] == 2:
        cross_norm = np.abs(v[:, 0] * chord[:, 1] - v[:, 1] * chord[:, 0])
    else:
        cross_norm = np.linalg.norm(np.cross(v, chord), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sag = np.where(clen > 0, cross_norm / clen, 0.0)
    diag = float(np.linalg.norm(y.max(axis=0) - y.min(axis=0)))
    if diag <= 0.0:
        return 0.0
    worst = float(np.nanmax(sag))
    return worst / diag


def smooth_arclength(
    points: np.ndarray,
    *,
    target: float = DEFAULT_SAGITTA_TARGET,
    nmin: int = 4000,
    nmax: int = 40000,
) -> np.ndarray:
    """Resample to the *fewest* arc-length vertices whose worst-case sagitta < ``target``.

    Because the resample is uniform in space, the sagitta of a locally circular arc
    of constant chord ``h`` scales as ``h**2`` — i.e. as ``1 / n**2`` — so growing the
    vertex count geometrically converges quickly.  Starts at ``nmin`` and grows by
    1.5x until the measured worst-case sagitta drops below ``target`` or the ``nmax``
    ceiling is reached: a slow, gently curving attractor stops at ``nmin`` and stays
    light, while a fast, tightly-curved one spends up to the cap.  The budget is
    spent where the curvature demands it, not uniformly.

    This is the *density selector* the docs viewer generator uses.  The threejs
    exporter instead applies a hard ceiling (:func:`resample_arclength` at
    ``max_points``), because it must never *grow* a curve the caller handed it.
    """
    y = np.asarray(points, dtype=float)
    if y.ndim != 2 or len(y) < 3:
        return y
    n = max(2, int(nmin))
    best, _ = _resample_up(y, min(n, int(nmax)))
    while max_sagitta_ratio(best) >= target and n < nmax:
        n = min(int(nmax), int(n * 1.5) + 1)
        best, _ = _resample_up(y, n)
    return best


def _resample_up(points: np.ndarray, n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Arc-length resample to exactly ``n`` vertices, up **or** down.

    :func:`resample_arclength` deliberately refuses to *grow* a curve (the exporter
    must never invent vertices).  :func:`smooth_arclength` genuinely needs both
    directions — its floor ``nmin`` may exceed a short input — so it goes through
    this unclamped helper.
    """
    pts = np.asarray(points, dtype=float)
    m = len(pts)
    if pts.ndim != 2 or m < 3 or n < 2:
        return pts, {}
    s = cumulative_arclength(pts)
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        return pts, {}
    u = np.linspace(0.0, total, int(n))
    out = np.stack([np.interp(u, s, pts[:, j]) for j in range(pts.shape[1])], axis=1)
    return out, {}


def uniform_subsample_indices(n: int, max_points: int, *, seed: int = 0) -> np.ndarray:
    """Deterministic, order-preserving thinning of a **point set** to ``max_points``.

    A map's iterate cloud is a *set*, not a swept curve: its samples carry no chord,
    so the arc-length criterion is meaningless for it and a *stride* would alias the
    orbit's own period (keeping every ``k``-th iterate of a period-``k``-resonant map
    collapses the cloud onto a few points).  A seeded uniform draw without
    replacement preserves the invariant density that *is* the attractor, and sorting
    the result keeps index order so a points-comet still sweeps the cloud in its
    original iteration order.

    Returns ``arange(n)`` unchanged when no thinning is needed, so the uncapped path
    is exactly the identity.
    """
    if max_points <= 0 or n <= max_points:
        return np.arange(int(n), dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(int(n), size=int(max_points), replace=False)
    idx.sort()
    return np.asarray(idx, dtype=int)
