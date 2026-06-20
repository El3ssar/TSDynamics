r"""
Fixed points of maps and equilibria of flows.

:func:`fixed_points` finds the roots of the defining residual by multi-start
root finding and classifies their linear stability from the Jacobian spectrum:

- **maps** (:class:`~tsdynamics.families.DiscreteMap`): solve :math:`f(x) = x`;
  stable iff every multiplier :math:`|\lambda_i| < 1`.
- **flows** (:class:`~tsdynamics.families.ContinuousSystem`): solve the
  equilibrium condition :math:`f(x) = 0` on the right-hand side; stable iff every
  eigenvalue has :math:`\operatorname{Re}\lambda_i < 0`.

The default ``method="newton"`` uses the exact analytic Jacobian.  For maps,
``method="sd"`` / ``"dl"`` additionally engage the Schmelcher--Diakonos (1997) /
Davidchack--Lai (1999) stabilising transformations, which find unstable fixed
points that pure Newton can miss by cycling a set of orthogonal matrices that
turn each instability type into a contracting one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tsdynamics.families import ContinuousSystem, DiscreteMap

from . import _common as _c

__all__ = ["FixedPoint", "fixed_points"]


@dataclass(frozen=True)
class FixedPoint:
    """A fixed point (map) or equilibrium (flow) with its linear stability data.

    Attributes
    ----------
    x : ndarray
        The point, shape ``(dim,)``.
    eigenvalues : ndarray
        Eigenvalues of the Jacobian at ``x`` — map multipliers (of :math:`Df`) for
        a discrete map, or eigenvalues of the vector-field Jacobian for a flow.
    stable : bool
        For a map, ``True`` iff every ``|lambda| < 1``; for a flow, ``True`` iff
        every ``Re(lambda) < 0``.
    continuous : bool
        ``True`` for a flow equilibrium, ``False`` for a map fixed point — sets
        which stability convention ``stable`` uses.
    """

    x: np.ndarray
    eigenvalues: np.ndarray
    stable: bool
    continuous: bool = False

    def __repr__(self) -> str:  # noqa: D105
        kind = "stable" if self.stable else "unstable"
        if self.continuous:
            gauge = f"Re(λ)max={self.eigenvalues.real.max():+.4f}"
        else:
            gauge = f"|λ|max={np.abs(self.eigenvalues).max():.4f}"
        return f"FixedPoint({np.round(self.x, 6)}, {kind}, {gauge})"


def fixed_points(
    system: Any,
    *,
    box: tuple | None = None,
    n_seeds: int = 200,
    tol: float = 1e-12,
    max_iter: int = 60,
    dedup_tol: float = 1e-6,
    method: str = "newton",
    lam: float = 0.05,
    beta: float = 1.0,
    max_c: int | None = None,
    seed: int | None = None,
) -> list[FixedPoint]:
    r"""
    Find fixed points of a map (``f(x) = x``) or equilibria of a flow (``f(x) = 0``).

    Seeds are drawn uniformly from ``box`` plus points sampled from a short orbit;
    each runs the chosen root finder, and converged roots are deduplicated and
    classified by the Jacobian spectrum (maps: ``|lambda| < 1``; flows:
    ``Re lambda < 0``).

    Parameters
    ----------
    system : DiscreteMap or ContinuousSystem
        A discrete map (fixed points) or a continuous flow (equilibria).  Delay
        and stochastic systems are not supported.
    box : ((lo...), (hi...)), optional
        Search box; defaults to a burn-in orbit's bounding box padded by 50 %,
        or ``[-2, 2]^dim`` if the orbit diverges.
    n_seeds : int
        Random seeds (orbit points are added on top).
    tol : float
        Residual tolerance (``‖f(x) − x‖`` for maps, ``‖f(x)‖`` for flows).
    max_iter : int
        Root-finding iterations per seed.
    dedup_tol : float
        Distance below which two roots are merged.
    method : {"newton", "sd", "dl"}
        ``"newton"`` (default) — Newton on the exact Jacobian.  ``"sd"`` /
        ``"dl"`` — Schmelcher--Diakonos / Davidchack--Lai stabilising
        transformations (maps only) for systematically reaching unstable points.
    lam : float
        Step size of the Schmelcher--Diakonos iteration (``method="sd"``).
    beta : float
        Regularisation strength of the Davidchack--Lai iteration
        (``method="dl"``); ``beta=0`` is plain Newton, larger ``beta`` enlarges
        the basin at the cost of more iterations.
    max_c : int, optional
        Cap on the number of stabilising matrices tried (``sd``/``dl``).  The full
        set has ``2^dim · dim!`` members; if capped, a warning is emitted.
    seed : int, optional
        RNG seed for reproducible seeding.

    Returns
    -------
    list[FixedPoint]
        Sorted by coordinate.

    Examples
    --------
    >>> fixed_points(Henon())              # two saddles of the Hénon map
    >>> fixed_points(Lorenz())             # the origin and the two C± equilibria

    References
    ----------
    Schmelcher & Diakonos (1997), *Phys. Rev. Lett.* 78, 4733.
    Davidchack & Lai (1999), *Phys. Rev. E* 60, 6172.
    """
    if isinstance(system, DiscreteMap):
        continuous = False
    elif isinstance(system, ContinuousSystem):
        continuous = True
    else:
        raise NotImplementedError(
            f"fixed_points supports discrete maps and continuous flows, not "
            f"{type(system).__name__}."
        )

    method = method.lower()
    if method not in ("newton", "sd", "dl"):
        raise ValueError(f"method must be 'newton', 'sd', or 'dl', got {method!r}.")
    if continuous and method != "newton":
        raise ValueError(
            "the 'sd'/'dl' stabilising transformations target unstable orbits of "
            "maps; flow equilibria are found with method='newton' on f(x)=0."
        )

    dim = int(system.dim)
    rng = np.random.default_rng(seed)

    if continuous:
        rhs, jac = _c.flow_fns(system)
        eye = np.eye(dim)

        def residual(x: np.ndarray) -> np.ndarray:
            return rhs(x, 0.0)

        def jac_resid(x: np.ndarray) -> np.ndarray:
            return jac(x, 0.0)

        def classify(r: np.ndarray) -> FixedPoint:
            eig = np.linalg.eigvals(jac(r, 0.0))
            return FixedPoint(
                x=r, eigenvalues=eig, stable=bool(np.all(eig.real < 0.0)), continuous=True
            )
    else:
        step, jac = _c.map_fns(system)
        eye = np.eye(dim)

        def residual(x: np.ndarray) -> np.ndarray:
            return step(x) - x

        def jac_resid(x: np.ndarray) -> np.ndarray:
            return jac(x) - eye

        def classify(r: np.ndarray) -> FixedPoint:
            eig = np.linalg.eigvals(jac(r))
            return FixedPoint(
                x=r, eigenvalues=eig, stable=bool(np.all(np.abs(eig) < 1.0)), continuous=False
            )

    lo, hi = _c.resolve_box(system, box, dim, rng)
    seeds = _build_seeds(system, dim, lo, hi, n_seeds, rng)
    c_mats = _stabilising_matrices(method, dim, max_c)

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
        bounds=(lo, hi),
    )
    out = [classify(r) for r in roots]
    out.sort(key=lambda fp: tuple(fp.x))
    return out


def _build_seeds(
    system: Any,
    dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
    n_seeds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random box seeds augmented with a subsample of an on-orbit burn-in."""
    seeds = rng.uniform(lo, hi, size=(int(n_seeds), dim))
    orbit = _c.sample_orbit_box(system, dim)
    if orbit.size:
        seeds = np.vstack([seeds, orbit[:: max(1, len(orbit) // 20)]])
    return seeds


def _stabilising_matrices(method: str, dim: int, max_c: int | None) -> list[np.ndarray]:
    """Return the ``C`` set for SD/DL (empty for Newton), with a truncation warning."""
    if method == "newton":
        return []
    full = 1
    for k in range(1, dim + 1):
        full *= 2 * k
    mats = _c.signed_permutation_matrices(dim, max_c)
    if max_c is not None and full > max_c:
        import warnings

        warnings.warn(
            f"using {max_c} of {full} stabilising matrices for dim={dim}; "
            f"some unstable orbits may be missed (raise max_c to search more).",
            stacklevel=3,
        )
    return mats


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
