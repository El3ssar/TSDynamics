r"""
Regression: ``fixed_points`` on a flow must not clip equilibria to the seed box.

When no ``region`` is given, the search box is auto-derived from a short burn-in
orbit's bounding hull.  That hull is only a *seeding* aid: a flow's equilibria
are typically saddles the on-attractor orbit never visits, so they fall outside
the hull.  The defect (FIX-FPFLOW) was that converged roots were additionally
*clipped* to that auto box, silently discarding genuine equilibria.

The canonical witness is the Lorenz system, whose chaotic attractor's hull
excludes both the origin saddle ``(0, 0, 0)`` and the unstable foci

    C± = (±√(β(ρ−1)), ±√(β(ρ−1)), ρ−1)

at the centres of the two wings.  With the default parameters
(``σ=10, ρ=28, β=8/3``) that is ``C± = (±√(72), ±√(72), 27)``.  A correct
``fixed_points(Lorenz())`` with ``region=None`` must return all three.

The complementary half of the contract is that an *explicit* ``region`` is still
honoured as a hard search domain (roots outside it are clipped), so a user can
deliberately restrict the search.
"""

from __future__ import annotations

import math

import numpy as np

import tsdynamics as ts
from tsdynamics import fixed_points


def _lorenz_equilibria() -> list[np.ndarray]:
    """The three analytic Lorenz equilibria at the default parameters."""
    rho, beta = 28.0, 8.0 / 3.0
    c = math.sqrt(beta * (rho - 1.0))
    return [
        np.array([0.0, 0.0, 0.0]),
        np.array([c, c, rho - 1.0]),
        np.array([-c, -c, rho - 1.0]),
    ]


def _match(found: list[np.ndarray], target: np.ndarray, tol: float = 1e-4) -> bool:
    return any(np.linalg.norm(f - target) < tol for f in found)


class TestFlowEquilibriaNoRegion:
    def test_lorenz_returns_all_three_equilibria_without_region(self) -> None:
        """region=None must recover origin + C± — not just the on-hull subset."""
        fps = fixed_points(ts.Lorenz(), seed=0)
        coords = [fp.x for fp in fps]

        # Exactly the three analytic equilibria, all classified as flow points.
        assert len(fps) == 3
        assert all(fp.continuous for fp in fps)
        for eq in _lorenz_equilibria():
            assert _match(coords, eq), f"missing equilibrium {eq} from {coords}"

        # The origin in particular is a real saddle the chaotic orbit avoids; it
        # is the equilibrium the pre-fix box-clip dropped.
        origin = next(fp for fp in fps if np.linalg.norm(fp.x) < 1e-5)
        assert origin.eigenvalues.real.max() > 0.0

    def test_explicit_region_still_clips_roots(self) -> None:
        """An explicit region remains a hard search domain (the complement)."""
        # A box around C+ only; the origin and C- lie outside and must be clipped.
        fps = fixed_points(ts.Lorenz(), region=([5, 5, 20], [12, 12, 32]), seed=0)
        coords = [fp.x for fp in fps]
        assert len(fps) == 1
        c = math.sqrt((8.0 / 3.0) * 27.0)
        assert _match(coords, np.array([c, c, 27.0]))
