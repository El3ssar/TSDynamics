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
import pytest

import tsdynamics as ts
from tsdynamics.analysis import fixed_points


def _rossler_equilibria() -> list[np.ndarray]:
    """Rossler's two analytic equilibria at ``(a, b, c) = (0.2, 0.2, 5.7)``."""
    a, b, c = 0.2, 0.2, 5.7
    out = []
    for sign in (+1.0, -1.0):
        x = (c + sign * math.sqrt(c * c - 4.0 * a * b)) / 2.0
        out.append(np.array([x, -x / a, x / a]))
    return out


def _shimizu_morioka_equilibria() -> list[np.ndarray]:
    """Shimizu-Morioka's three analytic equilibria: ``(0,0,0)`` and ``(±√b, 0, 1)``."""
    b = float(ts.systems.ShimizuMorioka().params["b"])
    r = math.sqrt(b)
    return [np.zeros(3), np.array([r, 0.0, 1.0]), np.array([-r, 0.0, 1.0])]


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
        fps = fixed_points(ts.systems.Lorenz(), seed=0)
        coords = [fp.x for fp in fps.details]

        # Exactly the three analytic equilibria, all classified as flow points.
        assert len(fps) == 3
        assert all(fp.continuous for fp in fps.details)
        for eq in _lorenz_equilibria():
            assert _match(coords, eq), f"missing equilibrium {eq} from {coords}"

        # The origin in particular is a real saddle the chaotic orbit avoids; it
        # is the equilibrium the pre-fix box-clip dropped.
        origin = next(fp for fp in fps.details if np.linalg.norm(fp.x) < 1e-5)
        assert origin.eigenvalues.real.max() > 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        ("name", "equilibria"),
        [
            # Rossler x' = -y-z, y' = x+ay, z' = b + z(x-c): equilibria at
            # x = (c ± sqrt(c^2 - 4ab))/2, y = -x/a, z = x/a.  (a,b,c)=(.2,.2,5.7)
            # puts the second at (5.693, -28.465, 28.465) — far outside the
            # attractor's |y| < 12 hull, and the one v5 lost at every seed.
            ("Rossler", _rossler_equilibria()),
            # Shimizu-Morioka x'=y, y'=x(1-z)-ay, z'=x^2-bz: (0,0,0) and
            # (±sqrt(b), 0, 1).
            ("ShimizuMorioka", _shimizu_morioka_equilibria()),
        ],
    )
    def test_analytic_flow_equilibria_are_complete_without_region(self, name, equilibria, seed):
        """The auto seed box must reach every analytic equilibrium, at every seed.

        Before v6 the auto box was the hull of a **2.0-time-unit** RK4 arc padded
        by 50 %: for Rossler that is ``lo=[-1.58, -0.14, 0.03]``,
        ``hi=[1.05, 1.09, 0.04]``, an arc rather than the attractor.  Measured
        over seeds 0-7, ``fixed_points`` then returned 1 of Rossler's 2
        equilibria **every time**, 1 of HyperRossler's 2 every time, and 1-3 of
        Shimizu-Morioka's 3 depending on the seed — silently, with no warning.
        """
        fps = fixed_points(getattr(ts.systems, name)(), seed=seed)
        coords = [fp.x for fp in fps.details]
        assert len(fps) == len(equilibria)
        for eq in equilibria:
            assert _match(coords, eq, tol=1e-4), f"missing {eq} from {coords}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_hyperrossler_finds_both_equilibria_without_region(self, seed) -> None:
        """The 4-D hyperchaotic Rossler has exactly 2 equilibria (v5 found 1)."""
        fps = fixed_points(ts.systems.HyperRossler(), seed=seed)
        assert len(fps) == 2

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_chua_keeps_its_three_equilibria(self, seed) -> None:
        r"""All three of Chua's equilibria — ``0`` and ``±(1.5, 0, -1.5)`` — are found.

        This used to be the *escaping burn-in* regression: ``Chua`` declared no
        ``default_ic``, so the burn-in started off-attractor and blew up, and
        seeds drawn from that hull were far too diffuse to find the origin saddle
        (3 equilibria -> 2 on half the seeds).  Since v6 round 7 ``Chua`` starts
        inside its basin, so the hull is the attractor's — but the escape guard
        below still has to work, because a user system can start anywhere.
        """
        fps = fixed_points(ts.systems.Chua(), seed=seed)
        assert len(fps) == 3

    def test_escaping_burn_in_orbit_is_rejected(self) -> None:
        """The escape guard itself: a blow-up yields no hull, a bounded run does."""
        from tsdynamics.analysis.fixedpoints import _common as _c

        class _UnstableFocus(ts.ContinuousSystem):
            """A spiral source — every orbit but the fixed point runs away."""

            params = {"a": 0.9}
            dim = 2
            variables = ("x", "y")
            default_ic = [1.0, 0.0]

            @staticmethod
            def _equations(Y, t, *, a):
                return [a * Y(0) - Y(1), Y(0) + a * Y(1)]

        escaping = _c.sample_orbit_box(_UnstableFocus(), 2, rng=np.random.default_rng(1))
        assert escaping.size == 0
        bounded = _c.sample_orbit_box(ts.systems.Lorenz(), 3, rng=np.random.default_rng(1))
        assert bounded.shape == (_c.ORBIT_SAMPLES, 3)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_jerk_circuit_finds_its_single_equilibrium(self, seed) -> None:
        r"""``JerkCircuit`` has exactly one equilibrium, and the guard must not lose it.

        Truth is analytic and unambiguous: :math:`\dot x = y`, :math:`\dot y = z`,
        :math:`\dot z = -z - x - \epsilon(e^{y/y_0} - 1)` forces ``y = z = 0`` and
        then ``x = 0`` — the origin, and nothing else.

        It is the counter-example to a *growth-rate-only* escape test, in two
        different ways, and each one silently cost the equilibrium:

        * at ``seed=1`` the ``exp(y / 0.026)`` term detonates inside the discarded
          burn-in transient, so the orbit's last quarter is no bigger than its
          first (ratio 1.115) and it looks *settled* — at
          ``|state| = 3.5e164``, where ``np.linalg.norm`` overflows to ``inf`` and
          even ``inf > 20 * inf`` is ``False``.  Seeds drawn from that
          ``+-2.9e164`` hull found **0** equilibria.
        * at ``seed=5`` the orbit goes non-finite after 79 of the 2000 samples,
          and the partial hull (``+-1.2e8``, ratio 1.155) was likewise accepted;
          that too returned **0**.
        """
        fps = fixed_points(ts.systems.JerkCircuit(), seed=seed)
        assert len(fps) == 1
        np.testing.assert_allclose(fps.details[0].x, np.zeros(3), atol=1e-6)

    def test_a_partially_sampled_orbit_is_not_a_hull(self) -> None:
        """An orbit that goes non-finite mid-sample yields no box at all."""
        from tsdynamics.analysis.fixedpoints import _common as _c

        orbit = _c.sample_orbit_box(ts.systems.JerkCircuit(), 3, rng=np.random.default_rng(5))
        assert orbit.size == 0

    def test_explicit_region_still_clips_roots(self) -> None:
        """An explicit region remains a hard search domain (the complement)."""
        # A box around C+ only; the origin and C- lie outside and must be clipped.
        fps = fixed_points(ts.systems.Lorenz(), region=[(5, 12), (5, 12), (20, 32)], seed=0)
        coords = [fp.x for fp in fps.details]
        assert len(fps) == 1
        c = math.sqrt((8.0 / 3.0) * 27.0)
        assert _match(coords, np.array([c, c, 27.0]))
