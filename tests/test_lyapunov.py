"""
Lyapunov-spectrum tests not covered elsewhere.

ODE / map Lyapunov smoke lives in ``test_ode_lyapunov_goldens.py`` and
``test_map_systems.py``.  This module keeps **DDE** checks only (JiTCDDE compile
+ long history).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_mackeyglass_lyapunov_positive() -> None:
    """
    Mackey-Glass with the default tau=17 is in the chaotic regime; LE1 > 0.

    We first integrate from a non-trivial history to seed the trajectory on
    the attractor, then call ``lyapunov_spectrum`` with ``ic=traj.y[-1]``.
    """
    import tsdynamics as ts

    mg = ts.MackeyGlass()

    def hist(s: float) -> list[float]:
        return [1.0 + 0.1 * np.sin(0.2 * s)]

    traj = mg.integrate(final_time=200.0, dt=0.5, history=hist, rtol=1e-4, atol=1e-4)
    exps = mg.lyapunov_spectrum(
        n_exp=1,
        dt=0.5,
        burn_in=100.0,
        final_time=1000.0,
        ic=traj.y[-1],
        rtol=1e-4,
        atol=1e-4,
    )
    assert exps.shape == (1,)
    assert exps[0] > 0.0, f"MackeyGlass LE1 should be > 0, got {exps[0]:.5f}"


@pytest.mark.slow
def test_mackeyglass_lyapunov_n_exp_2() -> None:
    """Requesting two exponents returns two finite values."""
    import tsdynamics as ts

    mg = ts.MackeyGlass()
    traj = mg.integrate(
        final_time=200.0,
        dt=0.5,
        history=lambda s: [1.0 + 0.1 * np.sin(0.2 * s)],
        rtol=1e-4,
        atol=1e-4,
    )
    exps = mg.lyapunov_spectrum(
        n_exp=2,
        dt=0.5,
        burn_in=50.0,
        final_time=300.0,
        ic=traj.y[-1],
        rtol=1e-4,
        atol=1e-4,
    )
    assert exps.shape == (2,)
    assert np.all(np.isfinite(exps))
