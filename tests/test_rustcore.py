"""
Numeric validation of the ``tsdynamics-core`` Rust kernels.

Skipped entirely unless the optional accelerator is installed.  These tests
prove the tape VM reproduces the symbolic RHS, that the RK4 stepper matches a
high-accuracy SciPy reference, and that the rayon ensemble runner is
deterministic and race-free (parallel == serial, bit for bit).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.backends import rustcore as rc

pytestmark = pytest.mark.skipif(
    not rc.available(), reason="tsdynamics-core (Rust accelerator) is not installed"
)

_EVAL_SAMPLE = ["Lorenz", "Rossler", "Chen", "Thomas", "Halvorsen", "SprottA"]


@pytest.mark.parametrize("name", _EVAL_SAMPLE)
def test_tape_eval_matches_symbolic_rhs(name: str) -> None:
    """The Rust tape evaluates the exact same RHS as the symbolic core."""
    sys = getattr(ts, name)()
    tape = rc.compile_tape(sys)
    f = sys._rhs_numeric()
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    for _ in range(8):
        u = rng.standard_normal(sys.dim)
        t = float(rng.uniform(0.0, 5.0))
        got = rc.eval_rhs(sys, u, t, tape=tape)
        ref = f(u, t)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("name", "ic"),
    [("Lorenz", [1.0, 1.0, 1.0]), ("Rossler", [0.1, 0.0, 0.0])],
)
def test_rk4_matches_scipy(name: str, ic: list[float]) -> None:
    """Fixed-step RK4 agrees with a high-accuracy SciPy reference."""
    from scipy.integrate import solve_ivp

    sys = getattr(ts, name)()
    t_eval = np.linspace(0.0, 8.0, 801)
    y = rc.integrate_dense(sys, ic, t_eval, h=1e-3)
    assert y.shape == (t_eval.size, sys.dim)
    f = sys._rhs_numeric()
    sol = solve_ivp(lambda t, u: f(u, t), (0.0, 8.0), ic, t_eval=t_eval, rtol=1e-11, atol=1e-13)
    # Chaotic divergence over 8 t.u. is bounded for matched truncation; require
    # tight early agreement and finiteness throughout.
    assert np.all(np.isfinite(y))
    early = t_eval <= 3.0
    np.testing.assert_allclose(y[early], sol.y.T[early], atol=1e-4)


def test_ensemble_is_deterministic_and_race_free() -> None:
    """Parallel ensemble == serial per-trajectory, bit for bit; all finite."""
    sys = ts.Lorenz()
    rng = np.random.default_rng(0)
    batch = rng.uniform(-12.0, 12.0, size=(500, 3))
    fin = rc.ensemble_final(sys, batch, 0.0, 4.0, h=2e-3)
    assert fin.shape == batch.shape
    assert np.all(np.isfinite(fin))
    # Re-running must be identical (no nondeterministic reduction).
    again = rc.ensemble_final(sys, batch, 0.0, 4.0, h=2e-3)
    np.testing.assert_array_equal(fin, again)
    # And it must equal the serial single-trajectory result for sampled rows.
    for i in (0, 17, 250, 499):
        serial = rc.integrate_dense(sys, batch[i], np.array([0.0, 4.0]), h=2e-3)[-1]
        np.testing.assert_array_equal(fin[i], serial)


def test_version_is_exposed() -> None:
    import tsdynamics_core

    assert isinstance(tsdynamics_core._version(), str)
