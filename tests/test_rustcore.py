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
from tsdynamics.base.ode_base import ContinuousSystem

pytestmark = pytest.mark.skipif(
    not rc.available(), reason="tsdynamics-core (Rust accelerator) is not installed"
)

_EVAL_SAMPLE = ["Lorenz", "Rossler", "Chen", "Thomas", "Halvorsen", "SprottA"]


# Tiny user-defined systems for the robustness/regression tests below. They
# register as non-builtin, so the builtin-only catalogue sweeps never see them.
class _Blowup(ContinuousSystem):
    """du/dt = u² — finite-time blow-up from u0 = 1 near t = 1."""

    params: dict = {}
    dim = 1

    @staticmethod
    def _equations(y, t):
        return [y(0) ** 2]


class _Twin(ContinuousSystem):
    """Both components are the bare symbol u0 → the tape CSEs to n_reg = 1 < dim."""

    params: dict = {}
    dim = 2

    @staticmethod
    def _equations(y, t):
        return [y(0), y(0)]


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
    y = rc.integrate_dense(sys, ic, t_eval, method="RK4", h=1e-3)
    assert y.shape == (t_eval.size, sys.dim)
    f = sys._rhs_numeric()
    sol = solve_ivp(lambda t, u: f(u, t), (0.0, 8.0), ic, t_eval=t_eval, rtol=1e-11, atol=1e-13)
    # Chaotic divergence over 8 t.u. is bounded for matched truncation; require
    # tight early agreement and finiteness throughout.
    assert np.all(np.isfinite(y))
    early = t_eval <= 3.0
    np.testing.assert_allclose(y[early], sol.y.T[early], atol=1e-4)


@pytest.mark.parametrize(
    ("name", "ic"),
    [("Lorenz", [1.0, 1.0, 1.0]), ("Rossler", [1.0, 1.0, 1.0]), ("Thomas", [0.1, 0.0, 0.0])],
)
def test_rk45_matches_scipy(name: str, ic: list[float]) -> None:
    """Adaptive Dormand-Prince agrees with SciPy's RK45 at the same tolerance."""
    from scipy.integrate import solve_ivp

    sys = getattr(ts, name)()
    rtol, atol = 1e-8, 1e-10
    t_eval = np.linspace(0.0, 5.0, 501)
    y = rc.integrate_dense(sys, ic, t_eval, method="RK45", rtol=rtol, atol=atol)
    assert y.shape == (t_eval.size, sys.dim)
    assert np.all(np.isfinite(y))
    f = sys._rhs_numeric()
    ref = solve_ivp(
        lambda t, u: f(u, t), (0.0, 5.0), ic, t_eval=t_eval, rtol=rtol, atol=atol, method="RK45"
    ).y.T
    # Two correct integrators on a chaotic flow diverge slowly; require tight
    # agreement in the early window where Lyapunov amplification is small.
    early = t_eval <= 2.0
    np.testing.assert_allclose(y[early], ref[early], atol=1e-3)


def test_rk45_converges_with_tolerance() -> None:
    """Tightening the tolerance drives the error toward a high-accuracy reference."""
    from scipy.integrate import solve_ivp

    sys = ts.Lorenz()
    ic = [1.0, 1.0, 1.0]
    t_eval = np.linspace(0.0, 3.0, 301)
    f = sys._rhs_numeric()
    ref = solve_ivp(
        lambda t, u: f(u, t), (0.0, 3.0), ic, t_eval=t_eval, rtol=1e-12, atol=1e-14, method="DOP853"
    ).y.T
    err_loose = np.max(np.abs(rc.integrate_dense(sys, ic, t_eval, rtol=1e-6, atol=1e-9) - ref))
    err_tight = np.max(np.abs(rc.integrate_dense(sys, ic, t_eval, rtol=1e-10, atol=1e-12) - ref))
    assert err_tight < err_loose
    assert err_tight < 1e-4  # adaptive at rtol=1e-10 tracks the reference closely


def test_adaptive_ensemble_deterministic_and_matches_serial() -> None:
    """Adaptive ensemble == serial adaptive, bit for bit, and is reproducible."""
    sys = ts.Lorenz()
    rng = np.random.default_rng(3)
    batch = rng.uniform(-12.0, 12.0, size=(400, 3))
    fin = rc.ensemble_final(sys, batch, 0.0, 4.0, method="RK45", rtol=1e-8, atol=1e-10)
    assert fin.shape == batch.shape and np.all(np.isfinite(fin))
    np.testing.assert_array_equal(
        fin, rc.ensemble_final(sys, batch, 0.0, 4.0, method="RK45", rtol=1e-8, atol=1e-10)
    )
    for i in (0, 200, 399):
        serial = rc.integrate_dense(
            sys, batch[i], np.array([0.0, 4.0]), method="RK45", rtol=1e-8, atol=1e-10
        )[-1]
        np.testing.assert_array_equal(fin[i], serial)


def test_ensemble_is_deterministic_and_race_free() -> None:
    """Fixed-step parallel ensemble == serial per-trajectory, bit for bit; all finite."""
    sys = ts.Lorenz()
    rng = np.random.default_rng(0)
    batch = rng.uniform(-12.0, 12.0, size=(500, 3))
    fin = rc.ensemble_final(sys, batch, 0.0, 4.0, method="RK4", h=2e-3)
    assert fin.shape == batch.shape
    assert np.all(np.isfinite(fin))
    # Re-running must be identical (no nondeterministic reduction).
    again = rc.ensemble_final(sys, batch, 0.0, 4.0, method="RK4", h=2e-3)
    np.testing.assert_array_equal(fin, again)
    # And it must equal the serial single-trajectory result for sampled rows.
    for i in (0, 17, 250, 499):
        serial = rc.integrate_dense(sys, batch[i], np.array([0.0, 4.0]), method="RK4", h=2e-3)[-1]
        np.testing.assert_array_equal(fin[i], serial)


# ---------------------------------------------------------------------------
# Robustness / regression — failure modes an adversarial review surfaced.
# Each must terminate (no infinite loop) and signal cleanly, not crash or lie.
# ---------------------------------------------------------------------------


def test_divergent_trajectory_raises_not_hangs() -> None:
    """A blow-up must raise (not spin forever on a NaN error norm)."""
    with pytest.raises(RuntimeError):
        rc.integrate_dense(_Blowup(), [1.0], np.linspace(0.0, 5.0, 101), method="RK45")


def test_ensemble_divergent_ics_yield_nan_rows() -> None:
    """Diverging ICs become NaN rows; the batch finishes and finite ICs survive."""
    batch = np.array([[0.1], [0.2], [5.0], [10.0]])
    fin = rc.ensemble_final(_Blowup(), batch, 0.0, 3.0, method="RK45")
    assert np.isfinite(fin[0, 0]) and np.isfinite(fin[1, 0])
    assert np.isnan(fin[2, 0]) and np.isnan(fin[3, 0])


def test_degenerate_tape_nreg_below_dim_does_not_panic() -> None:
    """A tape with fewer registers than state components must integrate, not panic."""
    y = rc.integrate_dense(_Twin(), [1.0, 1.0], np.linspace(0.0, 1.0, 11), method="RK45")
    assert np.all(np.isfinite(y))
    # Analytic: u0 = e^t, u1 = 1 + (e^t − 1) = e^t.
    np.testing.assert_allclose(y[-1], [np.e, np.e], rtol=1e-4)


def test_noncontiguous_inputs_are_accepted() -> None:
    """Strided (non-C-contiguous) ic / t_eval must be coerced, not panic."""
    lor = ts.Lorenz()
    t_eval = np.linspace(0.0, 2.0, 41)[::2]  # strided view
    ic = np.array([1.0, 1.0, 1.0, 9.0])[:3]  # strided-ish view
    y = rc.integrate_dense(lor, ic, t_eval, method="RK45")
    assert y.shape == (t_eval.size, 3) and np.all(np.isfinite(y))


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        rc.integrate_dense(ts.Lorenz(), [1.0, 1.0, 1.0], np.linspace(0, 1, 5), method="DOP853")


def test_dimension_mismatch_raises() -> None:
    lor = ts.Lorenz()
    with pytest.raises(ValueError):
        rc.integrate_dense(lor, [1.0, 1.0], np.linspace(0, 1, 5))
    with pytest.raises(ValueError):
        rc.ensemble_final(lor, np.zeros((5, 2)), 0.0, 1.0)


@pytest.mark.parametrize("method", ["RK45", "RK4"])
def test_empty_t_eval_returns_empty(method: str) -> None:
    y = rc.integrate_dense(ts.Lorenz(), [1.0, 1.0, 1.0], np.array([]), method=method)
    assert y.shape[0] == 0


def test_version_is_exposed() -> None:
    import tsdynamics_core

    assert isinstance(tsdynamics_core._version(), str)
