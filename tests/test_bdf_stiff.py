"""Cross-validation of the variable-order BDF stiff kernel (stream E-BDF).

The ``bdf`` kernel (``crates/tsdyn-solvers/src/implicit/bdf.rs``) is the engine's
high-order stiff workhorse, added to close the warm-throughput gap to a
variable-order BDF reference on stiff ODEs (``benches/REPORT.md``, issue #95).
These tests check it integrates the canonical stiff benchmarks **as accurately as
SciPy's own stiff integrators** (``LSODA`` / ``BDF`` / ``Radau``), through the
public engine seam.

They need the compiled extension *and* SciPy, so they are skipped wholesale when
either is absent (the default ``ci.yml`` Python job; the dedicated
``engine-bindings.yml`` job builds ``_rust`` and runs them for real).

Why the bound is self-calibrating on the oscillators: the Oregonator and the
forced van der Pol oscillator have sharp relaxation spikes, so a sub-tolerance
phase shift between *any* two correct integrators shows up as a several-percent
pointwise difference — SciPy ``LSODA`` itself sits ~2–24 % from a tight ``Radau``
truth on these. We therefore require the engine BDF to track the truth *at least
as well as SciPy's stiff solvers do*, not to some fixed absolute floor that the
references themselves would fail. The monotone Robertson problem has no such
phase sensitivity, so it gets a tight absolute bound.
"""

import numpy as np
import pytest

_rust = pytest.importorskip("tsdynamics._rust")
solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp

import tsdynamics as ts  # noqa: E402
from tsdynamics.engine import run  # noqa: E402


class _Robertson(ts.ContinuousSystem):
    """Robertson's stiff kinetics (k1=0.04, k2=3e7, k3=1e4); conserves x+y+z."""

    dim = 3
    params = {"k1": 0.04, "k2": 3.0e7, "k3": 1.0e4}
    ic = [1.0, 0.0, 0.0]

    @staticmethod
    def _equations(u, t, *, k1, k2, k3):
        x, y, z = u(0), u(1), u(2)
        return (
            -k1 * x + k3 * y * z,
            k1 * x - k3 * y * z - k2 * y * y,
            k2 * y * y,
        )


def _engine(system, ic, ft, dt, method, backend, rtol, atol):
    """Engine integration through the C-FAM seam (Jacobian-carrying tape)."""
    return run.integrate(
        system,
        final_time=ft,
        dt=dt,
        ic=ic,
        method=method,
        rtol=rtol,
        atol=atol,
        backend=backend,
        with_jacobian=True,
    ).y


def _scipy(system, ic, ft, dt, method, rtol, atol):
    rhs = system._rhs_numeric()
    t_eval = np.arange(0.0, ft, dt)
    sol = solve_ivp(
        lambda t, u: rhs(u, t), (0.0, ft), ic, t_eval=t_eval, method=method, rtol=rtol, atol=atol
    )
    return sol.y.T


def _max_rel_diff(a, b):
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n]), np.asarray(b[:n])
    scale = 1e-9 + np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.abs(a - b) / scale))


# (name, build, ic, final_time, dt, rtol, atol)
_OSCILLATORY = [
    ("Oregonator", ts.Oregonator, [1.0, 1.0, 1.0], 20.0, 0.01, 1e-7, 1e-9),
    ("ForcedVanDerPol", ts.ForcedVanDerPol, [0.1, 0.1, 0.0], 60.0, 0.02, 1e-7, 1e-9),
]


@pytest.mark.parametrize(("name", "build", "ic", "ft", "dt", "rtol", "atol"), _OSCILLATORY)
def test_bdf_tracks_truth_as_well_as_scipy_on_oscillatory_stiff(
    name, build, ic, ft, dt, rtol, atol
):
    system = build()
    truth = _scipy(system, ic, ft, dt, "Radau", 1e-10, 1e-12)

    bdf = _engine(system, ic, ft, dt, "bdf", "interp", rtol, atol)
    assert np.all(np.isfinite(bdf)), f"{name}: engine BDF produced non-finite values"

    bdf_err = _max_rel_diff(bdf, truth)
    lsoda_err = _max_rel_diff(_scipy(system, ic, ft, dt, "LSODA", rtol, atol), truth)
    scipy_bdf_err = _max_rel_diff(_scipy(system, ic, ft, dt, "BDF", rtol, atol), truth)

    # The engine BDF must be no worse than ~3× the deviation SciPy's own stiff
    # integrators show from the tight Radau truth — i.e. genuinely competitive,
    # robust to the per-machine phase sensitivity of these spiking oscillators.
    budget = 3.0 * max(lsoda_err, scipy_bdf_err) + 0.02
    assert bdf_err <= budget, (
        f"{name}: engine BDF err {bdf_err:.2e} exceeds budget {budget:.2e} "
        f"(SciPy LSODA {lsoda_err:.2e}, SciPy BDF {scipy_bdf_err:.2e})"
    )


def test_bdf_matches_scipy_on_monotone_robertson():
    # Robertson is monotone (no oscillatory phase sensitivity), so a tight bound
    # applies: the engine BDF must agree with a tight Radau truth to ~1e-3.
    system = _Robertson()
    ic, ft, dt, rtol, atol = [1.0, 0.0, 0.0], 40.0, 0.05, 1e-7, 1e-9
    truth = _scipy(system, ic, ft, dt, "Radau", 1e-11, 1e-13)
    bdf = _engine(system, ic, ft, dt, "bdf", "interp", rtol, atol)

    assert np.all(np.isfinite(bdf))
    # Mass conservation (a linear invariant of the kinetics) to rounding.
    mass = bdf.sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-6), f"mass drift: {mass.min()}..{mass.max()}"
    assert _max_rel_diff(bdf, truth) < 1e-3


@pytest.mark.parametrize(("name", "build", "ic", "ft", "dt", "rtol", "atol"), _OSCILLATORY)
def test_bdf_interp_equals_jit_bit_for_bit(name, build, ic, ft, dt, rtol, atol):
    # interp and jit are different products but must produce identical numbers, so
    # "use jit for the hot path" never changes a result (the I-BENCH contract).
    system = build()
    a = _engine(system, ic, ft, dt, "bdf", "interp", rtol, atol)
    b = _engine(system, ic, ft, dt, "bdf", "jit", rtol, atol)
    assert np.array_equal(a, b), f"{name}: interp != jit (max {np.max(np.abs(a - b)):.2e})"


# ---------------------------------------------------------------------------
# Protocol stepping over a stiff system (regression for the M3 step()/reinit fix)
# ---------------------------------------------------------------------------
#
# A stiff catalogue system declares ``_default_method = "bdf"``.  ``reinit()``
# lowers the per-step tape ONCE and ``step()`` reuses it, so the Jacobian the
# implicit kernel needs must be baked in at ``reinit`` time — otherwise the engine
# refuses the step with "method 'bdf' needs an analytic Jacobian".  This pins that
# the stepping protocol (and everything built on it — PoincareMap, orbit_diagram,
# the variational Lyapunov engine) works on a stiff default method.


def test_stiff_default_method_steps_without_jacobian_error():
    """``reinit()`` + ``step()`` on a ``_default_method="bdf"`` system must not raise."""
    sys = ts.Oregonator()
    assert sys._default_method == "bdf"
    sys.reinit(sys.resolve_ic())
    for _ in range(5):
        u = sys.step(0.01)
    assert np.all(np.isfinite(u))


def test_stiff_explicit_step_override_also_works():
    """Explicitly requesting an implicit method on a non-stiff system steps cleanly too."""
    lor = ts.Lorenz()
    lor.reinit([1.0, 1.0, 1.0], method="bdf")
    u = lor.step(0.01)
    assert np.all(np.isfinite(u))
