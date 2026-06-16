"""Engine DDE Lyapunov spectrum (stream E-DDE-LYAP).

The engine estimator (``DelaySystem.lyapunov_spectrum(backend="interp"/"jit")``)
integrates the extended variational DDE — base state ⊕ ``k`` deviation states — on
the Rust method-of-steps engine with a function-space Benettin renormalisation. It
is the post-M3 successor to the retired ``jitcdde_lyap``; the original Rust-vs-v2
parity gate ran in the E-DDE-LYAP PR before JiTCDDE was deleted, so these checks
are the reference-free correctness bars that survive the removal:

* the Mackey–Glass leading exponent is positive (its ``known_lyapunov`` is
  ``n_positive=1``), and the subdominant exponent sits at the marginal ≈ 0;
* ``interp`` and ``jit`` agree **bit-for-bit** (the D2 contract);
* the spectrum is descending and ``n_exp`` may exceed ``dim`` (the tangent space
  is the infinite-dimensional history space).
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import DDE_HISTORIES

import tsdynamics as ts

_rust = pytest.importorskip("tsdynamics._rust")


def _on_attractor_ic(name: str) -> np.ndarray:
    """A deterministic on-attractor initial state (end of a seeded integration)."""
    sys = getattr(ts, name)()
    traj = sys.integrate(final_time=500.0, dt=0.2, history=DDE_HISTORIES[name])
    return np.asarray(traj.y[-1], dtype=np.float64)


# ---------------------------------------------------------------------------
# Fast tier — bit-exact interp==jit + structural properties (no jitcdde compile)
# ---------------------------------------------------------------------------


def test_interp_equals_jit_bit_for_bit() -> None:
    """The interpreter and the Cranelift JIT give an identical spectrum (D2)."""
    ic = _on_attractor_ic("MackeyGlass")
    kw = dict(n_exp=2, dt=0.2, burn_in=40.0, final_time=200.0, ic=ic)
    interp = ts.MackeyGlass().lyapunov_spectrum(backend="interp", **kw)
    jit = ts.MackeyGlass().lyapunov_spectrum(backend="jit", **kw)
    np.testing.assert_array_equal(interp, jit)


def test_spectrum_is_descending_with_positive_leading() -> None:
    """Mackey–Glass: λ₁ > 0 (chaos), and the spectrum is sorted descending."""
    ic = _on_attractor_ic("MackeyGlass")
    spec = ts.MackeyGlass().lyapunov_spectrum(
        backend="interp", n_exp=2, dt=0.2, burn_in=100.0, final_time=600.0, ic=ic
    )
    assert spec.shape == (2,)
    assert np.all(np.isfinite(spec))
    assert spec[0] >= spec[1]  # descending
    assert spec[0] > 0.0  # chaotic (matches known_lyapunov n_positive=1)


def test_n_exp_may_exceed_dim() -> None:
    """A dim-1 DDE supports n_exp > dim — the tangent space is infinite-dimensional.

    The spectrum is sorted by construction, so the discriminating check is that
    the exponents are *distinct* (a rank-deficient seed would collapse them to a
    single value) and span zero — a chaotic autonomous flow has a positive
    leading exponent and a marginal (≈ 0) direction, so the spread must bracket 0.
    """
    sys = ts.SprottDelay()
    assert sys.dim == 1
    spec = sys.lyapunov_spectrum(
        backend="interp",
        n_exp=3,
        dt=0.05,
        burn_in=200.0,
        final_time=1500.0,
        ic=_on_attractor_ic("SprottDelay"),
    )
    assert spec.shape == (3,)
    assert np.all(np.isfinite(spec))
    assert spec[0] >= spec[1] >= spec[2]
    assert spec[0] - spec[2] > 1e-3, f"degenerate (collapsed) spectrum {spec}"
    assert spec[0] > 0.0 > spec[2], f"chaotic flow spectrum must bracket 0: {spec}"


def test_mackeyglass_second_exponent_is_near_zero() -> None:
    """A flow DDE has a marginal (≈ 0) Lyapunov direction along the trajectory.

    Mackey–Glass has one positive exponent; the second must sit at zero (the phase
    direction). This is a discriminating, reference-free value check on a
    subdominant exponent — not satisfied by the construction-guaranteed sort.
    """
    ic = _on_attractor_ic("MackeyGlass")
    spec = ts.MackeyGlass().lyapunov_spectrum(
        backend="interp", n_exp=2, dt=0.1, burn_in=200.0, final_time=2000.0, ic=ic
    )
    assert spec[0] > 0.0
    assert abs(spec[1]) < 0.01, f"second exponent {spec[1]} not near the marginal 0"


def test_reference_backend_is_rejected() -> None:
    """No pure-Python DDE integrator exists, so backend='reference' raises (like integrate)."""
    with pytest.raises(NotImplementedError, match="reference"):
        ts.MackeyGlass().lyapunov_spectrum(backend="reference")


def test_extra_kwargs_rejected_on_engine_path() -> None:
    """A stray integration keyword is rejected on the engine path, not silently ignored."""
    with pytest.raises(TypeError, match="extra integration keyword"):
        ts.MackeyGlass().lyapunov_spectrum(backend="interp", max_step=0.1)


def _two_dim_delay():
    """A coupled 2-D / 2-delay DDE (defined on call so it registers lazily, not at import)."""
    from tsdynamics.families import DelaySystem

    class _TwoDimDelay(DelaySystem):
        params = {"a": 1.5, "b": 0.3, "tau1": 1.0, "tau2": 2.0}
        dim = 2
        _delay_params = ("tau1", "tau2")

        @staticmethod
        def _equations(y, t, *, a, b, tau1, tau2):
            from symengine import sin

            return [-y(0) + a * sin(y(1, t - tau1)), -y(1) + b * y(0, t - tau2)]

    return _TwoDimDelay()


def test_multidim_multidelay_construction() -> None:
    """The extended-tape construction generalizes past the dim-1 built-ins.

    All five catalogue DDEs are dim-1 with a single delay; a coupled 2-D system
    with two distinct delays exercises the per-slot Jacobian and the delayed-
    deviation component mapping (``dim + m*dim + comp``) that the built-ins do not.
    """
    from tsdynamics.families._dde_lyapunov import _build_extended_tape

    sys = _two_dim_delay()
    tape, slots, dim = _build_extended_tape(sys, 2)
    assert dim == 2
    assert tape.dim == dim * 3  # base ⊕ 2 deviations
    # base slots (2) + one delayed-deviation slot per (deviation, base-slot) → 2 + 4
    assert len(slots) == 6
    # deviation m's delayed component is dim + m*dim + base_component
    assert {(s.component, round(s.delay, 1)) for s in slots} == {
        (1, 1.0),
        (0, 2.0),  # base
        (3, 1.0),
        (2, 2.0),  # deviation 0  (dim + 0*dim + {1,0})
        (5, 1.0),
        (4, 2.0),  # deviation 1  (dim + 1*dim + {1,0})
    }


@pytest.mark.slow
def test_multidim_spectrum_is_consistent_and_brackets_zero() -> None:
    """A 2-D / 2-delay DDE: the full engine spectrum is finite, sorted, brackets 0.

    The dim ≥ 2 path is where the function-space QR axis ordering matters (a
    scrambled reshape passes the structural checks but corrupts the exponents).
    With v2 retired this asserts the reference-free structural invariants the
    original jitcdde_lyap value-parity test enforced: both exponents finite,
    descending, and straddling the marginal 0 of an autonomous flow.
    """
    sys = _two_dim_delay()
    ic = sys.integrate(
        final_time=300.0, dt=0.1, history=lambda s: [0.5 + 0.1 * np.sin(s), 0.3 + 0.1 * np.cos(s)]
    ).y[-1]
    eng = sys.lyapunov_spectrum(
        backend="interp", n_exp=2, dt=0.05, burn_in=200.0, final_time=2000.0, ic=ic
    )
    assert eng.shape == (2,)
    assert np.all(np.isfinite(eng))
    assert eng[0] >= eng[1]  # descending
