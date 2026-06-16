"""Engine DDE Lyapunov spectrum vs JiTCDDE (stream E-DDE-LYAP).

The engine estimator (``DelaySystem.lyapunov_spectrum(backend="interp"/"jit")``,
the extended variational DDE integrated on the Rust engine with function-space
Benettin renormalisation) must reproduce the v2 ``jitcdde_lyap`` path before the
M3 migration can delete JiTCDDE without regressing the DDE-Lyapunov differentiator
(ROADMAP §12.1).  This module is that gate:

* the Mackey–Glass leading exponent is positive (its ``known_lyapunov`` is
  ``n_positive=1``) and close to ``jitcdde``;
* every built-in DDE's leading exponent agrees with ``jitcdde_lyap`` within a
  tolerance appropriate to DDE Lyapunov (which is genuinely method-sensitive);
* ``interp`` and ``jit`` agree **bit-for-bit** (the D2 contract);
* the spectrum is descending and ``n_exp`` may exceed ``dim`` (the tangent space
  is the infinite-dimensional history space).

The slow tier compiles ``jitcdde`` for the parity comparison; the bit-exact and
structural checks stay in the fast tier.
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import DDE_HISTORIES

import tsdynamics as ts

_rust = pytest.importorskip("tsdynamics._rust")

_DDES = ["MackeyGlass", "IkedaDelay", "SprottDelay", "ScrollDelay", "PiecewiseCircuit"]


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


def test_jitcdde_only_kwargs_rejected_on_engine_path() -> None:
    """A jitcdde-only integration keyword is rejected on the engine path, not silently ignored."""
    with pytest.raises(TypeError, match="jitcdde-only"):
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


# ---------------------------------------------------------------------------
# Slow tier — parity with the v2 JiTCDDE backend (compiles C)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mackey_glass_matches_jitcdde() -> None:
    """The canonical DDE: the engine λ₁ tracks ``jitcdde_lyap`` closely."""
    mg = ts.MackeyGlass()
    ic = _on_attractor_ic("MackeyGlass")
    ref = mg.lyapunov_spectrum(
        backend="jitcdde",
        n_exp=1,
        dt=0.1,
        burn_in=200.0,
        final_time=2000.0,
        ic=ic,
        rtol=1e-5,
        atol=1e-5,
    )
    eng = mg.lyapunov_spectrum(
        backend="interp", n_exp=1, dt=0.05, burn_in=200.0, final_time=2000.0, ic=ic
    )
    assert eng[0] > 0.0
    assert abs(eng[0] - ref[0]) <= 0.25 * abs(ref[0]) + 5e-4, f"engine {eng[0]} vs jitcdde {ref[0]}"


@pytest.mark.slow
@pytest.mark.parametrize("name", _DDES)
def test_engine_matches_jitcdde_leading_exponent(name) -> None:
    """Every built-in DDE: the engine λ₁ agrees with ``jitcdde_lyap`` within tolerance.

    DDE Lyapunov is genuinely method-sensitive (jitcdde itself drifts with its
    own tolerances), so the bar is the right sign plus agreement to ~40 % at
    matched, well-averaged parameters — tight enough to catch a wrong estimator,
    loose enough not to flag the inherent estimator spread.
    """
    sys = getattr(ts, name)()
    ic = _on_attractor_ic(name)
    common = dict(n_exp=1, burn_in=300.0, final_time=3000.0, ic=ic)
    ref = sys.lyapunov_spectrum(backend="jitcdde", dt=0.05, rtol=1e-5, atol=1e-5, **common)
    eng = sys.lyapunov_spectrum(backend="interp", dt=0.025, **common)
    assert eng[0] > 0.0, f"{name}: engine λ₁ = {eng[0]} not positive"
    assert np.sign(eng[0]) == np.sign(ref[0])
    rel = abs(eng[0] - ref[0]) / (abs(ref[0]) + 1e-12)
    assert rel <= 0.4, f"{name}: engine λ₁ {eng[0]:.4f} vs jitcdde {ref[0]:.4f} (rel {rel:.2f})"


@pytest.mark.slow
def test_multidim_spectrum_matches_jitcdde() -> None:
    """A 2-D / 2-delay DDE: the full engine spectrum tracks jitcdde_lyap.

    The dim ≥ 2 path is where the function-space QR axis ordering matters (a
    scrambled reshape passes the structural checks but corrupts the exponents),
    so this asserts the *values* of both engine exponents against an independent
    jitcdde_lyap run — the discriminating check the construction tests cannot make.
    """
    sys = _two_dim_delay()
    ic = sys.integrate(
        final_time=300.0, dt=0.1, history=lambda s: [0.5 + 0.1 * np.sin(s), 0.3 + 0.1 * np.cos(s)]
    ).y[-1]
    common = dict(n_exp=2, dt=0.05, burn_in=200.0, final_time=2000.0, ic=ic)
    ref = sys.lyapunov_spectrum(backend="jitcdde", rtol=1e-5, atol=1e-5, **common)
    eng = sys.lyapunov_spectrum(backend="interp", **common)
    np.testing.assert_allclose(eng, ref, atol=0.02, err_msg=f"engine {eng} vs jitcdde {ref}")
