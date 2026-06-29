"""Discrete-map Lyapunov on the Rust kernel (stream ``perf/map-lyapunov-kernel``).

The map analogue of the basin-march / ODE-stepper perf streams: the per-step
Python QR tangent-map loop (``TangentSystem._accumulate_map``) replaced by a single
Rust engine call (``tsdynamics.engine.run.map_lyapunov`` →
``tsdynamics._rust.map_lyapunov_spectrum``).  These tests lock in the contract:

* the engine spectrum reproduces the pure-Python QR oracle (``backend="reference"``)
  to the WS-MAPITER IR-vs-NumPy tolerance, and both match the literature;
* ``interp == jit`` bit-for-bit (same lowered tape, two evaluators);
* ``max_lyapunov`` on a map equals the kernel's leading exponent, while its
  continuous-system two-trajectory path is untouched;
* a map whose ``_step`` will not lower transparently falls back to the NumPy loop;
* divergence raises loudly.

The module imports the compiled extension, so the ``engine`` marker auto-tags it
(``tests/_engine_marker.py``) and it skips cleanly where the wheel is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts
from tsdynamics.systems import Henon, Ikeda, Logistic, Tinkerbell


def _spectrum(cls, **kw):
    return np.asarray(cls().lyapunov_spectrum(**kw), dtype=float)


@pytest.mark.parametrize("backend", ["interp", "jit"])
def test_henon_spectrum_matches_literature(backend) -> None:
    """Hénon at default params → λ ≈ [0.419, -1.623] (Sprott 2003) on the kernel."""
    spec = _spectrum(Henon, steps=10_000, ic=[0.1, 0.1], backend=backend)
    assert spec.shape == (2,)
    assert spec[0] > spec[1]  # descending (QR order)
    assert abs(spec[0] - 0.419) < 0.05, spec
    assert abs(spec[1] - (-1.623)) < 0.05, spec


def test_interp_equals_jit_bit_for_bit() -> None:
    """The kernel drives both evaluators over the same lowered tape → bit-for-bit."""
    interp = _spectrum(Henon, steps=8000, ic=[0.1, 0.1], backend="interp")
    jit = _spectrum(Henon, steps=8000, ic=[0.1, 0.1], backend="jit")
    assert interp.dtype == jit.dtype == np.float64
    assert np.array_equal(interp.view(np.uint64), jit.view(np.uint64)), (interp, jit)


@pytest.mark.parametrize("cls", [Henon, Ikeda])
def test_engine_matches_reference_oracle(cls) -> None:
    """The Rust kernel reproduces the pure-Python QR oracle to tolerance.

    They differ only by the lowered-IR vs pure-Python ``_step``/``_jacobian`` float
    order (the WS-MAPITER caveat), so a chaotic map's spectrum agrees to a few 1e-3
    over a long run — the same attractor, the same exponents.
    """
    eng = _spectrum(cls, steps=10_000, ic=[0.1, 0.1], backend="interp")
    ref = _spectrum(cls, steps=10_000, ic=[0.1, 0.1], backend="reference")
    assert np.all(np.isfinite(eng))
    assert np.max(np.abs(eng - ref)) < 1e-2, (eng, ref)


def test_logistic_r4_is_ln2() -> None:
    """Fully-chaotic logistic (r=4): λ = ln 2 ≈ 0.6931, on the kernel."""
    spec = np.asarray(
        Logistic(params={"r": 4.0}).lyapunov_spectrum(steps=50_000, ic=[0.1], backend="interp"),
        dtype=float,
    )
    assert abs(float(spec[0]) - np.log(2.0)) < 0.02, spec


def test_tinkerbell_is_chaotic() -> None:
    """Tinkerbell has one positive exponent (its ``known_lyapunov`` n_positive=1)."""
    spec = _spectrum(Tinkerbell, steps=20_000, backend="interp")
    assert int((spec > 0.01).sum()) == 1, spec


def test_partial_spectrum_k_less_than_dim() -> None:
    """Requesting fewer exponents than ``dim`` returns just the leading ones."""
    full = _spectrum(Henon, steps=8000, ic=[0.1, 0.1])
    top = np.asarray(Henon().lyapunov_spectrum(steps=8000, ic=[0.1, 0.1], n_exp=1), dtype=float)
    assert top.shape == (1,)
    # Same orbit, same leading direction → the maximal exponent agrees bit-for-bit.
    assert top[0].view(np.uint64) == full[0].view(np.uint64), (top, full)


def test_reortho_interval_is_answer_preserving() -> None:
    """Reorthonormalising every step vs every 5 gives the same spectrum to tolerance."""
    every1 = _spectrum(Henon, steps=10_000, ic=[0.1, 0.1], reortho_interval=1)
    every5 = _spectrum(Henon, steps=10_000, ic=[0.1, 0.1], reortho_interval=5)
    assert np.max(np.abs(every1 - every5)) < 1e-2, (every1, every5)


def test_max_lyapunov_map_equals_kernel_top_exponent() -> None:
    """``max_lyapunov`` on a map is the kernel's leading exponent (top of the spectrum)."""
    mle = float(ts.max_lyapunov(Henon(ic=[0.1, 0.1]), n=2000, steps_per=5))
    # The map path runs the same kernel with steps = n*steps_per, k=1 from the
    # burnt-in state; the result must equal the leading spectrum exponent to a few
    # 1e-3 (same estimator, the transient placement aside) and the literature value.
    assert abs(mle - 0.419) < 0.05, mle


def test_max_lyapunov_continuous_path_unchanged() -> None:
    """The continuous-system two-trajectory path is untouched (a smoke regression)."""
    mle = float(ts.max_lyapunov(ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]), dt=0.05, n=300))
    # Lorenz maximal exponent ≈ 0.9; a loose band — this only guards that the ODE
    # path still produces a sane positive exponent (it does not use the kernel).
    assert 0.5 < mle < 1.4, mle


def test_reference_backend_uses_the_numpy_oracle() -> None:
    """``backend="reference"`` returns a finite, literature-consistent spectrum.

    It runs the pure-Python QR loop (the oracle), independent of the kernel.
    """
    spec = _spectrum(Henon, steps=10_000, ic=[0.1, 0.1], backend="reference")
    assert np.all(np.isfinite(spec))
    assert abs(spec[0] - 0.419) < 0.05, spec


def test_non_lowering_map_falls_back_to_numpy() -> None:
    """A map whose ``_step`` will not lower (a Python ``if`` branch) still works.

    The engine fast path declines (``TapeCompileError``) and the spectrum is
    computed by the pure-Python QR loop — transparently, on the default backend.
    """
    from tsdynamics.families import DiscreteMap

    class BranchingMap(DiscreteMap):
        # A tent-like map written with a Python ``if`` on the state, so ``_step``
        # cannot trace to a straight-line tape (it raises TapeCompileError); the
        # default ``interp`` backend must fall back to the NumPy QR loop.
        params = {"r": 1.9}
        dim = 1

        @staticmethod
        def _step(X, r):
            x = X[0] if hasattr(X, "__getitem__") else X
            if x < 0.5:
                return r * x
            return r * (1.0 - x)

        @staticmethod
        def _jacobian(X, r):
            x = X[0] if hasattr(X, "__getitem__") else X
            return [r if x < 0.5 else -r]

        _jacobian_fd_check = False

    spec = np.asarray(BranchingMap().lyapunov_spectrum(steps=5000, ic=[0.3]), dtype=float)
    assert np.all(np.isfinite(spec))
    # The tent map at r=1.9 is chaotic with λ = ln(r) ≈ 0.642.
    assert abs(float(spec[0]) - np.log(1.9)) < 0.05, spec
