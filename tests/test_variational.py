"""Backend-neutral variational core (stream C-DERIV).

Covers the extended-variational lowering (:mod:`tsdynamics.derived._variational`)
and :class:`~tsdynamics.derived.tangent.TangentSystem` as the one Lyapunov engine
every family delegates to.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.derived._variational import (
    build_variational_tape,
    embed_extended,
    split_extended,
)
from tsdynamics.derived.tangent import TangentSystem
from tsdynamics.families import ContinuousSystem


class LinOsc(ContinuousSystem):
    """Overdamped linear oscillator ``x'=y, y'=-k x - c y``.

    Jacobian ``[[0, 1], [-k, -c]]`` is constant, so the Lyapunov spectrum equals
    the real parts of its eigenvalues ``(-c ± sqrt(c²-4k)) / 2``.  With the
    defaults below the spectrum is exactly ``[-1, -2]`` — a closed-form oracle
    for the variational machinery, with off-diagonal Jacobian coupling.
    """

    params = {"k": 2.0, "c": 3.0}
    dim = 2
    variables = ("x", "y")

    @staticmethod
    def _equations(y, t, k, c):
        return [y(1), -k * y(0) - c * y(1)]


# ---------------------------------------------------------------------------
# Extended-tape lowering and packing
# ---------------------------------------------------------------------------


def test_embed_split_roundtrip() -> None:
    x = np.array([1.0, 2.0, 3.0])
    w = np.arange(6.0).reshape(3, 2)  # (dim=3, k=2)
    z = embed_extended(x, w)
    assert z.shape == (9,)
    x2, w2 = split_extended(z, 3, 2)
    np.testing.assert_array_equal(x2, x)
    np.testing.assert_array_equal(w2, w)


def test_variational_tape_shape_and_rhs() -> None:
    from tsdynamics.engine.compile import eval_tape

    s = LinOsc()
    k = 2
    tape = build_variational_tape(s, k)
    assert tape.dim == s.dim * (k + 1) == 6

    # Extended RHS at x=[1, 0.5], W = I_2: [y1, -k y0 - c y1, J@w0, J@w1].
    x = np.array([1.0, 0.5])
    w = np.eye(2)
    z = embed_extended(x, w)
    p = np.array([s.params["k"], s.params["c"]])  # control-name order
    out = eval_tape(tape, z, p, 0.0)

    kk, cc = 2.0, 3.0
    J = np.array([[0.0, 1.0], [-kk, -cc]])
    base = np.array([x[1], -kk * x[0] - cc * x[1]])
    expected = np.concatenate([base, (J @ w[:, 0]), (J @ w[:, 1])])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_variational_tape_partial_k() -> None:
    s = LinOsc()
    tape = build_variational_tape(s, 1)
    assert tape.dim == s.dim * 2 == 4


# ---------------------------------------------------------------------------
# Backend-neutral ODE Lyapunov via the reference engine (no JiTCODE, no wheel)
# ---------------------------------------------------------------------------


def test_backend_neutral_linear_spectrum_reference() -> None:
    """The reference (pure-Python) variational path reproduces the analytic spectrum."""
    tang = TangentSystem(LinOsc(), k=2, backend="reference")
    spec = tang.lyapunov_spectrum(final_time=40.0, dt=0.25, burn_in=5.0, ic=[1.0, 0.5])
    np.testing.assert_allclose(spec, [-1.0, -2.0], atol=0.02)


def test_backend_neutral_partial_spectrum_reference() -> None:
    """Only the leading exponent, via k=1 deviation vector."""
    tang = TangentSystem(LinOsc(), k=1, backend="reference")
    spec = tang.lyapunov_spectrum(final_time=40.0, dt=0.25, burn_in=5.0, ic=[1.0, 0.5])
    assert spec.shape == (1,)
    np.testing.assert_allclose(spec, [-1.0], atol=0.02)


def test_backend_neutral_deviations_orthonormal() -> None:
    tang = TangentSystem(LinOsc(), k=2, backend="reference")
    tang.reinit([1.0, 0.5])
    for _ in range(10):
        tang.step(0.25)
    q = tang.deviations()
    assert q.shape == (2, 2)
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-10)


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ODE tangent backend"):
        TangentSystem(ts.Lorenz(), backend="bogus")


def test_engine_deviations_for_jitcode_backend_raise() -> None:
    tang = TangentSystem(ts.Lorenz(), k=3, backend="jitcode")
    with pytest.raises(RuntimeError, match="jitcode"):
        tang.deviations()


# ---------------------------------------------------------------------------
# TangentSystem is the one engine: families delegate to it (identical results)
# ---------------------------------------------------------------------------


def test_map_family_delegates_to_tangent() -> None:
    """Family map ``lyapunov_spectrum`` is exactly ``TangentSystem.lyapunov_spectrum``."""
    via_family = ts.Henon().lyapunov_spectrum(steps=4000, ic=[0.1, 0.1])
    via_tangent = TangentSystem(ts.Henon(), k=2).lyapunov_spectrum(steps=4000, ic=[0.1, 0.1])
    np.testing.assert_array_equal(via_family, via_tangent)


def test_map_partial_spectrum_via_n_exp() -> None:
    spec = ts.Henon().lyapunov_spectrum(steps=4000, ic=[0.1, 0.1], n_exp=1)
    assert spec.shape == (1,)
    assert 0.3 < spec[0] < 0.5  # leading Hénon exponent ≈ 0.42


def test_tangent_lyapunov_records_meta() -> None:
    m = ts.Henon()
    tang = TangentSystem(m, k=2)
    tang.lyapunov_spectrum(steps=2000, ic=[0.1, 0.1])
    # meta is the inner system's MetaStore.
    assert "lyapunov_spectrum" in m.meta
    rec = m.meta.history("lyapunov_spectrum")[-1]
    assert rec["context"]["steps"] == 2000
    assert rec["context"]["reortho_interval"] == 1


@pytest.mark.slow
def test_ode_family_delegates_to_tangent_jitcode() -> None:
    """Family ODE ``lyapunov_spectrum`` is the jitcode TangentSystem path.

    The family method *is* ``TangentSystem(self, backend="jitcode").lyapunov_spectrum``,
    so the two agree up to ``jitcode_lyap``'s random initial tangent directions
    (which make finite-time estimates non-bit-identical across fresh runs).
    """
    via_family = ts.Lorenz(ic=[1.0, 1.0, 1.0]).lyapunov_spectrum(
        final_time=80.0, dt=0.1, burn_in=20.0
    )
    via_tangent = TangentSystem(
        ts.Lorenz(ic=[1.0, 1.0, 1.0]), k=3, backend="jitcode"
    ).lyapunov_spectrum(final_time=80.0, dt=0.1, burn_in=20.0)
    np.testing.assert_allclose(via_family, via_tangent, atol=0.05)


@pytest.mark.slow
def test_backend_neutral_matches_jitcode_on_lorenz() -> None:
    """The engine variational path agrees with jitcode on the Lorenz spectrum."""
    ic = [1.0, 1.0, 1.0]
    jit = TangentSystem(ts.Lorenz(ic=ic), k=3, backend="jitcode").lyapunov_spectrum(
        final_time=120.0, dt=0.1, burn_in=40.0
    )
    ref = TangentSystem(ts.Lorenz(ic=ic), k=3, backend="reference").lyapunov_spectrum(
        final_time=120.0, dt=0.1, burn_in=40.0
    )
    # Finite-time Lyapunov estimates from two integrators converge but are not
    # bit-identical; the leading exponent and the (large negative) third agree well.
    assert abs(jit[0] - ref[0]) < 0.15
    assert abs(jit[1] - ref[1]) < 0.15
    assert abs(jit[2] - ref[2]) < 1.0
