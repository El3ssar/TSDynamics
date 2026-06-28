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


def test_jitcode_backend_is_rejected() -> None:
    """The retired ``jitcode`` variational backend is no longer a valid choice."""
    with pytest.raises(ValueError, match="unknown ODE tangent backend"):
        TangentSystem(ts.Lorenz(), k=3, backend="jitcode")


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


class _StiffLinOsc(ContinuousSystem):
    """Linear oscillator that *defaults to the stiff BDF kernel*.

    Same constant-Jacobian closed-form oracle as :class:`LinOsc` (spectrum
    ``[-1, -2]``), but with ``_default_method = "bdf"`` so the tangent flow is
    driven onto an *implicit* engine kernel.  Before the fix the extended
    variational tape was lowered with ``jacobian=False``, so the implicit kernel
    found no Jacobian tape and ``lyapunov_spectrum`` *raised*; this gives a fast,
    closed-form regression for that path without the cost of a genuinely stiff
    catalogue system.
    """

    params = {"k": 2.0, "c": 3.0}
    dim = 2
    variables = ("x", "y")
    _default_method = "bdf"

    @staticmethod
    def _equations(y, t, k, c):
        return [y(1), -k * y(0) - c * y(1)]


def test_variational_tape_emits_jacobian() -> None:
    """The extended variational tape carries its own Jacobian block.

    Regression guard for the P0 bug: a base flow whose ``_default_method`` is an
    implicit kernel (``"bdf"``) integrates the *pre-built* extended ODEProblem,
    and ``engine.run.integrate`` does not rebuild a pre-built problem
    ``with_jacobian=True`` — so the Jacobian must be present on the tape itself.
    """
    tape = build_variational_tape(LinOsc(), 2)
    assert tape.has_jacobian


def test_stiff_default_ode_lyapunov_does_not_raise() -> None:
    """A stiff-defaulted (``bdf``) flow's Lyapunov spectrum integrates cleanly.

    Fast closed-form proxy for the catalogue stiff systems (Oregonator,
    KuramotoSivashinsky, Duffing, the stiff Sprott jerks): the implicit kernel
    now has the extended Jacobian it needs, so the spectrum is finite, descending
    and matches the analytic ``[-1, -2]`` instead of raising.  Exercises the
    *engine* implicit path (the one the bug broke), so it skips without the
    compiled extension.
    """
    pytest.importorskip("tsdynamics._rust")
    tang = TangentSystem(_StiffLinOsc(), k=2, backend="interp")
    spec = tang.lyapunov_spectrum(final_time=40.0, dt=0.25, burn_in=5.0, ic=[1.0, 0.5])
    assert np.all(np.isfinite(spec))
    assert spec[0] >= spec[1]  # descending (QR order)
    np.testing.assert_allclose(spec, [-1.0, -2.0], atol=0.05)


def test_structural_change_rebuilds_extended_tape() -> None:
    """A live structural-parameter change re-lowers the cached extended tape.

    The extended tape bakes in structural parameters; a stale cache would carry
    the wrong dimension/structure.  Reinitialising after a structural change must
    rebuild it (the cache is keyed on the structural values).
    """
    tang = TangentSystem(LinOsc(), k=2, backend="reference")
    tang.reinit([1.0, 0.5])
    first_tape = tang._ext_tape
    first_key = tang._ext_tape_key
    # No structural params on LinOsc → key is the empty tuple and the tape is
    # reused across reinit (a control/IC change is not a structural change).
    tang.reinit([0.2, 0.3])
    assert tang._ext_tape is first_tape
    assert tang._ext_tape_key == first_key

    # Simulate a structural change by poking the cached key stale; the next
    # reinit must rebuild (key mismatch path).
    tang._ext_tape_key = (("N", 99),)
    tang.reinit([0.2, 0.3])
    assert tang._ext_tape is not first_tape
    assert tang._ext_tape_key == ()


@pytest.mark.slow
def test_oregonator_stiff_lyapunov_finite_descending() -> None:
    """The genuinely-stiff Oregonator Lyapunov spectrum is finite and descending.

    End-to-end guard for the named P0 system: ``ts.Oregonator()`` defaults to the
    implicit ``bdf`` kernel, so ``lyapunov_spectrum`` drives the extended
    variational ODE onto that kernel.  Before the fix this *raised* (no Jacobian
    tape); it must now return a finite, descending spectrum.  ``final_time`` is
    kept modest for speed — correctness of the *values* is covered by the
    closed-form oscillator oracles above; here we only assert it does not raise
    and is well-formed.
    """
    pytest.importorskip("tsdynamics._rust")
    spec = ts.Oregonator().lyapunov_spectrum(
        final_time=6.0, dt=0.01, burn_in=2.0, ic=[1.0, 1.0, 1.0]
    )
    assert spec.shape == (3,)
    assert np.all(np.isfinite(spec))
    assert spec[0] >= spec[1] >= spec[2]  # descending (QR order)


@pytest.mark.slow
def test_ode_family_delegates_to_tangent_engine() -> None:
    """Family ODE ``lyapunov_spectrum`` reproduces the literature Lorenz spectrum.

    The family method *is* ``TangentSystem(self).lyapunov_spectrum`` on the engine
    variational path; ``final_time`` is long enough that the finite-time estimate
    has converged to the canonical Lorenz spectrum ``[0.906, 0, -14.57]``.
    """
    spec = ts.Lorenz(ic=[1.0, 1.0, 1.0]).lyapunov_spectrum(final_time=240.0, dt=0.1, burn_in=40.0)
    # Leading exponent ≈ 0.906, middle ≈ 0, third ≈ -14.57 (Lorenz 1963 / Sprott).
    assert abs(spec[0] - 0.906) < 0.06
    assert abs(spec[1]) < 0.06
    assert abs(spec[2] + 14.57) < 0.6


@pytest.mark.slow
def test_backend_neutral_lorenz_spectrum_reference() -> None:
    """The engine variational path reproduces the Lorenz spectrum on the reference backend."""
    ref = TangentSystem(ts.Lorenz(ic=[1.0, 1.0, 1.0]), k=3, backend="reference").lyapunov_spectrum(
        final_time=120.0, dt=0.1, burn_in=40.0
    )
    assert abs(ref[0] - 0.906) < 0.1
    assert abs(ref[1]) < 0.1
    assert abs(ref[2] + 14.57) < 1.0
