"""Analytic map derivatives separate real collapse from norm/propagation loss."""

from fractions import Fraction

import numpy as np
import pytest

import tsdynamics as ts

pytest.importorskip("tsdynamics._rust")


class ScalarMultiplier(ts.DiscreteMap):
    variables = ("x",)
    params = {"factor": 0.0}
    _default_ic = (0.0,)

    @staticmethod
    def _step(x, factor):
        return np.array([factor * x[0]])

    @staticmethod
    def _jacobian(x, factor):
        return np.array([[factor]])


class LinearPlane(ts.DiscreteMap):
    variables = ("x", "y")
    params = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0}
    _default_ic = (0.0, 0.0)

    @staticmethod
    def _step(x, a, b, c, d):
        return np.array([a * x[0] + b * x[1], c * x[0] + d * x[1]])

    @staticmethod
    def _jacobian(x, a, b, c, d):
        return np.array([[a, b], [c, d]])


@pytest.mark.parametrize("backend", ["jit", "interp"])
@pytest.mark.parametrize("factor", [0.0, 1e-320, 1e-300, 1e-200, 1e200, -1e200])
def test_scalar_map_preserves_exact_logarithmic_rate(backend, factor):
    result = ts.analysis.lyapunov_spectrum(
        ScalarMultiplier(params={"factor": factor}),
        n=6,
        transient=0,
        reortho_interval=1,
        backend=backend,
    )
    expected = np.log(abs(factor)) if factor else -np.inf
    np.testing.assert_allclose(result.exponents, [expected], rtol=1e-14)


@pytest.mark.parametrize("backend", ["jit", "interp"])
@pytest.mark.parametrize("diagonal", [(0.0, 2.0), (2.0, 0.0), (0.0, 0.0)])
def test_zero_before_or_after_surviving_diagonal_direction(backend, diagonal):
    result = ts.analysis.lyapunov_spectrum(
        LinearPlane(params={"a": diagonal[0], "d": diagonal[1]}),
        n=6,
        transient=0,
        backend=backend,
    )
    expected = sorted((np.log(value) if value else -np.inf for value in diagonal), reverse=True)
    np.testing.assert_allclose(result.exponents, expected, rtol=1e-14)


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_nilpotent_image_survives_one_step_then_collapses(backend):
    model = LinearPlane(params={"a": 0.0, "b": 2.0, "d": 0.0})
    one = ts.analysis.lyapunov_spectrum(model, n=1, transient=0, backend=backend)
    np.testing.assert_allclose(one.exponents, [np.log(2.0), -np.inf])
    two = ts.analysis.lyapunov_spectrum(model, n=2, transient=0, backend=backend)
    assert np.all(np.isneginf(two.exponents))


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_dependent_nonzero_columns_are_refused_without_a_fabricated_complement(backend):
    model = LinearPlane(params={"a": 1.0, "b": 1.0, "d": 0.0})
    with pytest.raises(ValueError, match="dependent or numerically unresolved"):
        ts.analysis.lyapunov_spectrum(model, n=2, transient=0, backend=backend)


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_underflow_between_renormalisations_is_not_reported_as_superstability(backend):
    with pytest.raises(ValueError, match="underflowed to zero.*reortho_interval"):
        ts.analysis.lyapunov_spectrum(
            ScalarMultiplier(params={"factor": 1e-300}),
            n=2,
            transient=0,
            reortho_interval=2,
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_tangent_overflow_is_not_called_orbit_divergence(backend):
    # The base state remains exactly zero. Only an overly long tangent interval
    # overflows; retrying random initial states would misdiagnose this failure.
    with pytest.raises(ValueError, match="non-finite tangent propagation.*reortho_interval"):
        ts.analysis.lyapunov_spectrum(
            ScalarMultiplier(params={"factor": 1e200}),
            n=2,
            transient=0,
            reortho_interval=2,
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_cancellation_to_zero_is_not_mistaken_for_exact_rank_loss(backend):
    a = np.nextafter(1.0, 2.0)
    c = -(a * a)
    # These exact floating coefficients define A²=2^-104 I, not a nilpotent map.
    square = Fraction(float(a)) ** 2 + Fraction(float(c))
    assert square == Fraction(1, 2**104)
    model = LinearPlane(params={"a": a, "b": 1.0, "c": c, "d": -a})
    try:
        result = ts.analysis.lyapunov_spectrum(
            model, n=2, transient=0, reortho_interval=2, backend=backend
        )
    except ValueError as exc:
        assert "cancellation" in str(exc) or "numerically unresolved" in str(exc)
    else:
        # A kernel that resolves the cancellation must retain the finite rates.
        np.testing.assert_allclose(result.exponents, [0.5 * np.log(float(square))] * 2)
