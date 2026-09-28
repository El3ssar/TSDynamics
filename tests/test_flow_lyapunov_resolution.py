"""Exact flow oracles distinguish contraction from the integration error floor."""

import numpy as np
import pytest

import tsdynamics as ts

pytest.importorskip("tsdynamics._rust")


class FastDecay(ts.ContinuousSystem):
    variables = ("x",)
    params = {"rate": -1000.0}
    _default_ic = (0.0,)

    @staticmethod
    def _equations(y, t, rate):
        return [rate * y(0)]


class CoupledDecay(ts.ContinuousSystem):
    variables = ("x", "y")
    params = {}
    _default_ic = (0.0, 0.0)

    @staticmethod
    def _equations(y, t):
        # Orthogonal rotation of diag(-1000,-1). Both raw tangent columns
        # remain large; only their QR residual exposes the fast contraction.
        return [-500.5 * y(0) - 499.5 * y(1), -499.5 * y(0) - 500.5 * y(1)]


class ShearDecay(ts.ContinuousSystem):
    variables = ("x", "y")
    params = {}
    _default_ic = (0.0, 0.0)

    @staticmethod
    def _equations(y, t):
        return [-y(0) + 100 * y(1), -2 * y(1)]


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_contraction_at_absolute_error_scale_is_refused(backend):
    with pytest.raises(ValueError, match="integration error scale.*reduce dt"):
        ts.analysis.lyapunov_spectrum(
            FastDecay(), final_time=0.2, transient=0, dt=0.1, backend=backend
        )


@pytest.mark.parametrize("backend", ["jit", "interp"])
@pytest.mark.parametrize("dt", [0.01, 0.001])
def test_shorter_renormalisation_intervals_resolve_exact_rate(backend, dt):
    result = ts.analysis.lyapunov_spectrum(
        FastDecay(), final_time=0.2, transient=0, dt=dt, backend=backend
    )
    np.testing.assert_allclose(result.exponents, [-1000.0], rtol=1e-6)


@pytest.mark.parametrize("backend", ["jit", "interp"])
@pytest.mark.parametrize("atol", [1e-60, 0.0])
def test_explicit_tolerance_can_resolve_the_same_long_chunk(backend, atol):
    # A nonzero base state keeps pure-relative control defined on all components.
    result = ts.analysis.lyapunov_spectrum(
        FastDecay(),
        ic=[1.0],
        final_time=0.2,
        transient=0,
        dt=0.1,
        rtol=1e-9,
        atol=atol,
        backend=backend,
    )
    np.testing.assert_allclose(result.exponents, [-1000.0], rtol=1e-7)


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_coupled_contraction_is_screened_after_projection(backend):
    with pytest.raises(ValueError, match="integration error scale|numerically unresolved"):
        ts.analysis.lyapunov_spectrum(
            CoupledDecay(), final_time=0.2, transient=0, dt=0.1, backend=backend
        )
    result = ts.analysis.lyapunov_spectrum(
        CoupledDecay(), final_time=0.2, transient=0, dt=0.001, backend=backend
    )
    # Exact first-column length of exp(A*T); the second QR stretch follows
    # from det(exp(A*T))=exp(trace(A)*T), without subtracting tiny exponentials.
    first = 0.5 * np.logaddexp(-2 * 0.2, -2000 * 0.2) / 0.2 - 0.5 * np.log(2) / 0.2
    np.testing.assert_allclose(result.exponents, [first, -1001 - first], rtol=1e-6)


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_resolved_nonnormal_flow_and_default_lorenz_remain_usable(backend):
    shear = ts.analysis.lyapunov_spectrum(
        ShearDecay(), final_time=1.0, transient=0, dt=0.1, backend=backend
    )
    np.testing.assert_allclose(shear.exponents, [-1.0, -2.0], atol=1e-7)
    lorenz = ts.analysis.lyapunov_spectrum(
        ts.systems.Lorenz(), ic=[1, 1, 1], final_time=1.0, transient=0, backend=backend
    )
    # Liouville's formula supplies an independent finite-time volume-growth oracle.
    assert np.sum(lorenz.exponents) == pytest.approx(-(10 + 1 + 8 / 3), abs=2e-6)
