"""Changing time units must not silently erase a variational integration."""

import numpy as np
import pytest

import tsdynamics as ts

pytest.importorskip("tsdynamics._rust")


class LinearRate(ts.ContinuousSystem):
    variables = ("x",)
    params = {"rate": 1.0}
    _default_ic = (1e-8,)

    @staticmethod
    def _equations(y, t, rate):
        return [rate * y(0)]


@pytest.mark.parametrize("backend", ["jit", "interp"])
@pytest.mark.parametrize("rate", [-1e12, -1.0, -1e-12, 1e-12, 1.0, 1e12])
def test_public_linear_rate_is_independent_of_time_units(backend, rate):
    duration = 1.0 / abs(rate)
    result = ts.analysis.lyapunov_spectrum(
        LinearRate(params={"rate": rate}),
        final_time=duration,
        transient=duration,
        dt=duration / 5,
        backend=backend,
        rtol=1e-9,
        atol=1e-11,
    )
    # Exact fundamental solution exp(rate*t): average log growth is rate.
    np.testing.assert_allclose(np.asarray(result) / rate, [1.0], rtol=1e-7)


@pytest.mark.parametrize("backend", ["jit", "interp"])
def test_positive_window_shorter_than_renormalisation_interval_runs_once(backend):
    result = ts.analysis.lyapunov_spectrum(
        LinearRate(params={"rate": 1e12}),
        final_time=1e-13,
        transient=0,
        dt=1.0,
        backend=backend,
        rtol=1e-9,
        atol=1e-11,
    )
    assert float(result[0]) / 1e12 == pytest.approx(1.0, rel=1e-7)
