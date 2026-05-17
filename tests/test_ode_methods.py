"""Rust ERK methods agree on Lorenz dense output (reference: DP8)."""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts


@pytest.mark.parametrize(
    "method",
    ["DP5", "TSIT5", "BS3", "VERN6", "VERN7", "VERN8", "VERN9"],
)
def test_lorenz_dense_output_matches_dp8(method: str) -> None:
    """Explicit ERKs converge to DP8 sampling on Lorénz once tolerances dominate truncation error."""

    lor = ts.Lorenz()
    ic = np.array([1.0, 1.0, 1.0], dtype=float)
    rtol_req = atol_req = 1e-13
    ref = lor.integrate(
        final_time=10.0,
        dt=0.05,
        ic=ic,
        method="DP8",
        rtol=rtol_req,
        atol=atol_req,
    )
    tr = lor.integrate(
        final_time=10.0,
        dt=0.05,
        ic=ic,
        method=method,
        rtol=rtol_req,
        atol=atol_req,
    )
    np.testing.assert_allclose(tr.t, ref.t, rtol=0.0, atol=1e-14)
    rtol_y = 1e-7
    if method == "BS3":
        atol_y = 3e-8
    elif method == "VERN6":
        atol_y = 2e-8
    elif method in ("VERN7", "VERN8"):
        atol_y = 1e-8
    else:
        atol_y = 9e-9
    np.testing.assert_allclose(tr.y, ref.y, rtol=rtol_y, atol=atol_y)


@pytest.mark.parametrize("method", ["ROSENBROCK23", "ROSENBROCK34", "RODAS4"])
def test_lorenz_stiff_rust_trajectory_finite(method: str) -> None:
    """Stiff Rosenbrock family: finite trajectory on Lorenz (no DP8 comparison — chaotic)."""

    lor = ts.Lorenz()
    ic = np.array([1.0, 1.0, 1.0], dtype=float)
    tr = lor.integrate(
        final_time=0.05,
        dt=0.025,
        ic=ic,
        method=method,
        rtol=1e-5,
        atol=1e-8,
    )
    assert tr.y.shape == (tr.t.shape[0], 3)
    assert np.isfinite(tr.y).all()


def test_rk4_dense_integration_is_repeatable() -> None:
    """RK4 steps uniformly at ``dt`` — trajectory must be bitwise repeatable."""
    lor = ts.Lorenz()
    ic = np.array([1.0, 1.0, 1.0], dtype=float)
    a = lor.integrate(final_time=10.0, dt=0.05, ic=ic, method="RK4")
    b = lor.integrate(final_time=10.0, dt=0.05, ic=ic, method="RK4")
    np.testing.assert_allclose(a.t, b.t, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(a.y, b.y, rtol=0.0, atol=1e-15)


def test_dop853_alias_runs_rust_dp8() -> None:
    lor = ts.Lorenz()
    ic = np.array([1.0, 1.0, 1.0], dtype=float)
    a = lor.integrate(final_time=5.0, dt=0.1, ic=ic, method="DP8")
    b = lor.integrate(final_time=5.0, dt=0.1, ic=ic, method="DOP853")
    np.testing.assert_allclose(a.y, b.y, rtol=1e-14, atol=1e-14)
