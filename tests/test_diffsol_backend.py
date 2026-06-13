"""
Cross-validation of the experimental diffsol backend against JiTCODE.

Skipped entirely when the ``tsdynamics[diffsol]`` extra is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydiffsol")

import tsdynamics as ts
from tsdynamics.backends import diffsol as dsl

# ---------------------------------------------------------------------------
# Translator (fast)
# ---------------------------------------------------------------------------


def test_to_diffsl_lorenz_structure() -> None:
    code, control = dsl.to_diffsl(ts.Lorenz())
    assert control == ["sigma", "rho", "beta"]
    assert "in_i {" in code and "u_i {" in code and "F_i {" in code
    assert "ic0" in code and "ic2" in code


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        ts.Lorenz().integrate(final_time=1.0, dt=0.1, backend="quantum")


# ---------------------------------------------------------------------------
# Solving + cross-validation (no C compiler needed — pure LLVM JIT)
# ---------------------------------------------------------------------------


def test_lorenz_diffsol_short_integration() -> None:
    traj = ts.Lorenz().integrate(
        final_time=2.0, dt=0.01, ic=[1.0, 1.0, 1.0], backend="diffsol", rtol=1e-9, atol=1e-11
    )
    assert traj.y.shape == (201, 3)  # grid includes the endpoint
    assert np.all(np.isfinite(traj.y))
    assert traj.meta["backend"] == "diffsol"


def test_param_change_reuses_compiled_module() -> None:
    a = ts.Lorenz().integrate(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0], backend="diffsol")
    before = dict(dsl._ODE_CACHE)
    b = ts.Lorenz(params={"rho": 35.0}).integrate(
        final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0], backend="diffsol"
    )
    assert dict(dsl._ODE_CACHE) == before  # no recompilation
    assert not np.allclose(a.y[-1], b.y[-1])  # but different dynamics


@pytest.mark.slow
@pytest.mark.parametrize("name,ic", [("Lorenz", [1.0, 1.0, 1.0]), ("Rossler", [1.0, 0.0, 0.0])])
def test_cross_validation_against_jitcode(name: str, ic: list) -> None:
    """Both backends must agree to tight tolerance over a short window."""
    sys_a = getattr(ts, name)()
    sys_b = getattr(ts, name)()
    kw = dict(final_time=5.0, dt=0.01, ic=ic, rtol=1e-10, atol=1e-12)
    ref = sys_a.integrate(method="dop853", **kw)
    alt = sys_b.integrate(backend="diffsol", **kw)
    np.testing.assert_allclose(alt.y, ref.y, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_cross_validation_over_sample() -> None:
    """
    diffsol (BDF) reproduces JiTCODE (dop853) across the curated ODE sample.

    Short horizon + tight tolerances keep chaotic sensitivity from masking a
    real translator/solver discrepancy; max abs deviation must stay < 1e-3.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from _sampling import INTEGRATION_SAMPLE

    from tsdynamics import registry

    bad = []
    for name in INTEGRATION_SAMPLE:
        cls = registry.get(name).cls
        ic = cls().resolve_ic(None)
        yj = cls().integrate(
            ic=ic, final_time=1.5, dt=0.03, method="dop853", rtol=1e-10, atol=1e-12
        ).y
        yd = cls().integrate(
            ic=ic, final_time=1.5, dt=0.03, backend="diffsol", method="LSODA",
            rtol=1e-10, atol=1e-12,
        ).y
        n = min(len(yj), len(yd))
        dev = float(np.max(np.abs(yj[:n] - yd[:n])))
        if dev >= 1e-3:
            bad.append((name, dev))
    assert not bad, f"diffsol disagrees with jitcode on: {bad}"


@pytest.mark.slow
def test_stiff_solver_path() -> None:
    """The BDF mapping handles a stiff-ish problem."""
    traj = ts.Lorenz().integrate(
        final_time=2.0, dt=0.01, ic=[1.0, 1.0, 1.0], backend="diffsol", method="LSODA"
    )
    assert np.all(np.isfinite(traj.y))
