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


def test_auto_backend_uses_diffsol_when_available() -> None:
    """With pydiffsol installed, backend='auto' routes to diffsol."""
    traj = ts.Lorenz().integrate(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0], backend="auto")
    assert traj.meta["backend"] == "diffsol"


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
        yj = (
            cls()
            .integrate(ic=ic, final_time=1.5, dt=0.03, method="dop853", rtol=1e-10, atol=1e-12)
            .y
        )
        yd = (
            cls()
            .integrate(
                ic=ic,
                final_time=1.5,
                dt=0.03,
                backend="diffsol",
                method="LSODA",
                rtol=1e-10,
                atol=1e-12,
            )
            .y
        )
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


@pytest.mark.full
def test_diffsol_integrates_full_catalogue() -> None:
    """
    The gate for flipping the default backend to diffsol: *every* built-in ODE
    integrates on diffsol (BDF) to a bounded, finite trajectory.  Runs nightly
    (``-m full``) with the diffsol extra.

    Numeric agreement with JiTCODE is checked separately on the curated sample
    (``test_cross_validation_over_sample``); here we only assert each diffsol
    trajectory is finite.  ``HARD_TO_INTEGRATE | DIFFSOL_SKIP`` is honoured as
    an escape hatch for systems that genuinely cannot be integrated, but both
    sets are currently empty — the whole ODE catalogue integrates on diffsol.
    """
    import sys
    import zlib
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from _sampling import DIFFSOL_SKIP, HARD_TO_INTEGRATE

    from tsdynamics import registry

    skip = set(HARD_TO_INTEGRATE) | set(DIFFSOL_SKIP)
    bad = []
    for e in registry.all_systems(family="ode"):
        if e.name in skip:
            continue
        # Deterministic per-system IC (default_ic where set, else a fixed draw).
        np.random.seed(zlib.crc32(e.name.encode()) & 0xFFFFFFFF)
        ic = e.cls().resolve_ic(None)
        try:
            y = (
                e.cls()
                .integrate(
                    ic=ic,
                    final_time=1.0,
                    dt=0.05,
                    backend="diffsol",
                    method="LSODA",
                    rtol=1e-8,
                    atol=1e-10,
                )
                .y
            )
        except Exception as exc:  # noqa: BLE001 — record which system & why
            bad.append((e.name, f"error: {str(exc).splitlines()[-1][:50]}"))
            continue
        if not np.all(np.isfinite(y)):
            bad.append((e.name, "non-finite"))
    assert not bad, f"diffsol failed to integrate: {bad}"
