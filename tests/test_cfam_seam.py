"""C-FAM: the shared engine-dispatch seam + SDE registry detection.

Stream C-FAM unifies how every family reaches the Rust engine.  A single
``_default_backend`` knob plus a thin :meth:`SystemBase._dispatch` template route
ODE / DDE / map integration through the one engine seam
(:func:`tsdynamics.engine.run.integrate`); diagonal-Itô SDEs keep their dedicated
seed-carrying ``run.sde_*`` seam (``run.integrate`` cannot carry the noise
seed/step).  The registry now also detects the ``sde`` family.

Most of these run on the pure-Python ``reference`` backend, so they need no
compiled engine; the few that assert the compiled ``interp`` path skip without
``tsdynamics._rust``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.engine import run
from tsdynamics.engine.problem import sde_problem

# ---------------------------------------------------------------------------
# An SDE fixture (a StochasticSystem subclass; registers as a non-builtin sde)
# ---------------------------------------------------------------------------


class _SeamGBM(ts.StochasticSystem):
    params = {"mu": 0.1, "sigma": 0.3}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _drift(y, t, mu, sigma):
        return [mu * y(0)]

    @staticmethod
    def _diffusion(y, t, mu, sigma):
        return [sigma * y(0)]


# ---------------------------------------------------------------------------
# _default_backend — one knob per family, the M3 flip point
# ---------------------------------------------------------------------------


def test_default_backend_per_family() -> None:
    """Post-M3 every family defaults to the Rust engine interpreter (the one knob)."""
    assert ts.ContinuousSystem._default_backend == "interp"
    assert ts.DelaySystem._default_backend == "interp"
    assert ts.DiscreteMap._default_backend == "interp"
    assert ts.StochasticSystem._default_backend == "interp"
    # The abstract base keeps the wheel-free oracle as its default; every concrete
    # family overrides it to the engine interpreter above.
    from tsdynamics.families.base import SystemBase

    assert SystemBase._default_backend == "reference"


def test_backend_none_resolves_to_family_default_ode() -> None:
    """``backend=None`` is the same as the family default (the engine seam)."""
    pytest.importorskip("tsdynamics._rust")
    lor = ts.Lorenz()
    kw = dict(final_time=1.0, dt=0.5, ic=[1.0, 1.0, 1.0])
    explicit = lor.integrate(backend="interp", **kw)
    implicit = lor.integrate(backend=None, **kw)
    np.testing.assert_array_equal(explicit.y, implicit.y)
    # The default path now *is* the engine seam.
    assert implicit.meta.get("engine") == "rust"


def test_backend_none_resolves_to_family_default_map() -> None:
    h = ts.Henon()
    a = h.iterate(steps=20, ic=[0.1, 0.1], backend="reference")
    b = h.iterate(steps=20, ic=[0.1, 0.1], backend=None)
    np.testing.assert_array_equal(a.y, b.y)


# ---------------------------------------------------------------------------
# _dispatch — every family funnels its engine path through run.integrate
# ---------------------------------------------------------------------------


def test_dispatch_routes_through_run_integrate(monkeypatch) -> None:
    """``SystemBase._dispatch`` calls ``run.integrate`` with the system + backend."""
    seen: dict = {}

    def fake_integrate(system, **kwargs):
        seen["system"] = system
        seen["kwargs"] = kwargs
        return "SENTINEL"

    monkeypatch.setattr(run, "integrate", fake_integrate)
    lor = ts.Lorenz()
    out = lor._dispatch(backend="interp", final_time=5.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    assert out == "SENTINEL"
    assert seen["system"] is lor
    assert seen["kwargs"]["backend"] == "interp"
    assert seen["kwargs"]["final_time"] == 5.0


def test_ode_engine_backends_route_through_the_seam(monkeypatch) -> None:
    """ODE ``integrate(backend='interp'/'jit'/'reference')`` goes through ``_dispatch``."""
    calls: list[str] = []

    def fake_dispatch(self, *, backend, **kwargs):
        calls.append(backend)
        return "ROUTED"

    monkeypatch.setattr(ts.ContinuousSystem, "_dispatch", fake_dispatch, raising=True)
    lor = ts.Lorenz()
    for be in ("interp", "jit", "reference"):
        assert lor.integrate(backend=be, final_time=1.0, dt=0.5, ic=[1.0, 1.0, 1.0]) == "ROUTED"
    assert calls == ["interp", "jit", "reference"]


# ---------------------------------------------------------------------------
# ODE reference path through the family (no compiled engine needed)
# ---------------------------------------------------------------------------


def test_ode_reference_via_family_carries_engine_provenance_with_ic_t0() -> None:
    """``Lorenz().integrate(backend='reference')`` runs the lowered tape via the seam.

    The engine-path provenance now threads ``ic`` and ``t0`` (the C-FAM fix), so
    an engine-produced Trajectory carries the same provenance a v2 run would.
    """
    lor = ts.Lorenz()
    traj = lor.integrate(
        backend="reference", final_time=2.0, dt=0.1, t0=0.0, ic=[1.0, 1.0, 1.0], method="DOP853"
    )
    assert traj.meta["engine"] == "rust"
    assert traj.meta["family"] == "ode"
    assert traj.meta["backend"] == "reference"
    assert "ic" in traj.meta and "t0" in traj.meta
    np.testing.assert_array_equal(traj.meta["ic"], [1.0, 1.0, 1.0])
    assert traj.meta["t0"] == 0.0


def test_ode_reference_matches_scipy_on_a_short_window() -> None:
    """The seam's lowered-tape integration agrees with an independent SciPy oracle."""
    from scipy.integrate import solve_ivp

    lor = ts.Lorenz()
    kw = dict(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0], rtol=1e-10, atol=1e-12)
    seam = lor.integrate(backend="reference", method="DOP853", **kw)
    rhs = lor._rhs_numeric()
    sol = solve_ivp(
        lambda t, y: rhs(y, t),
        (0.0, 2.0),
        [1.0, 1.0, 1.0],
        method="DOP853",
        t_eval=seam.t,
        rtol=1e-10,
        atol=1e-12,
    )
    assert seam.y.shape == sol.y.T.shape
    np.testing.assert_allclose(seam.y, sol.y.T, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Map engine path provenance — steps + ic threaded (the C-FAM fix)
# ---------------------------------------------------------------------------


def test_map_engine_provenance_carries_steps_and_ic() -> None:
    h = ts.Henon()
    traj = h.iterate(steps=30, ic=[0.1, 0.1], backend="reference")
    assert traj.meta["engine"] == "rust"
    assert traj.meta["family"] == "map"
    assert traj.meta["steps"] == 30
    np.testing.assert_array_equal(traj.meta["ic"], [0.1, 0.1])


# ---------------------------------------------------------------------------
# DDE / SDE: the seam's family boundaries
# ---------------------------------------------------------------------------


def test_dde_reference_is_rejected_through_the_seam() -> None:
    """A DDE has no reference integrator — the seam refuses it loudly."""
    with pytest.raises(NotImplementedError, match="reference"):
        ts.MackeyGlass().integrate(backend="reference", final_time=1.0, dt=0.5, ic=[1.0])


def test_run_integrate_refuses_sde_without_the_engine() -> None:
    """``run.integrate`` cannot carry the SDE seed/step — it refuses before the FFI.

    Reaches the rejection on the pure-Python path (no ``tsdynamics._rust`` needed):
    the SDE branch raises before any engine call.
    """
    prob = sde_problem(_SeamGBM(), ic=[1.0])
    with pytest.raises(NotImplementedError, match="SDE"):
        run.integrate(prob, final_time=1.0, dt=0.01, backend="interp")
    with pytest.raises(NotImplementedError, match="SDE"):
        run.ensemble(prob, np.ones((3, 1)), final_time=1.0, backend="interp")


def test_dde_engine_absent_gives_install_guidance(monkeypatch) -> None:
    """When ``tsdynamics._rust`` is absent, the DDE seam steers users to install the wheel.

    The shared ``run._engine`` accessor recommends ``backend='reference'``, which a
    DDE rejects (no pure-Python delay integrator) — so ``_run_dde`` re-raises with
    the DDE-correct guidance (install the compiled wheel) instead.
    """
    from tsdynamics.engine.run import EngineNotAvailableError

    def boom():
        raise EngineNotAvailableError("simulated: extension not built")

    monkeypatch.setattr(run, "_engine", boom)
    with pytest.raises(EngineNotAvailableError, match="compiled wheel"):
        ts.MackeyGlass().integrate(backend="interp", final_time=1.0, dt=0.5, ic=[1.0])


def test_run_integrate_and_ensemble_refuse_dde_ensemble() -> None:
    """The engine has no batched method-of-steps path; ``run.ensemble`` refuses a DDE."""
    from tsdynamics.engine.problem import dde_problem

    prob = dde_problem(ts.MackeyGlass(), ic=[1.0])
    with pytest.raises(NotImplementedError, match="DDE"):
        run.ensemble(prob, np.ones((3, 1)), final_time=1.0, backend="interp")


# ---------------------------------------------------------------------------
# SDE registry detection — the family is now tagged 'sde'
# ---------------------------------------------------------------------------


def test_sde_subclass_is_detected_as_family_sde() -> None:
    entry = registry.get("_SeamGBM", builtin=False)
    assert entry.family == "sde"
    assert not entry.is_builtin


def test_sde_subclass_excluded_from_builtin_sweeps() -> None:
    """User SDE classes register but stay out of the builtin (test/docs) sweeps."""
    builtin_names = {e.name for e in registry.all_systems()}
    assert "_SeamGBM" not in builtin_names
    # ... but are visible when builtins are not the only filter.
    all_sde = {e.name for e in registry.all_systems(family="sde", builtin=None)}
    assert "_SeamGBM" in all_sde


def test_sde_detection_does_not_disturb_builtin_family_counts() -> None:
    counts = registry.families()  # builtin only
    assert counts == {"ode": 118, "dde": 5, "map": 26}
    assert "sde" not in counts  # no built-in SDE systems yet


def test_drift_only_class_is_registrable() -> None:
    """``_has_concrete_rhs`` now accepts ``_drift`` (the SDE marker)."""
    from tsdynamics.registry import _has_concrete_rhs

    assert _has_concrete_rhs(_SeamGBM)


# ---------------------------------------------------------------------------
# make_output_grid — the single hoisted output-grid helper
# ---------------------------------------------------------------------------


def test_make_output_grid_is_the_single_definition() -> None:
    """Every family and the engine seam share one ``make_output_grid``.

    The four byte-identical ``_make_t_eval`` copies are gone; importing the helper
    from any layer resolves to the one ``utils.grids`` definition.
    """
    from tsdynamics.engine import run as run_mod
    from tsdynamics.families import continuous, delay, stochastic
    from tsdynamics.utils import make_output_grid as canonical
    from tsdynamics.utils.grids import make_output_grid

    assert make_output_grid is canonical
    # The private per-family copies were removed (no shadowing definitions).
    assert not hasattr(continuous, "_make_t_eval")
    assert not hasattr(delay, "_make_t_eval")
    assert not hasattr(stochastic, "_make_t_eval")
    assert not hasattr(run_mod, "_make_t_eval")
    # Post-M3 the ODE/DDE families delegate their output grid to ``run.integrate``;
    # the modules that still build a grid directly (the engine seam and the SDE
    # family) import the one canonical helper.
    assert run_mod.make_output_grid is canonical
    assert stochastic.make_output_grid is canonical


def test_make_output_grid_samples_endpoint_inclusive() -> None:
    g = ts.utils.make_output_grid(0.0, 1.0, 0.3)
    assert g[0] == 0.0
    assert g[-1] == 1.0  # tf appended even when dt does not divide the span
    # An exactly-dividing dt needs no append.
    np.testing.assert_array_equal(ts.utils.make_output_grid(0.0, 1.0, 0.5), [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# Compiled-engine path (skipped without tsdynamics._rust)
# ---------------------------------------------------------------------------


def test_ode_interp_via_family_matches_reference() -> None:
    pytest.importorskip("tsdynamics._rust")
    lor = ts.Lorenz()
    kw = dict(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0], method="dop853", rtol=1e-10, atol=1e-12)
    interp = lor.integrate(backend="interp", **kw)
    reference = lor.integrate(backend="reference", **kw)
    assert interp.meta["engine"] == "rust"
    np.testing.assert_allclose(interp.y, reference.y, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", ["Lorenz", "Rossler", "Halvorsen"])
def test_representative_odes_integrate_on_the_rust_engine(name) -> None:
    """A spread of built-in ODEs integrate via the Rust engine through the family."""
    pytest.importorskip("tsdynamics._rust")
    sys = getattr(ts, name)()
    ic = sys.resolve_ic(None)
    traj = sys.integrate(backend="interp", final_time=2.0, dt=0.05, ic=ic, method="rk45")
    assert traj.y.shape[1] == sys.dim
    assert np.all(np.isfinite(traj.y))
    assert traj.meta["engine"] == "rust"
