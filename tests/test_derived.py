"""Derived-system wrappers: PoincareMap, StroboscopicMap, TangentSystem, etc."""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import ConvergenceError

# ---------------------------------------------------------------------------
# Construction / validation (fast)
# ---------------------------------------------------------------------------


def test_poincare_plane_parsing_errors() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ts.derived.PoincareMap(ts.systems.Lorenz(), plane=(7, 0.0))
    with pytest.raises(ValueError, match="non-zero"):
        ts.derived.PoincareMap(ts.systems.Lorenz(), plane=(np.zeros(3), 0.0))


class _DivergingFlow:
    """A minimal fake flow whose state goes non-finite on the first step.

    Just enough of the ``System`` surface for ``PoincareMap._advance_to_crossing``
    (``dim`` / ``state`` / ``time`` / ``step``).  No ``_rhs_numeric``, so the
    Poincaré map uses linear refinement and never tries to lower a tape.
    """

    dim = 2

    def __init__(self) -> None:
        self._t = 0.0
        self._u = np.array([1.0, 1.0])

    def state(self) -> np.ndarray:
        return self._u.copy()

    def time(self) -> float:
        return self._t

    def step(self, dt: float) -> np.ndarray:
        self._t += dt
        self._u = np.array([np.inf, np.inf])  # blow up before any crossing
        return self._u.copy()


def test_poincare_python_march_divergence_raises_convergence_error() -> None:
    """A non-finite inner step surfaces ConvergenceError, not a spun-out timeout.

    Without the divergence guard, every ``_is_crossing`` NaN-compare is false, so
    the march would spin the full ``max_time`` and mis-report "plane may miss the
    attractor".  The guard catches the real cause on the very first step.
    """
    pm = ts.derived.PoincareMap(_DivergingFlow(), plane=(0, 0.0), dt=0.1, max_time=1e6)
    with pytest.raises(ConvergenceError, match="diverged"):
        pm.step()


def test_stroboscopic_rejects_bad_period() -> None:
    with pytest.raises(ValueError, match="period"):
        ts.derived.StroboscopicMap(ts.systems.Lorenz(), period=0.0)


def test_tangent_rejects_dde() -> None:
    with pytest.raises(NotImplementedError, match="lyapunov_spectrum"):
        ts.derived.TangentSystem(ts.systems.MackeyGlass())


def test_tangent_k_validation() -> None:
    with pytest.raises(ValueError, match="k must be"):
        ts.derived.TangentSystem(ts.systems.Henon(), k=5)


def test_derived_forwards_params_and_with_params() -> None:
    pmap = ts.derived.PoincareMap(ts.systems.Lorenz(), plane=(2, 27.0))
    assert pmap.params["rho"] == 28.0
    pmap2 = pmap.with_params(rho=35.0)
    assert pmap2.params["rho"] == 35.0
    assert pmap.params["rho"] == 28.0  # original untouched
    assert isinstance(pmap2, ts.derived.PoincareMap)
    assert pmap2.plane == (2, 27.0)


# ---------------------------------------------------------------------------
# ProjectedSystem on a map (fast)
# ---------------------------------------------------------------------------


class TestProjected:
    def test_projected_step_and_state(self) -> None:
        proj = ts.derived.ProjectedSystem(ts.systems.Henon(), [1])
        proj.reinit([0.1, 0.2])
        out = proj.step()
        assert out.shape == (1,)
        assert proj.dim == 1

    def test_projected_by_name(self) -> None:
        proj = ts.derived.ProjectedSystem(ts.systems.Henon(), ["y"])
        assert proj.components == (1,)

    def test_projected_set_state_needs_complete(self) -> None:
        proj = ts.derived.ProjectedSystem(ts.systems.Henon(), [0])
        proj.reinit([0.1, 0.2])
        with pytest.raises(NotImplementedError, match="complete"):
            proj.set_state([0.5])
        proj2 = ts.derived.ProjectedSystem(ts.systems.Henon(), [0], complete=lambda u: [u[0], 0.0])
        proj2.reinit([0.1, 0.2])
        proj2.set_state([0.5])
        np.testing.assert_array_equal(proj2.system.state(), [0.5, 0.0])

    def test_projected_trajectory(self) -> None:
        proj = ts.derived.ProjectedSystem(ts.systems.Henon(), [0])
        traj = proj.run(steps=50, ic=[0.1, 0.1])
        assert traj.y.shape == (51, 1)  # the start, then 50 iterates
        assert traj.meta["projected"] == (0,)

    def test_projected_trajectory_named_access(self) -> None:
        # Regression: the projected trajectory must name its OWN columns, not the
        # inner system's full variables (which silently mislabel / IndexError).
        proj = ts.derived.ProjectedSystem(ts.systems.Henon(), ["y"])
        assert proj.variables == ("y",)
        traj = proj.run(steps=20, ic=[0.1, 0.1])
        np.testing.assert_array_equal(traj["y"], traj.y[:, 0])
        # "x" is outside the projected view → KeyError, never a mislabel/IndexError.
        with pytest.raises(KeyError):
            _ = traj["x"]

    def test_projected_trajectory_named_access_reordered(self) -> None:
        # A reordering projection: names must track the projection, not the inner
        # column order.
        proj = ts.derived.ProjectedSystem(ts.systems.Henon(), ["y", "x"])
        assert proj.variables == ("y", "x")
        traj = proj.run(steps=20, ic=[0.1, 0.1])
        np.testing.assert_array_equal(traj["y"], traj.y[:, 0])
        np.testing.assert_array_equal(traj["x"], traj.y[:, 1])


# ---------------------------------------------------------------------------
# Ensemble on a map (fast)
# ---------------------------------------------------------------------------


class TestEnsemble:
    def test_lockstep_matches_individuals(self) -> None:
        states = [[0.1, 0.2], [0.3, 0.1]]
        ens = ts.derived.Ensemble(ts.systems.Henon(), states)
        out = ens.step()
        assert out.shape == (2, 2)
        solo = ts.systems.Henon()
        solo.reinit(states[0])
        np.testing.assert_allclose(out[0], solo.step())

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            ts.derived.Ensemble(ts.systems.Henon(), [[0.1, 0.2, 0.3]])

    def test_set_states(self) -> None:
        ens = ts.derived.Ensemble(ts.systems.Henon(), [[0.1, 0.2], [0.3, 0.1]])
        ens.set_states([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_equal(ens.states(), [[0.0, 0.0], [1.0, 1.0]])


# ---------------------------------------------------------------------------
# TangentSystem on maps (fast — pure-NumPy variational iteration)
# ---------------------------------------------------------------------------


class TestTangentMap:
    def test_henon_exponents(self) -> None:
        tang = ts.derived.TangentSystem(ts.systems.Henon(), k=2)
        tang.reinit([0.1, 0.1])
        tang.step(3000)
        exps = tang.exponents()
        assert 0.2 < exps[0] < 0.6  # literature ≈ 0.42
        assert -2.0 < exps[1] < -1.0  # literature ≈ -1.62

    def test_growths_exposed(self) -> None:
        tang = ts.derived.TangentSystem(ts.systems.Henon())
        tang.reinit([0.1, 0.1])
        tang.step()
        assert tang.growths().shape == (2,)
        assert tang.deviations().shape == (2, 2)

    def test_matches_family_lyapunov(self) -> None:
        tang = ts.derived.TangentSystem(ts.systems.Henon(), k=2)
        tang.reinit([0.1, 0.1])
        tang.step(5000)
        family = ts.systems.Henon()._lyapunov_spectrum(n=5000, ic=[0.1, 0.1])
        np.testing.assert_allclose(tang.exponents(), family, atol=0.05)


# ---------------------------------------------------------------------------
# Flow wrappers (slow — engine integration runtime)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPoincareFlow:
    def test_rossler_crossings_lie_on_plane(self) -> None:
        pmap = ts.derived.PoincareMap(ts.systems.Rossler(), plane=(0, 0.0), direction=+1, dt=0.05)
        pmap.reinit([1.0, 1.0, 0.0])
        section = pmap.run(steps=30, transient=5)
        assert section.y.shape == (30, 3)
        # refined crossings sit on the plane to much better than dt accuracy
        assert np.max(np.abs(section.y[:, 0])) < 1e-6
        assert np.all(np.diff(section.t) > 0)

    def test_direction_filter(self) -> None:
        up = ts.derived.PoincareMap(ts.systems.Rossler(), plane=(0, 0.0), direction=+1, dt=0.05)
        up.reinit([1.0, 1.0, 0.0])
        # crossing with direction +1 means dx/dt > 0 → y-component of RHS... check g increased
        p1 = up.step()
        assert np.isfinite(p1).all()

    def test_is_discrete_view(self) -> None:
        pmap = ts.derived.PoincareMap(ts.systems.Rossler(), plane=(0, 0.0))
        assert pmap._is_discrete is True
        assert pmap.system._is_discrete is False


@pytest.mark.slow
class TestStroboscopicFlow:
    def test_forced_vdp_sampling(self) -> None:
        w = 0.63
        smap = ts.derived.StroboscopicMap(ts.systems.ForcedVanDerPol(), period=2 * np.pi / w)
        smap.reinit([0.1, 0.1, 0.0])
        samples = smap.run(steps=10, transient=3)
        assert samples.y.shape == (10, 3)
        # sample times are exactly one period apart
        np.testing.assert_allclose(np.diff(samples.t), 2 * np.pi / w, rtol=1e-9)


@pytest.mark.slow
class TestTangentODE:
    def test_lorenz_exponents(self) -> None:
        tang = ts.derived.TangentSystem(ts.systems.Lorenz(), k=3)
        tang.reinit([1.0, 1.0, 1.0])
        # burn-in
        for _ in range(200):
            tang.step(0.1)
        tang._sum_growths[:] = 0.0
        tang._elapsed = 0.0
        for _ in range(2000):
            tang.step(0.1)
        exps = tang.exponents()
        assert abs(exps[0] - 0.906) < 0.45
        assert abs(exps[1]) < 0.2
        assert exps[2] < -10.0


# ---------------------------------------------------------------------------
# Engine ODE-Lyapunov kernel (stream perf/ode-lyapunov-engine)
# ---------------------------------------------------------------------------
#
# The whole burn-in + averaging Benettin loop runs in one Rust engine call for an
# explicit kernel on the compiled engine (interp/jit).  These pin the correctness
# contract: interp == jit bit-for-bit, and the engine fast path agrees with the
# per-chunk reference/Python loop to the documented chaotic-flow tolerance.


@pytest.mark.slow
class TestTangentEngineLyapunov:
    """The engine ODE-Lyapunov fast path (stream perf/ode-lyapunov-engine)."""

    def test_interp_equals_jit_bit_for_bit(self) -> None:
        # The engine kernel drives two numerically-identical evaluators over the
        # same lowered extended-variational tape, so the spectrum is bit-identical.
        pytest.importorskip("tsdynamics._rust")
        kw = dict(dt=0.05, transient=30.0, final_time=200.0, ic=[1.0, 1.0, 1.0])
        interp = ts.systems.Lorenz()._lyapunov_spectrum(backend="interp", **kw)
        jit = ts.systems.Lorenz()._lyapunov_spectrum(backend="jit", **kw)
        np.testing.assert_array_equal(
            interp, jit, err_msg="engine ODE Lyapunov: interp != jit (must be bit-for-bit)"
        )

    def test_engine_matches_reference_to_tolerance(self) -> None:
        """The engine kernel and the pure-Python oracle find the same attractor.

        Two float-distinct computations of the *same* finite-time Benettin
        estimate over a chaotic flow.  They trace different trajectories on the
        same attractor, so the comparable quantity is the attractor, not the
        digits.

        The raw-spectrum bound is set by the **finite-time statistical scatter**,
        not by the integration tolerance.  Measured over 8 initial conditions at
        these settings (``T=300``): ``max|engine - reference|`` has median 0.0056
        / max 0.0286 at the v6 default ``rtol=1e-9``, and median 0.0094 / max
        0.0155 at the pre-v6 ``rtol=1e-6``.  The two distributions overlap and
        **both exceed 1e-2** — the old ``atol=1e-2`` was a latent flake that held
        only for this one pinned initial condition, not a contract.  Sweeping
        ``rtol`` from ``1e-5`` to ``1e-10`` at the pinned IC gives 0.0041, 0.0040,
        0.0165, 0.0087, 0.0257, 0.0076: non-monotonic, i.e. scatter.  Widening
        ``dt``/``final_time`` is the only thing that shrinks it (``T=1000`` →
        0.0089, ``T=3000`` → 0.0040), and tripling this slow test is not worth it.

        So the raw bound is loosened to the measured scatter, and the test is
        *strengthened* with two quantities that are not scatter-dominated: the
        Kaplan–Yorke dimension (max measured gap 0.0015, a 6x margin on 1e-2) and
        the **exact** trace identity — for Lorenz, ``sum(lambda)`` must equal the
        constant divergence ``-(sigma + 1 + beta)``.  That residual *does* track
        the solver tolerance monotonically (2.1e-6 / 2.4e-7 / 2.3e-8 / 2.2e-9 /
        2.2e-10 at ``rtol=1e-6 … 1e-10``), so it cannot be satisfied by luck.
        """
        pytest.importorskip("tsdynamics._rust")
        kw = dict(dt=0.05, transient=50.0, final_time=300.0, ic=[1.0, 1.0, 1.0])
        engine = ts.systems.Lorenz()._lyapunov_spectrum(backend="interp", **kw)
        reference = ts.systems.Lorenz()._lyapunov_spectrum(backend="reference", **kw)
        np.testing.assert_allclose(engine, reference, rtol=0.0, atol=5e-2)
        # Kaplan–Yorke dimension is preserved across the two paths.
        assert (
            abs(
                ts.analysis.kaplan_yorke_dimension(engine)
                - ts.analysis.kaplan_yorke_dimension(reference)
            )
            < 1e-2
        )
        # The exact invariant: the Lorenz flow's divergence is the constant
        # -(sigma + 1 + beta), so the spectrum must sum to it on BOTH paths.
        lor = ts.systems.Lorenz()
        trace = -(lor.params["sigma"] + 1.0 + lor.params["beta"])
        for name, spec in (("engine", engine), ("reference", reference)):
            assert abs(float(np.sum(spec)) - trace) < 1e-5, (
                f"{name}: sum(lambda) = {np.sum(spec)!r} != divergence {trace!r}"
            )

    def test_partial_spectrum_k_less_than_dim(self) -> None:
        # k < dim takes the same engine kernel (the leading exponent is positive).
        pytest.importorskip("tsdynamics._rust")
        exps = ts.systems.Lorenz()._lyapunov_spectrum(
            dt=0.05, transient=30.0, final_time=200.0, ic=[1.0, 1.0, 1.0], k=2
        )
        assert exps.shape == (2,)
        assert exps[0] > 0.5  # the positive exponent (~0.9)

    def test_stiff_default_keeps_the_slow_path(self) -> None:
        # An implicit (stiff) default method declines the engine kernel
        # (make_ode_stepper rejects it) and keeps the per-chunk loop — still finite.
        pytest.importorskip("tsdynamics._rust")
        sys = ts.systems.Oregonator()
        exps = sys._lyapunov_spectrum(dt=0.05, transient=10.0, final_time=40.0, k=2)
        assert np.all(np.isfinite(exps))


# ---------------------------------------------------------------------------
# Pickling — every derived wrapper must cross a process boundary
# ---------------------------------------------------------------------------


def _wrappers() -> dict[str, object]:
    """One instance of each derived wrapper, all around the same pinned flow."""
    inner = ts.systems.Rossler(ic=[1.0, 1.0, 1.0])
    return {
        "PoincareMap": ts.derived.PoincareMap(inner.copy(), plane=("y", 0.0, "up")),
        "StroboscopicMap": ts.derived.StroboscopicMap(inner.copy(), period=6.0),
        "TangentSystem": ts.derived.TangentSystem(inner.copy(), k=2),
        "Ensemble": ts.derived.Ensemble(inner.copy(), np.arange(12.0).reshape(4, 3)),
        "ProjectedSystem": ts.derived.ProjectedSystem(inner.copy(), [0, 1]),
    }


@pytest.mark.parametrize("name", sorted(_wrappers()))
def test_derived_wrappers_round_trip_through_pickle(name: str) -> None:
    """A parallel bifurcation sweep pickles its wrapper to reach the workers.

    ``PoincareMap`` used to fail here: it caches ``system._rhs_numeric()`` for
    Hermite refinement, and that used to be a *local function*, which pickle
    refuses.  The cached RHS is now a module-level callable.
    """
    import pickle

    wrapper = _wrappers()[name]
    restored = pickle.loads(pickle.dumps(wrapper))
    assert type(restored) is type(wrapper)
    # ``Ensemble`` names its inner system ``template``; the rest ``system``.
    inner_attr = "system" if hasattr(wrapper, "system") else "template"
    inner, restored_inner = getattr(wrapper, inner_attr), getattr(restored, inner_attr)
    assert type(restored_inner) is type(inner)
    assert restored.dim == wrapper.dim
    np.testing.assert_array_equal(restored_inner.params.as_tuple(), inner.params.as_tuple())


def test_pickled_poincare_map_keeps_its_hermite_rhs_and_the_same_section() -> None:
    """The restored wrapper must not silently degrade to linear refinement."""
    import pickle

    pmap = ts.derived.PoincareMap(ts.systems.Rossler(ic=[1.0, 1.0, 1.0]), plane=("y", 0.0, "up"))
    restored = pickle.loads(pickle.dumps(pmap))
    assert restored._rhs is not None, "the Hermite RHS must survive the round trip"

    # Independent truth for the RHS itself: Rossler at (1, 2, 3) with the
    # default a=0.2, b=0.2, c=5.7 is (-y-z, x+a*y, b+z*(x-c)) = (-5, 1.4, -13.9).
    np.testing.assert_allclose(restored._rhs(np.array([1.0, 2.0, 3.0]), 0.0), [-5.0, 1.4, -13.9])

    # ... and end to end: the same 50 crossings, bit for bit.
    np.testing.assert_array_equal(pmap.run(50).y, restored.run(50).y)
