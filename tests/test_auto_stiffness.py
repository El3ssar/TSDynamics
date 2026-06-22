"""Auto-stiffness wiring: ``method="auto"`` → :func:`solvers.recommend` (FIX-AUTOSTIFF).

The selection *layer* (``solvers.is_stiff`` / ``solvers.recommend``) already
existed; this stream wires it into the integrate seam so a literal
``method="auto"`` probes the Jacobian spectrum at the start state and picks the
implicit ``bdf`` kernel on a stiff RHS, the explicit ``rk45`` otherwise.

The integration-level tests are *failing-first*: before the wiring,
``integrate(method="auto")`` raised ``ValueError`` ("unknown solver method
'auto'") from the registry resolver — so asserting it now selects a kernel and
returns a finite trajectory fails on the old code.

Most tests run on ``backend="reference"`` (the pure-Python SciPy oracle), so the
selection logic is exercised without the compiled engine; the parity test at the
end is engine-gated and asserts ``interp == jit`` bit-for-bit.

The probe is read at the canonical Oregonator IC ``[1, 1, 1]`` (Field–Noyes),
where the one-point stiffness heuristic correctly fires — the heuristic is
IC-dependent by construction (see :func:`tsdynamics.solvers.is_stiff`), which is
why a reliably-stiff catalogue system also declares ``_default_method``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import solvers

# Field–Noyes Oregonator IC where the a-priori one-point heuristic detects the
# structural stiffness (mu=1e-6, epsilon=1e-2 put a ~1e6 factor on the fast mode).
OREGONATOR_IC = [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Selection layer (engine-free): recommend() picks bdf for stiff, rk45 else
# ---------------------------------------------------------------------------


def test_recommend_picks_bdf_for_stiff_oregonator() -> None:
    """``recommend`` returns the implicit ``bdf`` kernel for the stiff Oregonator."""
    res = solvers.recommend(ts.systems.Oregonator(), ic=OREGONATOR_IC)
    assert res.name == "bdf"
    assert res.is_implicit
    assert res.needs_jacobian


def test_recommend_picks_rk45_for_nonstiff_lorenz() -> None:
    """``recommend`` returns the explicit ``rk45`` kernel for the non-stiff Lorenz."""
    res = solvers.recommend(ts.systems.Lorenz())
    assert res.name == "rk45"
    assert not res.is_implicit


def test_is_stiff_separates_oregonator_from_lorenz() -> None:
    """The one-point stiffness verdict is True for Oregonator, False for Lorenz."""
    assert solvers.is_stiff(ts.systems.Oregonator(), ic=OREGONATOR_IC) is True
    assert solvers.is_stiff(ts.systems.Lorenz()) is False


# ---------------------------------------------------------------------------
# Integration wiring (failing-first): integrate(method="auto") selects a kernel
# ---------------------------------------------------------------------------


def test_auto_selects_bdf_on_oregonator() -> None:
    """``method="auto"`` selects ``bdf`` on the stiff Oregonator (was: raised)."""
    traj = ts.systems.Oregonator().integrate(
        final_time=5.0, dt=0.1, ic=OREGONATOR_IC, method="auto", backend="reference"
    )
    assert traj.meta["method"] == "bdf"
    assert np.isfinite(traj.y).all()


def test_auto_selects_rk45_on_lorenz() -> None:
    """``method="auto"`` selects ``rk45`` on the non-stiff Lorenz (was: raised)."""
    traj = ts.systems.Lorenz().integrate(
        final_time=5.0, dt=0.05, ic=[1.0, 1.0, 1.0], method="auto", backend="reference"
    )
    assert traj.meta["method"] == "rk45"
    assert np.isfinite(traj.y).all()


def test_auto_is_case_insensitive() -> None:
    """``"AUTO"`` / ``"Auto"`` normalise to the auto path, not an unknown-method error."""
    for spelling in ("AUTO", "Auto", " auto "):
        traj = ts.systems.Lorenz().integrate(
            final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0], method=spelling, backend="reference"
        )
        assert traj.meta["method"] == "rk45"


def test_auto_matches_the_explicitly_named_kernel() -> None:
    """Auto-on-Oregonator gives the *same* trajectory as the explicit ``bdf`` it picks."""
    kw = dict(final_time=5.0, dt=0.1, ic=OREGONATOR_IC, backend="reference")
    auto = ts.systems.Oregonator().integrate(method="auto", **kw)
    explicit = ts.systems.Oregonator().integrate(method="bdf", **kw)
    assert auto.meta["method"] == explicit.meta["method"] == "bdf"
    np.testing.assert_array_equal(auto.y, explicit.y)


# ---------------------------------------------------------------------------
# Default unchanged unless auto
# ---------------------------------------------------------------------------


def test_default_method_unchanged_without_auto() -> None:
    """Omitting ``method=`` keeps each system's own default — auto changes nothing here."""
    # Lorenz keeps the global default RK45 -> rk45.
    lor = ts.systems.Lorenz().integrate(
        final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0], backend="reference"
    )
    assert lor.meta["method"] == "rk45"
    # Oregonator keeps its declared _default_method = "bdf" (NOT via the auto probe).
    assert ts.systems.Oregonator._default_method == "bdf"
    oreg = ts.systems.Oregonator().integrate(
        final_time=5.0, dt=0.1, ic=OREGONATOR_IC, backend="reference"
    )
    assert oreg.meta["method"] == "bdf"


def test_explicit_method_still_resolves_unchanged() -> None:
    """A named, non-auto method resolves through the registry exactly as before."""
    traj = ts.systems.Lorenz().integrate(
        final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0], method="dop853", backend="reference"
    )
    assert traj.meta["method"] == "dop853"


def test_auto_is_a_noop_on_a_map() -> None:
    """A map iterates without a solver kernel, so ``method="auto"`` must not raise."""
    from tsdynamics.engine import run
    from tsdynamics.engine.problem import build_problem

    # Pin a deterministic in-basin IC: Henon has no default_ic, and a random draw
    # (what a bare build_problem resolves) escapes the attractor's basin and
    # diverges ~25% of the time — unrelated to method= (the map branch ignores it).
    problem = build_problem(ts.systems.Henon(), ic=[0.1, 0.1])
    traj = run.integrate(problem, final_time=20, method="auto", backend="reference")
    assert np.isfinite(traj.y).all()


# ---------------------------------------------------------------------------
# Engine invariant: interp == jit bit-for-bit on the auto-selected kernel
# ---------------------------------------------------------------------------


def test_auto_interp_equals_jit_bitforbit() -> None:
    """The auto path selects the same kernel and integrates identically on interp/jit."""
    pytest.importorskip("tsdynamics._rust")
    kw = dict(final_time=5.0, dt=0.05, ic=OREGONATOR_IC, method="auto")
    interp = ts.systems.Oregonator().integrate(backend="interp", **kw)
    jit = ts.systems.Oregonator().integrate(backend="jit", **kw)
    assert interp.meta["method"] == jit.meta["method"] == "bdf"
    np.testing.assert_array_equal(interp.y, jit.y)
