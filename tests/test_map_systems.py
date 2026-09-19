"""
Tests for discrete map systems (``DiscreteMap`` subclasses).

All sweeps are registry-driven: a new map is covered automatically.
Iteration runs on the engine; the longer iterate/Lyapunov sweeps are marked
``slow`` on runtime grounds.

Scope note: as in ``test_ode_systems``, the iteration sweep here checks shape
and finiteness only — the pre-v6 Baker, whose every orbit collapsed to ``(0, 0)``
within ~53 iterations, passed it.  The dynamical content (bounded,
non-degenerate, recurrent, claim-consistent) is asserted per system by
``tests/test_catalogue_dynamics.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import MAP_LYAPUNOV_EXCLUDE

import tsdynamics as ts

# ---------------------------------------------------------------------------
# Instantiation (fast)
# ---------------------------------------------------------------------------


def test_map_instantiation(map_entry) -> None:
    m = map_entry.cls()
    assert m.dim is not None and m.dim > 0


def test_map_params_as_attributes(map_entry) -> None:
    m = map_entry.cls()
    for key in m.params:
        assert hasattr(m, key)


def test_tinkerbell_uses_default_ic() -> None:
    """Tinkerbell sets ``default_ic`` because random ICs always escape the basin."""
    import tsdynamics as ts

    tb = ts.systems.Tinkerbell()
    assert tb.ic is None
    assert tb.info.default_ic is not None
    traj = tb.run(steps=100)
    np.testing.assert_array_almost_equal(tb.ic, ts.systems.Tinkerbell._default_ic)
    assert np.all(np.isfinite(traj.y))


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

_STEPS = 200


@pytest.mark.slow
def test_map_iterate_shape_and_finiteness(map_entry) -> None:
    m = map_entry.cls()
    # PIN THE IC.  A map with no declared ``_default_ic`` draws a fresh random
    # start on every call, so this swept assertion was a dice roll over a finite
    # basin: ``Bogdanov`` (eps=0, mu=0) failed here roughly once in a thousand
    # runs, on a diff that had not touched it.  Seeding makes the sweep a
    # property of the map rather than of the draw — and it is only *reliable*
    # because ``run(seed=)`` now seeds the divergence RETRIES too (qodo #5);
    # before that fix the retry draws still came from OS entropy.
    traj = m.run(steps=_STEPS, max_retries=15, seed=0)
    # ``steps=N`` yields N + 1 rows: the initial condition, then N iterates —
    # exactly as a flow returns its ``ic`` at ``t0``.  Before v6 a map returned N
    # rows starting at f(ic), so ``t[0] = 0`` was labelling x_1, ``traj["x"][n]``
    # was x_{n+1}, and every cobweb began one iterate late.
    assert traj.t.shape == (_STEPS + 1,)
    assert traj.y.shape == (_STEPS + 1, m.dim)
    np.testing.assert_array_equal(traj.t, np.arange(_STEPS + 1))
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_map_explicit_ic_is_used_but_never_latched() -> None:
    """An explicit ``run(ic=)`` starts the orbit and leaves the map alone (v6).

    See ``test_ode_systems.py::test_ode_explicit_ic_is_used_but_never_latched``
    — the same rule, in the map's own horizon word.
    """
    import tsdynamics as ts

    h = ts.systems.Henon()
    ic = np.array([0.2, 0.3])
    traj = h.run(steps=50, ic=ic)
    # A map's grid starts at the INITIAL CONDITION, exactly as a flow's does, so
    # row 0 IS the ic and row 1 is f(ic).  It used to start at f(ic), which made
    # ``t[0] = 0`` label the first iterate and put every cobweb one step out.
    a, b = 1.4, 0.3
    np.testing.assert_array_almost_equal(traj.y[0], ic)
    np.testing.assert_array_almost_equal(traj.y[1], [1 - a * ic[0] ** 2 + ic[1], b * ic[0]])
    assert h.ic is None


# ---------------------------------------------------------------------------
# Lyapunov spectrum — shape/finiteness sweep
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_map_lyapunov_shape(map_entry) -> None:
    if map_entry.name in MAP_LYAPUNOV_EXCLUDE:
        pytest.skip(MAP_LYAPUNOV_EXCLUDE[map_entry.name])
    m = map_entry.cls()
    exps = ts.analysis.lyapunov_spectrum(m, n=300, k=m.dim)
    assert exps.shape == (m.dim,)
    assert np.all(np.isfinite(exps))


@pytest.mark.slow
def test_map_lyapunov_partial_spectrum(map_entry) -> None:
    if map_entry.name in MAP_LYAPUNOV_EXCLUDE:
        pytest.skip(MAP_LYAPUNOV_EXCLUDE[map_entry.name])
    m = map_entry.cls()
    exps = ts.analysis.lyapunov_spectrum(m, n=300, k=1)
    assert exps.shape == (1,)
    assert np.isfinite(exps[0])


def test_a_seeded_map_run_is_reproducible_across_divergence_retries() -> None:
    """``run(seed=...)`` fixes the retry initial conditions, not only the first one.

    ``_resolve_ic`` reaches the seeded generator only on the branch where it draws
    the *first* IC at random.  A map that starts from a declared ``_default_ic``
    (Hénon does) never took that branch, so the random-IC retries drew from a
    fresh OS-entropy generator and two ``run(seed=7)`` calls that diverged once
    traced different orbits — on exactly the runs the seed is there to pin (qodo
    #5).
    """
    from tsdynamics.errors import ConvergenceError

    def retry_ics(seed: int) -> list[np.ndarray]:
        system = ts.systems.Henon()
        seen: list[np.ndarray] = []

        def always_diverges(*, steps: int, ic: object, backend: str) -> None:
            seen.append(np.asarray(ic).copy())
            raise ConvergenceError("forced divergence")

        system._iterate_engine = always_diverges  # type: ignore[method-assign]
        with pytest.warns(RuntimeWarning, match="Retrying"), pytest.raises(ConvergenceError):
            system.run(20, seed=seed, max_retries=4)
        return seen

    first, second, other = retry_ics(7), retry_ics(7), retry_ics(8)
    assert len(first) == 4  # the default IC plus three retry draws
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert not np.array_equal(first[1], other[1])
