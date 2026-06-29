"""Pin the engine's *parallel == serial* ensemble contract for ODEs and maps.

The Rust engine fans a batch of initial conditions out over a rayon thread pool
and returns one final-state row per IC (:func:`tsdynamics.engine.run.ensemble`,
``(n, dim)``).  The engine's promise — stated in ``CLAUDE.md`` and in the kernel
docstrings — is that this batched run is *deterministic and per-row*: a batch
row is the same trajectory the single-IC :meth:`integrate` / :meth:`iterate`
path produces, the result does not depend on a row's position in the batch, and
a single diverging trajectory becomes a ``NaN`` row without poisoning its
neighbours.

The SDE ensemble's seeded parallel==serial contract is already pinned in
``tests/test_engine_wire.py`` / ``tests/test_sde_seed_parity.py``; the *ODE*
(and deterministic-*map*) batch was found UNPINNED by the backend audit.  This
file pins it.

Why the specific tolerances / methods below:

* **Maps** lower the ``_step`` to the same IR the serial loop runs, with no
  floating-point reordering, so an ensemble row is **bit-for-bit** equal to the
  serial :meth:`iterate` last state (the Hénon map is ``+``/``-``/``*`` only —
  CLAUDE.md, WS-MAPITER).  We assert exact equality.
* **ODEs** integrated with the **fixed-step ``rk4``** kernel agree with the
  serial dense path to a few ULP (~1e-13 observed): the only difference between
  the two float-distinct computations is the dense path's output-grid landing
  vs the ensemble path's ``first_step`` seeding — the *math* is the same step
  sequence.  We assert ``atol=1e-9`` (comfortably above the ~1e-13 we measure)
  and document that it is not bit-for-bit.  We deliberately avoid an *adaptive*
  kernel over a *chaotic* horizon, where these two distinct float computations
  diverge by the Lyapunov-amplified roundoff (~1e-8 over 4 Lorenz time units) —
  that would be a calibration trap, not a contract violation.
* The **reference** backend loops the *same* pure-Python integrator the serial
  ``integrate(backend="reference")`` uses, so there its rows are **bit-for-bit**
  equal to the serial final states.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.engine import run

pytestmark = pytest.mark.engine

# A short, fixed-step horizon: long enough to leave the IC behind, short enough
# that fixed-step rk4 roundoff has not Lyapunov-amplified into a calibration trap.
_ODE_KW = dict(final_time=2.0, dt=0.005, method="rk4", rtol=1e-9, atol=1e-12)
# Far above the ~1e-13 we measure between the dense and ensemble float paths,
# far below any physically meaningful state difference.
_ODE_ATOL = 1e-9


def _lorenz_ics() -> np.ndarray:
    return np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 3.0], [0.5, 0.5, 0.5]])


def _rossler_ics() -> np.ndarray:
    return np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 0.0], [-1.0, 0.5, 1.5]])


# ---------------------------------------------------------------------------
# 1. Row i == single integrate of IC i  (ODE)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, ics",
    [
        (ts.Lorenz, _lorenz_ics()),
        (ts.Rossler, _rossler_ics()),
    ],
)
def test_ode_ensemble_row_equals_single_integrate(factory, ics):
    """Each engine ensemble row reproduces the serial single-IC integration.

    Fixed-step rk4 over a short horizon: the two float-distinct computations
    (dense vs batched) agree to ~1e-13 (asserted at 1e-9), not bit-for-bit.
    """
    sys = factory()
    ens = run.ensemble(sys, ics, backend="interp", **_ODE_KW)
    assert ens.shape == (len(ics), sys.dim)
    assert np.all(np.isfinite(ens))

    for i, ic in enumerate(ics):
        # Fresh instance per IC so no stepping state leaks between rows.
        last = factory().integrate(ic=ic, **_ODE_KW).y[-1]
        np.testing.assert_allclose(
            ens[i],
            last,
            atol=_ODE_ATOL,
            rtol=0.0,
            err_msg=f"ensemble row {i} != single integrate of {ic}",
        )


def test_reference_ensemble_row_is_bit_for_bit_single_integrate():
    """The reference backend loops the *same* SciPy integrator the serial path
    uses, so its rows are bit-for-bit identical to the serial final states."""
    ics = _lorenz_ics()
    ens = run.ensemble(ts.Lorenz(), ics, backend="reference", **_ODE_KW)
    for i, ic in enumerate(ics):
        last = ts.Lorenz().integrate(ic=ic, backend="reference", **_ODE_KW).y[-1]
        np.testing.assert_array_equal(
            ens[i], last, err_msg=f"reference ensemble row {i} not bit-identical"
        )


# ---------------------------------------------------------------------------
# 2. Batch-order independence (ODE)
# ---------------------------------------------------------------------------


def test_ode_ensemble_is_batch_order_independent():
    """Permuting the input ICs permutes the output rows *identically* (bit-for-bit):
    a row's content is a function of its IC, never its position in the batch."""
    ics = _rossler_ics()
    perm = [2, 0, 1]
    base = run.ensemble(ts.Rossler(), ics, backend="interp", **_ODE_KW)
    permuted = run.ensemble(ts.Rossler(), ics[perm], backend="interp", **_ODE_KW)
    for new_pos, old_pos in enumerate(perm):
        # Engine is deterministic and per-row → the SAME float result regardless
        # of where the IC sat → bit-for-bit, not merely close.
        np.testing.assert_array_equal(
            permuted[new_pos],
            base[old_pos],
            err_msg=f"row for IC {old_pos} changed when moved to position {new_pos}",
        )


# ---------------------------------------------------------------------------
# 3. A diverged trajectory becomes a NaN row, without poisoning the others (ODE)
# ---------------------------------------------------------------------------


def test_ode_ensemble_diverged_row_is_nan_and_isolated():
    """One blow-up IC yields an all-NaN row; the finite rows are unaffected and
    equal what they would be integrated alone."""
    finite0 = np.array([1.0, 1.0, 1.0])
    finite2 = np.array([0.5, 0.5, 0.5])
    blowup = np.array([1e9, 1e9, 1e9])  # overflows fixed-step rk4 immediately
    ics = np.stack([finite0, blowup, finite2])

    ens = run.ensemble(ts.Lorenz(), ics, backend="interp", **_ODE_KW)

    # The diverged row is all-NaN; neighbours stay finite (no poisoning).
    assert np.all(np.isnan(ens[1])), "diverged trajectory should be a NaN row"
    assert np.all(np.isfinite(ens[0])), "row before the diverged one was poisoned"
    assert np.all(np.isfinite(ens[2])), "row after the diverged one was poisoned"

    # And the surviving rows match their stand-alone integration exactly.
    for pos, ic in ((0, finite0), (2, finite2)):
        last = ts.Lorenz().integrate(ic=ic, **_ODE_KW).y[-1]
        np.testing.assert_allclose(ens[pos], last, atol=_ODE_ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# 4. The same contract for a discrete map (Hénon) via the engine map-ensemble.
#    Maps lower with no float reordering → bit-for-bit equality throughout.
# ---------------------------------------------------------------------------

# ``run.ensemble`` of a MapProblem runs ``steps = round(final_time)`` iterations,
# matching ``iterate(steps=...)``.
_MAP_STEPS = 200


def _henon_ics() -> np.ndarray:
    return np.array([[0.1, 0.1], [0.2, -0.1], [-0.3, 0.05]])


def test_map_ensemble_row_equals_serial_iterate_bit_for_bit():
    """Each Hénon ensemble row is bit-for-bit the serial ``iterate`` final state
    (the map lowers to the same IR with no floating-point reordering)."""
    ics = _henon_ics()
    ens = run.ensemble(ts.Henon(), ics, final_time=float(_MAP_STEPS), backend="interp")
    assert ens.shape == (len(ics), 2)
    for i, ic in enumerate(ics):
        last = ts.Henon().iterate(ic=ic, steps=_MAP_STEPS, backend="interp").y[-1]
        np.testing.assert_array_equal(
            ens[i], last, err_msg=f"map ensemble row {i} not bit-identical to iterate"
        )


def test_map_ensemble_is_batch_order_independent_bit_for_bit():
    """Permuting the map ICs permutes the rows identically, bit-for-bit."""
    ics = _henon_ics()
    perm = [2, 0, 1]
    base = run.ensemble(ts.Henon(), ics, final_time=float(_MAP_STEPS), backend="interp")
    permuted = run.ensemble(ts.Henon(), ics[perm], final_time=float(_MAP_STEPS), backend="interp")
    for new_pos, old_pos in enumerate(perm):
        np.testing.assert_array_equal(permuted[new_pos], base[old_pos])


def test_map_ensemble_diverged_row_is_nan_and_isolated():
    """A divergent map IC becomes a NaN row; the finite rows are untouched and
    bit-for-bit equal to their serial iteration."""
    finite0 = np.array([0.1, 0.1])
    finite2 = np.array([-0.3, 0.05])
    blowup = np.array([1e6, 1e6])  # Hénon's quadratic term overflows to inf/NaN
    ics = np.stack([finite0, blowup, finite2])

    ens = run.ensemble(ts.Henon(), ics, final_time=float(_MAP_STEPS), backend="interp")

    assert np.all(np.isnan(ens[1])), "diverged map trajectory should be a NaN row"
    assert np.all(np.isfinite(ens[0])), "row before the diverged one was poisoned"
    assert np.all(np.isfinite(ens[2])), "row after the diverged one was poisoned"

    for pos, ic in ((0, finite0), (2, finite2)):
        last = ts.Henon().iterate(ic=ic, steps=_MAP_STEPS, backend="interp").y[-1]
        np.testing.assert_array_equal(ens[pos], last)
