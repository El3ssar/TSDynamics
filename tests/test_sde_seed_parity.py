"""SDE seed parity + fixed-step landing (stream FIX-SDE-WIENER).

Two correctness guarantees this stream pins down:

1. **Tolerant fixed-step landing.** The output grid is uniform in ``dt``, but the
   grid is built with ``np.arange``, so a nominally ``dt``-wide segment computes a
   ``remaining = t_end - t`` that is ``dt`` plus a few ULP. The old
   ``dt >= remaining`` landing test then took a full ``dt`` step **and a spurious
   sub-ULP sliver step** — drawing an extra ``N(0, ~ULP)`` Wiener increment that
   desynced the noise stream and added a meaningless noise kick. The fix
   (:data:`tsdynamics.families.stochastic._LANDING_REL_TOL`, mirrored by Rust's
   ``LANDING_REL_TOL``) absorbs the roundoff into one canonical ``dt`` step, so
   the discretisation is exactly **one ``N(0, dt)`` increment per step** and
   ``integrate`` and the ``step()`` loop trace the *same* path.

2. **Honest oracle.** The pure-Python reference reproduces the compiled engine
   **to floating-point tolerance**, not bit-for-bit: the integer RNG stream and
   the draw order are identical, but the Box–Muller normal itself differs by ≤1
   ULP (Python libm ``sin``/``cos`` vs the engine's Rust ``sin_cos``). The
   genuinely bit-for-bit guarantees — ``interp == jit`` and the reference
   ``integrate == step``-loop — are asserted exactly; the reference-vs-engine
   agreement is asserted to a tight tolerance.

The engine-backed tests ``importorskip`` the compiled extension, so they skip
cleanly where ``tsdynamics._rust`` is not built (and are auto-tagged ``engine``).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics.families.stochastic as sde_mod
from tsdynamics import StochasticSystem
from tsdynamics.families.stochastic import _seed_for
from tsdynamics.utils.grids import make_output_grid

# A guarded import (not a module-level ``importorskip``): it auto-tags this module
# ``engine`` for the engine CI job (see ``tests/_engine_marker.py``) yet lets the
# pure-Python landing tests below run even where the extension is not built — only
# the engine-backed parity tests are gated on it.
try:
    import tsdynamics._rust  # noqa: F401

    _HAS_RUST = True
except ImportError:  # pragma: no cover - exercised only on wheel-free machines
    _HAS_RUST = False

requires_engine = pytest.mark.skipif(
    not _HAS_RUST, reason="compiled tsdynamics._rust extension is not built"
)

# A grid that *provokes* the bug: with dt = 0.02 over [0, 1], np.arange makes 34
# of the 50 interior segments compute a width a few ULP above dt, so the old
# landing logic took spurious sliver sub-steps on the segments where the leftover
# survived the round-trip add. tf is an exact multiple of dt (no genuine short
# final segment), so every segment is a clean one-dt step.
_ROUNDOFF_TF = 1.0
_ROUNDOFF_DT = 0.02


# ---------------------------------------------------------------------------
# Test-local SDE systems (autonomous, so a t pinned to the grid and a t
# accumulated step-by-step evaluate the drift/diffusion identically — the
# precondition for the bit-for-bit comparisons below). Unique names so they do
# not collide in the registry with the fixtures in test_sde.py / test_engine_wire.py.
# ---------------------------------------------------------------------------


class _ParityGBM(StochasticSystem):
    """dX = μX dt + σX dW — multiplicative noise (g' ≠ 0), exercises Milstein."""

    params = {"mu": 0.12, "sigma": 0.3}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _drift(y, t, mu, sigma):
        return [mu * y(0)]

    @staticmethod
    def _diffusion(y, t, mu, sigma):
        return [sigma * y(0)]


class _ParityOU(StochasticSystem):
    """dX = θ(μ − X) dt + σ dW — additive noise (g' = 0)."""

    params = {"theta": 1.0, "mu": 0.0, "sigma": 0.4}
    dim = 1
    variables = ("x",)

    @staticmethod
    def _drift(y, t, theta, mu, sigma):
        return [theta * (mu - y(0))]

    @staticmethod
    def _diffusion(y, t, theta, mu, sigma):
        return [sigma]


# ===========================================================================
# 1. The fixed-step landing fix — pure Python, no engine (the failing-first set)
# ===========================================================================


@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_reference_integrate_matches_step_loop_bit_for_bit(method):
    """``integrate`` over a uniform-``dt`` grid == stepping ``dt`` by hand.

    This is the headline regression for the spurious sliver: under the old
    landing each grid segment drew ``N(0, dt+ε)`` (and sometimes an extra
    ``N(0, ε)`` sliver), so ``integrate`` diverged from a clean ``step(dt)`` loop.
    With the tolerant landing both take exactly one ``N(0, dt)`` increment per
    segment from the same seeded stream, so the whole dense trajectory matches
    **bit-for-bit** (autonomous system ⇒ the pinned-vs-accumulated time is
    irrelevant).
    """
    sys = _ParityGBM()
    ic = [1.0]
    seed = 20240614

    ref = sys.integrate(
        final_time=_ROUNDOFF_TF,
        dt=_ROUNDOFF_DT,
        ic=ic,
        seed=seed,
        method=method,
        backend="reference",
    )
    grid = ref.t

    sys.reinit(ic, t=0.0, seed=seed, dt=_ROUNDOFF_DT, method=method)
    by_hand = [sys.state()]
    for _ in range(1, grid.size):
        by_hand.append(sys.step(_ROUNDOFF_DT))
    got = np.asarray(by_hand)

    assert got.shape == ref.y.shape
    np.testing.assert_array_equal(got, ref.y)


@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_no_spurious_wiener_substep_on_roundoff_grid(method, monkeypatch):
    """Exactly one diagonal Wiener increment per ``dt`` segment — no slivers.

    Counts every ``_wiener`` draw during a reference ``integrate`` on the
    roundoff-prone grid. The old code drew *more* than one increment per segment
    (a full ``dt`` step plus a tiny sliver) on the segments where float roundoff
    left a positive remainder; the fix draws exactly one ``N(0, dt)`` per segment.
    """
    grid = make_output_grid(0.0, _ROUNDOFF_TF, _ROUNDOFF_DT)
    n_segments = grid.size - 1

    drawn_h: list[float] = []
    real_wiener = sde_mod._wiener

    def counting_wiener(rng, h, dim):
        drawn_h.append(h)
        return real_wiener(rng, h, dim)

    monkeypatch.setattr(sde_mod, "_wiener", counting_wiener)

    _ParityGBM().integrate(
        final_time=_ROUNDOFF_TF,
        dt=_ROUNDOFF_DT,
        ic=[1.0],
        seed=1,
        method=method,
        backend="reference",
    )

    assert len(drawn_h) == n_segments, (
        f"expected one Wiener draw per segment ({n_segments}); "
        f"got {len(drawn_h)} (spurious sub-steps: {len(drawn_h) - n_segments})"
    )
    # Every increment is a canonical dt step (no roundoff sliver leaked through).
    assert all(h == _ROUNDOFF_DT for h in drawn_h), (
        "a landing step used a width != dt on a uniform grid: "
        f"{[h for h in drawn_h if h != _ROUNDOFF_DT]}"
    )


def test_tolerant_landing_preserves_a_genuine_short_final_step(monkeypatch):
    """A grid whose ``dt`` does *not* divide the span keeps its short final step.

    The tolerance must not swallow a *genuine* short landing step: with
    ``dt = 0.03`` over ``[0, 1]`` the appended final sample sits ``0.01`` after
    the last lattice point, so the last increment must be ``N(0, 0.01)`` — one
    draw, not absorbed into ``dt`` and not split into a ``dt`` step plus a sliver.
    """
    dt = 0.03
    tf = 1.0
    grid = make_output_grid(0.0, tf, dt)
    n_segments = grid.size - 1
    short = float(grid[-1] - grid[-2])
    assert short < dt  # the genuinely short final segment

    drawn_h: list[float] = []
    real_wiener = sde_mod._wiener
    monkeypatch.setattr(
        sde_mod,
        "_wiener",
        lambda rng, h, dim: (drawn_h.append(h), real_wiener(rng, h, dim))[1],
    )

    _ParityGBM().integrate(
        final_time=tf,
        dt=dt,
        ic=[1.0],
        seed=2,
        backend="reference",
    )

    assert len(drawn_h) == n_segments, "short final step must be a single increment"
    # The final increment matches the genuine remaining span (to float roundoff),
    # not dt; every earlier increment is a clean dt step.
    assert drawn_h[-1] == pytest.approx(short, abs=1e-15)
    assert all(h == dt for h in drawn_h[:-1])


@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_reference_ensemble_row_equals_single_trajectory_with_index_seed(method):
    """Each ensemble row == a lone ``integrate`` seeded by ``seed_for(seed, i)``.

    The parallel-equals-serial contract at the Python level: trajectory ``i`` in a
    seeded batch draws exactly the stream of a single trajectory whose seed is
    ``seed_for(base_seed, i)`` — bit-for-bit, since both walk the same clean
    ``dt`` steps over the same span.
    """
    sys = _ParityGBM()
    rng = np.random.default_rng(7)
    ics = 1.0 + 0.1 * rng.standard_normal((6, 1))
    base_seed = 4242
    tf, dt = 0.5, 0.02

    batch = sys.ensemble(
        ics,
        final_time=tf,
        dt=dt,
        method=method,
        seed=base_seed,
        backend="reference",
    )

    for i, ic in enumerate(ics):
        lone = sys.integrate(
            final_time=tf,
            dt=dt,
            ic=ic,
            seed=_seed_for(base_seed, i),
            method=method,
            backend="reference",
        )
        np.testing.assert_array_equal(batch[i], lone.y[-1])


def test_reference_same_seed_is_reproducible_and_index_decorrelates():
    """Same base seed ⇒ identical batch; distinct indices ⇒ distinct rows."""
    sys = _ParityOU()
    ics = np.full((8, 1), 0.5)
    a = sys.ensemble(ics, final_time=1.0, dt=0.01, seed=123, backend="reference")
    b = sys.ensemble(ics, final_time=1.0, dt=0.01, seed=123, backend="reference")
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[1]), "distinct indices gave identical draws"


# ===========================================================================
# 2. Engine parity — interp == jit (bit-for-bit) and reference ≈ engine
#    (tolerance), on the *roundoff-provoking* grid so the landing fix is the
#    thing under test. Skipped where the compiled extension is absent.
# ===========================================================================


@requires_engine
@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_engine_interp_equals_jit_bit_for_bit_on_roundoff_grid(method):
    """The interpreter and the Cranelift JIT land identically (bit-for-bit).

    Same seed, same tolerant landing, bit-identical evaluators ⇒ the dense
    trajectories agree exactly, including on the roundoff-prone grid.
    """
    sys = _ParityGBM()
    kw = dict(final_time=_ROUNDOFF_TF, dt=_ROUNDOFF_DT, ic=[1.0], seed=7, method=method)
    interp = sys.integrate(backend="interp", **kw)
    jit = sys.integrate(backend="jit", **kw)
    np.testing.assert_array_equal(interp.y, jit.y)


@requires_engine
@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_engine_matches_reference_to_tolerance_on_roundoff_grid(method):
    """Reference ≈ engine to a tight float tolerance (the downgraded oracle).

    Both paths share the SplitMix64 stream and the *same* tolerant landing, so
    they draw the same number of increments in the same order on the roundoff
    grid (a desync would show up as an O(1) divergence, not a ULP one). The only
    residual difference is the Box–Muller normal (libm ``sin``/``cos`` vs Rust
    ``sin_cos``, ≤1 ULP/draw), so a short window matches very tightly — but is
    *not* asserted bit-for-bit, because that ULP is platform/libm dependent.
    """
    sys = _ParityGBM()
    kw = dict(final_time=_ROUNDOFF_TF, dt=_ROUNDOFF_DT, ic=[1.0], seed=20240614, method=method)
    ref = sys.integrate(backend="reference", **kw)
    eng = sys.integrate(backend="interp", **kw)
    assert eng.meta["engine"] == "rust"
    np.testing.assert_allclose(eng.y, ref.y, rtol=1e-9, atol=1e-11)


@requires_engine
@pytest.mark.parametrize("method", ["euler_maruyama", "milstein"])
def test_engine_ensemble_interp_equals_jit_bit_for_bit(method):
    """Seeded SDE ensemble: interp == jit bit-for-bit (parallel == serial)."""
    sys = _ParityGBM()
    ics = np.linspace(0.8, 1.2, 8).reshape(-1, 1)
    kw = dict(final_time=0.5, dt=_ROUNDOFF_DT, method=method, seed=3)
    interp = sys.ensemble(ics, backend="interp", **kw)
    jit = sys.ensemble(ics, backend="jit", **kw)
    np.testing.assert_array_equal(interp, jit)
