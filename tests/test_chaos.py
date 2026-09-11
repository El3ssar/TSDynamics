r"""
Tests for the chaos indicators (stream **A-CHAOS**).

Each indicator is pinned to a literature value where one is exact or
well-established:

- **GALI** (Skokos et al. 2007/2008): the GALI\ :sub:`k` of a chaotic orbit
  decays like ``exp(-[(λ₁-λ₂)+…+(λ₁-λ_k)] t)``.  For the Lorenz flow the GALI₂
  decay rate must reproduce ``λ₁-λ₂ ≈ 0.906`` (Lorenz max exponent, λ₂=0); for a
  harmonic oscillator (regular) GALI₂ stays at 1.
- **0–1 test** (Gottwald & Melbourne 2004/2009): the logistic map gives
  ``K ≈ 1`` at ``r=4`` (chaos) and ``K ≈ 0`` at ``r=3.5`` (a period-4 cycle).
- **Expansion entropy** (Hunt & Ott 2015): the unit-height tent map has
  ``|f'| ≡ 2`` so ``H = ln 2`` exactly; the Hénon map reproduces its topological
  entropy ``≈ 0.465`` (Newhouse–Pignataro); a quasi-periodic circle map gives
  ``H ≈ 0``.

The map indicators run on the pure-Python ``_step``/``_jacobian``; the flow
indicators use a self-contained RK4 variational integrator over the
SymEngine-lambdified RHS and Jacobian, so none of these tests need the Rust engine.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import ContinuousSystem, registry
from tsdynamics.analysis import (
    ExpansionEntropyResult,
    GALIResult,
    expansion_entropy,
    gali,
    zero_one_test,
)
from tsdynamics.data import Box
from tsdynamics.errors import InvalidInputError, InvalidParameterError

LN2 = float(np.log(2.0))
LORENZ_GAP = 0.9056  # λ₁ - λ₂ for the classic Lorenz attractor (λ₂ = 0)
HENON_HTOP = 0.4651  # topological entropy of the Hénon map (Newhouse–Pignataro)


class _Harmonic(ContinuousSystem):
    """Undamped harmonic oscillator ``x' = v, v' = -w² x`` — a regular (λ=0) flow."""

    params = {"w": 1.0}
    dim = 2
    variables = ("x", "v")

    @staticmethod
    def _equations(y, t, w):
        return [y(1), -w * w * y(0)]


# ── GALI ─────────────────────────────────────────────────────────────────────


def test_gali_henon_chaotic_collapses():
    """Hénon GALI₂ collapses exponentially; the rate is near λ₁-λ₂ ≈ 2.04."""
    g = gali(ts.systems.Henon(), k=2, ic=[0.1, 0.1], n=70, seed=0)
    assert isinstance(g, GALIResult)
    assert g.is_chaotic()
    assert g.final < 1e-8
    # λ₁-λ₂ = 0.419 - (-1.623) ≈ 2.04 (Sprott 2003); the finite-time estimate is
    # a touch steeper from the initial alignment transient.
    assert 1.6 < g.decay_rate() < 2.7


def test_gali_lorenz_decay_rate_matches_lyapunov_gap():
    """Lorenz GALI₂ decay rate reproduces λ₁-λ₂ ≈ 0.906 (Skokos law)."""
    g = gali(
        ts.systems.Lorenz(),
        k=2,
        ic=[1.0, 1.0, 1.0],
        final_time=22.0,
        dt=0.05,
        transient=15.0,
        seed=0,
    )
    rate = g.decay_rate(floor=1e-10, t_min=5.0)
    assert rate == pytest.approx(LORENZ_GAP, abs=0.15)
    assert g.is_chaotic()


def test_gali_lorenz_k3_collapses_faster():
    """GALI₃ adds the (large) λ₁-λ₃ gap, so it dies almost immediately."""
    g = gali(
        ts.systems.Lorenz(),
        k=3,
        ic=[1.0, 1.0, 1.0],
        final_time=6.0,
        dt=0.02,
        transient=15.0,
        seed=0,
    )
    assert g.final < 1e-10


def test_gali_regular_orbit_stays_unity():
    """A harmonic oscillator is regular: GALI₂ neither aligns nor decays."""
    g = gali(_Harmonic(), k=2, ic=[1.0, 0.0], final_time=40.0, dt=0.1, transient=0.0, seed=0)
    assert g.values.min() > 0.9
    assert float(g) == pytest.approx(1.0, abs=0.05)
    assert not g.is_chaotic()


def test_gali_input_validation():
    with pytest.raises(ValueError, match="k must satisfy"):
        gali(ts.systems.Henon(), k=1)
    with pytest.raises(ValueError, match="k must satisfy"):
        gali(ts.systems.Henon(), k=3)  # dim is 2
    with pytest.raises(ValueError, match="dt has no meaning"):
        gali(ts.systems.Henon(), k=2, dt=0.1)
    with pytest.raises(ValueError, match="n applies to maps"):
        gali(ts.systems.Lorenz(), k=2, n=100)
    with pytest.raises(NotImplementedError):
        gali(ts.systems.MackeyGlass(), k=2)  # DDE: no finite tangent space here


# ── 0–1 test ─────────────────────────────────────────────────────────────────


def test_zero_one_logistic_chaotic():
    """Logistic r=4 is fully chaotic → K ≈ 1."""
    x = ts.systems.Logistic(params={"r": 4.0}).run(steps=4000, ic=[0.2]).component("x")
    assert zero_one_test(x, n_c=50, seed=1) > 0.9


def test_zero_one_logistic_regular():
    """Logistic r=3.5 settles on a period-4 cycle → K ≈ 0."""
    x = ts.systems.Logistic(params={"r": 3.5}).run(steps=4000, ic=[0.2]).component("x")
    assert zero_one_test(x, n_c=50, seed=1) < 0.1


def test_zero_one_quasiperiodic_is_regular():
    """A quasi-periodic (two-tone) signal is non-chaotic → K ≈ 0."""
    j = np.arange(3000, dtype=float)
    x = np.sin(0.4 * j) + np.sin(np.sqrt(2.0) * 0.4 * j)
    assert zero_one_test(x, n_c=50, seed=2) < 0.1


def test_zero_one_distribution_and_errors():
    x = ts.systems.Logistic(params={"r": 4.0}).run(steps=2500, ic=[0.3]).component("x")
    k, k_c = zero_one_test(x, n_c=20, seed=0, return_distribution=True)
    assert k_c.shape == (20,)
    assert k == pytest.approx(float(np.median(k_c)))
    # The system overload integrates/iterates internally (like gali): a chaotic
    # map handed straight to the test scores K ≈ 1.
    assert zero_one_test(ts.systems.Logistic(params={"r": 4.0}), n=2500, ic=[0.3]) > 0.9
    with pytest.raises(ValueError, match="long series"):
        zero_one_test(np.zeros(50))
    # v6 bug fix: a multi-component system NO LONGER raises.  The test reads ONE
    # observable and ``components`` defaults to 0, so the flagship call for the
    # flagship system works; before v6 it raised for every system of dim > 1.
    assert float(zero_one_test(ts.systems.Lorenz(), dt=0.5, final_time=3000.0)) > 0.9
    # ...and a component that does not exist still says which ones do.
    with pytest.raises(ValueError, match="no component named"):
        zero_one_test(ts.systems.Lorenz(), components="w", dt=0.5, final_time=400.0)


# ── expansion entropy ────────────────────────────────────────────────────────


class TestZeroOneWorksForEveryDimension:
    """The v6 bug fix: ``zero_one_test`` is registered ``needs="system"`` and
    headlined in its own module docstring, and before v6 it **raised for every
    system of dim > 1** — the flagship call for the flagship system."""

    def test_the_flagship_call_for_the_flagship_system(self) -> None:
        assert float(zero_one_test(ts.systems.Lorenz(), dt=0.5, final_time=3000.0)) > 0.9

    def test_a_two_component_map_too(self) -> None:
        assert float(zero_one_test(ts.systems.Henon(), n=6000)) > 0.9

    def test_components_picks_the_observable_by_index_or_name(self) -> None:
        lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
        by_index = float(zero_one_test(lor, components=2, dt=0.5, final_time=1500.0))
        by_name = float(zero_one_test(lor, components="z", dt=0.5, final_time=1500.0))
        assert by_index == pytest.approx(by_name)

    def test_a_name_that_does_not_exist_lists_the_ones_that_do(self) -> None:
        with pytest.raises(ValueError, match=r"components are: x, y, z"):
            zero_one_test(ts.systems.Lorenz(), components="w", dt=0.5, final_time=400.0)


def test_expansion_entropy_tent_is_ln2():
    """Unit-height tent map: |f'| ≡ 2 ⇒ E(t)=2ᵗ ⇒ H = ln 2 (exact)."""
    h = expansion_entropy(
        ts.systems.Tent(params={"mu": 1.0}), Box([0.0], [1.0]), n_samples=200, n=18
    )
    assert isinstance(h, ExpansionEntropyResult)
    assert float(h) == pytest.approx(LN2, abs=0.02)
    assert h.n_survivors == h.n_samples  # nothing leaves [0, 1]


def test_expansion_entropy_henon_topological():
    """Hénon expansion entropy reproduces its topological entropy ≈ 0.465."""
    box = Box([-1.6, -0.5], [1.6, 0.5])
    h = expansion_entropy(ts.systems.Henon(), box, n_samples=400, n=12, seed=0)
    assert float(h) == pytest.approx(HENON_HTOP, abs=0.1)


def test_expansion_entropy_circle_quasiperiodic_is_zero():
    """A sub-critical circle map is quasi-periodic (λ=0) → H ≈ 0."""
    h = expansion_entropy(
        ts.systems.Circle(params={"omega": 0.3333, "k": 0.5}),
        Box([0.0], [1.0]),
        n_samples=200,
        n=18,
    )
    assert abs(float(h)) < 0.05


def test_expansion_entropy_lorenz_flow_positive():
    """The Lorenz flow expands on its attractor → H clearly positive."""
    box = Box([-20.0, -25.0, 0.0], [20.0, 25.0, 50.0])
    h = expansion_entropy(ts.systems.Lorenz(), box, n_samples=80, final_time=2.5, dt=0.25, seed=0)
    assert float(h) > 0.5


def test_expansion_entropy_input_validation():
    with pytest.raises(ValueError, match="dt has no meaning"):
        expansion_entropy(ts.systems.Henon(), Box([-2, -2], [2, 2]), dt=0.1)
    with pytest.raises(ValueError, match="region dimension"):
        expansion_entropy(ts.systems.Henon(), Box([0.0], [1.0]))  # 1-D box, 2-D map
    with pytest.raises(NotImplementedError):
        expansion_entropy(ts.systems.MackeyGlass(), Box([0.0], [1.0]))


# ── result objects & registry ────────────────────────────────────────────────


def test_result_repr_and_float():
    g = gali(ts.systems.Henon(), k=2, ic=[0.1, 0.1], n=30, seed=0)
    assert "GALIResult" in repr(g)
    assert float(g) == g.final
    h = expansion_entropy(
        ts.systems.Tent(params={"mu": 1.0}), Box([0.0], [1.0]), n_samples=50, n=10
    )
    assert "ExpansionEntropyResult" in repr(h)
    assert float(h) == h.entropy


def test_indicators_self_register():
    for name in ("gali", "zero_one_test", "expansion_entropy"):
        assert name in registry.analyses


# ── robustness: degenerate / diverging frames must not crash (regression) ─────


def test_gali_random_ic_henon_never_crashes():
    """Random-IC GALI on the Hénon map must never raise (regression).

    ``Henon`` declares no ``default_ic``, so every call rolls a random IC; many
    land outside the attractor's basin and escape to infinity, which used to make
    the deviation frame non-finite and crash ``np.linalg.svd`` with
    ``LinAlgError`` (run-to-run non-deterministically, and at every long horizon).
    ``gali`` must retry onto the attractor and return a chaotic result every time.
    """
    for _ in range(20):
        g = gali(ts.systems.Henon(), k=2, n=1500)
        assert isinstance(g, GALIResult)
        assert np.all(np.isfinite(g.values))
        assert g.is_chaotic()  # Hénon is chaotic → GALI₂ collapses to ~0


def test_gali_offbasin_explicit_ic_raises():
    """An explicit IC that escapes the basin must raise, not be silently re-rolled.

    GALI characterises a *specific* orbit, so a pinned ``ic`` that diverges is a
    user error to surface (``InvalidInputError``) — never a cue to substitute a
    different (random) orbit and hand back a result for an orbit the caller never
    asked about (the FIX-GALI-IC contract).  The off-basin re-roll survives only
    for the ``ic=None`` default-draw case (covered separately).
    """
    with pytest.raises(InvalidInputError):
        gali(ts.systems.Henon(), k=2, ic=[10.0, 10.0], n=80, seed=0)


def test_gali_volume_degenerate_returns_zero():
    """A collapsed or non-finite deviation frame spans zero volume, never raises."""
    from tsdynamics.analysis.chaos._common import gali_volume

    # two perfectly aligned unit columns → zero parallelepiped volume
    assert gali_volume(np.array([[1.0, 1.0], [0.0, 0.0]])) == pytest.approx(0.0)
    # a non-finite frame (diverged orbit) is treated as collapsed, not a crash
    assert gali_volume(np.array([[np.inf, 0.0], [np.nan, 1.0]])) == 0.0


def test_expansion_volume_overflow_returns_inf():
    """A non-finite fundamental matrix (overflowed growth) reports +inf, never raises."""
    from tsdynamics.analysis.chaos._common import expansion_volume

    assert expansion_volume(np.array([[np.inf, 0.0], [0.0, 1.0]])) == np.inf
    assert expansion_volume(np.array([[np.nan, 0.0], [0.0, 1.0]])) == np.inf
    assert expansion_volume(np.eye(2)) == pytest.approx(1.0)  # non-expanding


def test_expansion_entropy_long_horizon_no_crash():
    """A long un-renormalised horizon overflows the tangent product but must not crash."""
    box = Box([-1.6, -0.5], [1.6, 0.5])
    h = expansion_entropy(ts.systems.Henon(), box, n_samples=60, n=300, seed=0)
    assert isinstance(h, ExpansionEntropyResult)
    assert np.isfinite(float(h))


# ── robustness: degenerate step counts raise a clean typed error (regression) ──


def test_gali_degenerate_step_count_raises_clean_error():
    """A degenerate step count must raise a clean typed error, not an opaque IndexError.

    ``gali(..., n=0)`` used to build an empty GALI series; the very first read of
    the result (``float`` / ``.final`` → ``values[-1]``) then raised a bare
    ``IndexError`` deep in the accessor.  It must instead reject the degenerate
    horizon up front with an ``InvalidParameterError`` (a ``ValueError``).
    """
    for bad in (0, -3):
        with pytest.raises(InvalidParameterError, match="must be >= 1"):
            gali(ts.systems.Henon(), k=2, ic=[0.1, 0.1], n=bad)
    # Flow horizon: a non-positive final_time / dt spans no step.
    with pytest.raises(InvalidParameterError, match="dt must be positive"):
        gali(ts.systems.Lorenz(), k=2, ic=[1.0, 1.0, 1.0], dt=0.0)
    with pytest.raises(InvalidParameterError, match="final_time must be positive"):
        gali(ts.systems.Lorenz(), k=2, ic=[1.0, 1.0, 1.0], final_time=0.0)


def test_gali_empty_result_final_raises_clean_error():
    """Reading ``.final`` on an empty GALI series raises a clean typed error."""
    g = GALIResult(k=2, times=np.empty(0), values=np.empty(0))
    with pytest.raises(InvalidParameterError, match="no samples"):
        _ = g.final
    with pytest.raises(InvalidParameterError, match="no samples"):
        float(g)


def test_expansion_entropy_degenerate_inputs_raise_clean_error():
    """Degenerate sample / step counts raise a clean typed error, never a numpy crash."""
    box = Box([0.0], [1.0])
    with pytest.raises(InvalidParameterError, match="n_samples must be >= 1"):
        expansion_entropy(ts.systems.Tent(params={"mu": 1.0}), box, n_samples=0)
    with pytest.raises(InvalidParameterError, match="must be >= 1"):
        expansion_entropy(ts.systems.Tent(params={"mu": 1.0}), box, n_samples=10, n=0)
    lorenz_box = Box([-20.0, -25.0, 0.0], [20.0, 25.0, 50.0])
    with pytest.raises(InvalidParameterError, match="final_time must be positive"):
        expansion_entropy(ts.systems.Lorenz(), lorenz_box, n_samples=5, final_time=0.0)


def test_zero_one_short_series_raises_typed_error():
    """A too-short / empty observable raises the typed InvalidParameterError (a ValueError)."""
    with pytest.raises(InvalidParameterError, match="long series"):
        zero_one_test(np.zeros(50))
    with pytest.raises(InvalidParameterError, match="long series"):
        zero_one_test(np.empty(0))


# ---------------------------------------------------------------------------
# 0–1 test: the oversampling guard
#
# The Gottwald–Melbourne test needs an observable sampled about once per
# oscillation.  Handed an oversampled flow it does not degrade gracefully — it
# reports a plainly chaotic orbit as REGULAR: Lorenz x(t) at dt = 0.02 gave
# K = -0.02.  The library walked straight into that with no guard at all.
# ---------------------------------------------------------------------------


def _lorenz_x(dt: float, final_time: float = 650.0) -> np.ndarray:
    lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    return lor.run(final_time=final_time, dt=dt, ic=[1.0, 1.0, 1.0]).after(50.0).y[:, 0]


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.05, 0.1])
def test_zero_one_chaotic_flow_is_chaotic_at_every_sampling_rate(dt: float):
    """A chaotic Lorenz must give K ≈ 1 however finely it was sampled.

    Regression: at dt = 0.02 this returned K = -0.02 ("regular").
    """
    result = zero_one_test(_lorenz_x(dt, final_time=350.0))
    assert float(result) > 0.9, (float(result), result.meta)
    assert result.meta["samples_per_oscillation"] > 10.0
    assert result.meta["stride"] > 1, "an oversampled flow must have been decimated"


def test_zero_one_oversampling_ignore_reproduces_the_defect():
    """``oversampling='ignore'`` pins the failure the guard exists to prevent."""
    result = zero_one_test(_lorenz_x(0.02, final_time=130.0)[:4000], oversampling="ignore")
    assert float(result) < 0.5, float(result)


def test_zero_one_oversampling_warn_says_so_and_does_not_touch_the_data():
    from tsdynamics.analysis.chaos.zero_one import OversamplingWarning

    x = _lorenz_x(0.02, final_time=130.0)[:4000]
    with pytest.warns(OversamplingWarning, match="samples per oscillation"):
        result = zero_one_test(x, oversampling="warn")
    assert result.meta["stride"] == 1
    assert result.meta["n_samples"] == 4000


def test_zero_one_warns_when_the_record_is_too_short_to_decimate_enough():
    """Decimation is capped at a floor of kept samples; when that is not enough, say so."""
    from tsdynamics.analysis.chaos.zero_one import OversamplingWarning

    with pytest.warns(OversamplingWarning, match="too short to decimate"):
        zero_one_test(_lorenz_x(0.02, final_time=80.0)[:1200])


@pytest.mark.parametrize("dt", [0.02, 0.2])
def test_zero_one_periodic_flow_stays_regular(dt: float):
    """The guard must not manufacture chaos: a limit cycle keeps K ≈ 0."""
    periodic = ts.systems.Rossler(params={"a": 0.1, "b": 0.1, "c": 6.0}, ic=[1.0, 1.0, 1.0])
    result = zero_one_test(periodic, components=0, dt=dt, final_time=2000.0, transient=400.0)
    assert abs(float(result)) < 0.1, float(result)


def test_zero_one_oversampled_quasiperiodic_stays_regular():
    """An *oversampled* 2-torus is regular before and after the guard decimates it."""
    t = np.arange(200_000) * 0.02
    result = zero_one_test(np.sin(t) + np.sin(np.sqrt(2.0) * t))
    assert result.meta["stride"] > 1
    assert abs(float(result)) < 0.1, float(result)


@pytest.mark.parametrize(
    ("system", "expected_chaotic"),
    [
        (ts.systems.Logistic(params={"r": 4.0}, ic=[0.1]), True),
        (ts.systems.Logistic(params={"r": 3.5}, ic=[0.1]), False),
    ],
)
def test_zero_one_maps_are_left_alone_by_the_guard(system, expected_chaotic):
    """A map is already sampled once per iteration — stride must stay 1."""
    result = zero_one_test(system, n=5000, transient=1000)
    assert result.meta["stride"] == 1
    assert (float(result) > 0.9) is expected_chaotic


def test_zero_one_rejects_an_unknown_oversampling_policy():
    with pytest.raises(InvalidParameterError, match="oversampling"):
        zero_one_test(np.zeros(500), oversampling="maybe")
