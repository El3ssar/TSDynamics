"""``lyapunov_from_data`` — maximal Lyapunov exponent from a time series.

Literature targets: Hénon map λ_max ≈ 0.419 and Lorenz λ_max ≈ 0.906, both
recovered from a single scalar coordinate via delay embedding (Kantz 1994;
Rosenstein, Collins & De Luca 1993).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis.lyapunov import (
    LyapunovFromData,
    ScalingRegionWarning,
    lyapunov_from_data,
)
from tsdynamics.analysis.lyapunov.from_data import _delay_embed
from tsdynamics.errors import ConvergenceError, InvalidParameterError


@pytest.fixture(scope="module")
def henon_x() -> np.ndarray:
    """A long, transient-free Hénon ``x`` series."""
    return ts.systems.Henon().run(6000, transient=500, ic=[0.1, 0.1]).y[:, 0]


# ---------------------------------------------------------------------------
# Delay embedding helper
# ---------------------------------------------------------------------------


class TestDelayEmbed:
    def test_scalar_shape_and_values(self) -> None:
        x = np.arange(10.0)
        emb = _delay_embed(x, m=3, tau=2)
        assert emb.shape == (10 - 2 * 2, 3)  # rows = n - (m-1)*tau
        # row n is [x[n], x[n+tau], x[n+2 tau]]
        np.testing.assert_array_equal(emb[0], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(emb[1], [1.0, 3.0, 5.0])

    def test_multivariate_interleaves_channels(self) -> None:
        x = np.column_stack([np.arange(6.0), 10 + np.arange(6.0)])
        emb = _delay_embed(x, m=2, tau=1)
        assert emb.shape == (5, 4)  # (m * n_channels) columns
        np.testing.assert_array_equal(emb[0], [0.0, 10.0, 1.0, 11.0])

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            _delay_embed(np.arange(5.0), m=4, tau=3)


# ---------------------------------------------------------------------------
# Hénon map — fast tier
# ---------------------------------------------------------------------------


class TestHenon:
    @pytest.mark.parametrize("m", [2, 4])
    def test_kantz_recovers_mlle(self, henon_x: np.ndarray, m: int) -> None:
        res = lyapunov_from_data(
            henon_x, dimension=m, delay=1, k_max=12, method="kantz", fit=(0, 6)
        )
        assert float(res) == pytest.approx(0.419, abs=0.06)

    def test_rosenstein_recovers_mlle(self, henon_x: np.ndarray) -> None:
        res = lyapunov_from_data(
            henon_x, dimension=4, delay=1, k_max=12, method="rosenstein", fit=(0, 8)
        )
        assert float(res) == pytest.approx(0.419, abs=0.08)

    def test_auto_fit_is_sensible(self, henon_x: np.ndarray) -> None:
        # The default (no explicit fit) must land near the true value for a
        # clean, monotone divergence curve like the Hénon map's.
        res = lyapunov_from_data(henon_x, dimension=2, delay=1, k_max=12)
        assert 0.30 < float(res) < 0.55
        lo, hi = res.fit_region
        assert 0 <= lo < hi <= 12

    def test_multivariate_input(self, henon_x: np.ndarray) -> None:
        traj = ts.systems.Henon().run(6000, transient=500, ic=[0.1, 0.1])
        res = lyapunov_from_data(
            traj.y, dimension=2, delay=1, theiler=2, k_max=12, method="kantz", fit=(0, 6)
        )
        assert float(res) == pytest.approx(0.419, abs=0.08)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


class TestResult:
    def test_fields_and_float(self, henon_x: np.ndarray) -> None:
        res = lyapunov_from_data(henon_x, dimension=3, delay=1, k_max=15, dt=1.0, fit=(0, 7))
        assert isinstance(res, LyapunovFromData)
        assert res.embedding_dim == 3
        assert res.delay == 1
        assert res.theiler == 2  # (m-1)*tau
        assert res.method == "kantz"
        assert res.times.shape == res.divergence.shape == (16,)  # k = 0..k_max
        np.testing.assert_allclose(res.times, np.arange(16.0))  # dt = 1
        assert res.fit_region == (0, 7)
        assert float(res) == res.lyapunov
        assert "λ_max" in repr(res) and "kantz" in repr(res)

    def test_dt_scales_exponent(self, henon_x: np.ndarray) -> None:
        # Per-time exponent halves when each sample spans twice the time.
        a = lyapunov_from_data(henon_x, dimension=3, k_max=12, dt=1.0, fit=(0, 6))
        b = lyapunov_from_data(henon_x, dimension=3, k_max=12, dt=2.0, fit=(0, 6))
        assert float(b) == pytest.approx(0.5 * float(a), rel=1e-12)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dimension": 0}, "embedding dimension"),
            ({"delay": 0}, "embedding delay"),
            ({"k_max": 1}, "k_max"),
            ({"n_neighbors": 0}, "n_neighbors"),
            ({"dt": 0.0}, "dt"),
            ({"method": "nope"}, "method"),
            ({"theiler": -1}, "theiler"),
            ({"fit": (5, 5)}, "fit region"),
            ({"fit": (0, 99)}, "fit region"),
        ],
    )
    def test_bad_params_raise(self, henon_x: np.ndarray, kwargs: dict, match: str) -> None:
        call = {"k_max": 10, **kwargs}  # kwargs overrides the default k_max
        with pytest.raises(ValueError, match=match):
            lyapunov_from_data(henon_x, **call)

    def test_constant_series_raises(self) -> None:
        with pytest.raises(ValueError, match="eps must be positive"):
            lyapunov_from_data(np.ones(500), dimension=2, k_max=10, method="kantz")

    def test_k_max_too_large_for_series(self) -> None:
        # n_rows = 40 - (3-1)*2 = 36; k_max=40 leaves no forward images.
        with pytest.raises(ValueError, match="too large"):
            lyapunov_from_data(
                np.random.default_rng(0).normal(size=40), dimension=3, delay=2, k_max=40
            )

    def test_bad_params_raise_typed_invalid_parameter(self) -> None:
        """Reconstruction-parameter errors are the typed InvalidParameterError.

        Regression: these used to be bare ``ValueError``/numpy crashes. They must
        now subclass :class:`~tsdynamics.errors.InvalidParameterError` (still a
        ``ValueError``, so existing ``except ValueError`` keeps working).
        """
        with pytest.raises(InvalidParameterError, match="embedding dimension"):
            lyapunov_from_data(np.zeros(500), dimension=0, k_max=10)
        with pytest.raises(InvalidParameterError, match="method"):
            lyapunov_from_data(np.zeros(500), method="nope", k_max=10)
        with pytest.raises(InvalidParameterError, match="fit region"):
            lyapunov_from_data(np.arange(500.0), k_max=10, fit=(0, 99))

    def test_too_short_series_raises_typed_error(self) -> None:
        """A series too short for the embedding raises a clean typed error, not a numpy crash."""
        with pytest.raises(InvalidParameterError, match="too short"):
            lyapunov_from_data(np.arange(4.0), dimension=4, delay=3, k_max=2)

    def test_no_neighbours_raises_convergence_error(self) -> None:
        """No neighbour within ``eps`` outside the Theiler window → ConvergenceError.

        Regression: a strictly increasing ramp queried with a tiny ``eps`` finds
        no dynamical neighbours, which used to crash on an empty reduction. It
        must now raise a clean :class:`~tsdynamics.errors.ConvergenceError`.
        """
        ramp = np.arange(500.0)  # consecutive points differ by 1.0
        with pytest.raises(ConvergenceError, match="neighbour"):
            lyapunov_from_data(ramp, dimension=2, delay=1, k_max=10, method="kantz", eps=1e-9)


# ---------------------------------------------------------------------------
# Lorenz flow — slow tier (the literature acceptance value)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("method", ["kantz", "rosenstein"])
def test_lorenz_from_x_series(method: str) -> None:
    lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
    traj = lor.run(final_time=300.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    xs = traj.y[1000:, 0]  # drop the initial transient
    # Fit the settled linear scaling region (t ≈ 0.8–1.9), past the early
    # overshoot and before saturation.
    res = lyapunov_from_data(
        xs, dt=0.05, dimension=5, delay=3, k_max=60, method=method, fit=(16, 38)
    )
    assert float(res) == pytest.approx(0.906, abs=0.15)


# ---------------------------------------------------------------------------
# Automatic scaling-region selection
#
# The v6 defect: the automatic region was hardwired to ``[0, knee]``, i.e. it
# always anchored at k = 0 and therefore fitted the *initial transient* of the
# stretching curve — the part before neighbours have aligned with the unstable
# manifold, where the local slope is many times the Lyapunov exponent.  Combined
# with a fixed ``delay=1`` (a near-collinear embedding for any finely-sampled
# flow) and a fixed ``k_max=20`` (a look-ahead far shorter than one e-folding
# time), an oversampled Lorenz x-series returned 11.34 against a true 0.906 —
# 12.5x high, with no warning.  The only test covered a map.
# ---------------------------------------------------------------------------


def _synthetic_curve(dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A textbook stretching curve: steep transient, plateau of slope 1, saturation."""
    k = np.arange(60, dtype=float)
    s = np.empty_like(k)
    s[:10] = -4.0 + 0.30 * k[:10]  # transient, slope 0.30/sample
    s[10:40] = s[9] + 0.10 * (k[10:40] - k[9])  # scaling region, slope 0.10/sample
    s[40:] = s[39] + 0.002 * (k[40:] - k[39])  # saturation, ~flat
    return k * dt, s


class TestAutoScalingRegion:
    def test_picks_the_plateau_not_the_transient_or_the_tail(self) -> None:
        from tsdynamics.analysis.lyapunov.from_data import _auto_fit_region

        t, s = _synthetic_curve()
        region, _peak = _auto_fit_region(t, s)
        assert region is not None
        lo, hi = region
        assert lo >= 10, "region must start above the transient, not at k = 0"
        assert hi <= 41, "region must stop at the onset of saturation"
        slope = np.polyfit(t[lo : hi + 1], s[lo : hi + 1], 1)[0]
        assert slope == pytest.approx(0.10, rel=0.1)

    def test_refuses_a_curve_that_is_all_transient(self) -> None:
        """A curve truncated before it plateaus has no scaling region to report."""
        from tsdynamics.analysis.lyapunov.from_data import _auto_fit_region

        k = np.arange(30, dtype=float)
        decaying = 3.0 * (1.0 - np.exp(-k / 6.0))  # slope falls monotonically
        assert _auto_fit_region(k, decaying)[0] is None

    def test_refuses_a_saturated_curve(self) -> None:
        """A dead-flat curve is the longest, flattest 'plateau' there is — and useless."""
        from tsdynamics.analysis.lyapunov.from_data import _auto_fit_region

        k = np.arange(40, dtype=float)
        assert _auto_fit_region(k, np.zeros_like(k))[0] is None

    def test_auto_delay_is_one_for_a_map_and_grows_with_oversampling(self) -> None:
        from tsdynamics.analysis.lyapunov.from_data import _auto_delay

        rng = np.random.default_rng(0)
        tau, found = _auto_delay(rng.standard_normal(4000))
        assert (tau, found) == (1, True)
        # A sine at 40 samples/period decorrelates to 1/e about a sixth of a
        # period in — several samples, never 1.
        tau, found = _auto_delay(np.sin(np.arange(4000) * 2 * np.pi / 40.0))
        assert found and 4 <= tau <= 12

    def test_auto_delay_reports_a_series_with_no_decorrelation_scale(self) -> None:
        """A monotone ramp never decorrelates: the fallback must not be diagnosed on."""
        from tsdynamics.analysis.lyapunov.from_data import _auto_delay

        _, found = _auto_delay(np.arange(2000.0))
        assert not found


class TestDefaultsRecoverTheExponent:
    """The headline contract: at its defaults the estimator lands on the truth.

    One map with an *exact* exponent, one map with a literature value, and two
    flows — the flows are what the old defaults got wrong.
    """

    def test_logistic_r4_is_ln_two(self) -> None:
        x = ts.systems.Logistic(params={"r": 4.0}).run(steps=20_000, ic=[0.1]).y[2000:, 0]
        res = lyapunov_from_data(x)
        assert res.trusted
        assert float(res) == pytest.approx(np.log(2.0), abs=0.05)

    def test_henon(self, henon_x: np.ndarray) -> None:
        res = lyapunov_from_data(henon_x)
        assert res.trusted
        assert float(res) == pytest.approx(0.419, abs=0.05)

    @pytest.mark.slow
    def test_lorenz_oversampled_flow(self) -> None:
        """The reproducer: Lorenz x(t) at dt = 0.02 used to return 11.34 (12.5x high)."""
        lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
        x = lor.run(final_time=250.0, dt=0.02, ic=[1.0, 1.0, 1.0]).after(50.0).y[:, 0]
        res = lyapunov_from_data(x, dt=0.02)
        assert res.trusted
        assert res.delay > 1, "the delay must be read from the data, not fixed at 1"
        assert float(res) == pytest.approx(0.906, rel=0.25)

    @pytest.mark.slow
    def test_rossler_flow(self) -> None:
        ros = ts.systems.Rossler(ic=[1.0, 1.0, 1.0])
        x = ros.run(final_time=3000.0, dt=0.1, ic=[1.0, 1.0, 1.0]).after(200.0).y[:, 0]
        res = lyapunov_from_data(x, dt=0.1)
        assert res.trusted
        assert float(res) == pytest.approx(0.0714, rel=0.25)


class TestRefusesToGuess:
    @pytest.mark.slow
    def test_degenerate_embedding_warns_and_is_flagged_untrusted(self) -> None:
        """The old defaults (m=3, tau=1, k_max=20) on an oversampled flow.

        They still produce a number — a caller may force any reconstruction —
        but it must now arrive with a loud warning and ``trusted=False`` instead
        of looking exactly like a good answer.
        """
        from tsdynamics.analysis.lyapunov.from_data import ScalingRegionWarning

        lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
        x = lor.run(final_time=250.0, dt=0.02, ic=[1.0, 1.0, 1.0]).after(50.0).y[:, 0]
        with pytest.warns(ScalingRegionWarning, match="near-collinear"):
            res = lyapunov_from_data(x, dt=0.02, dimension=3, delay=1, k_max=20)
        assert not res.trusted
        assert "UNTRUSTED" in repr(res)

    def test_no_scaling_region_is_flagged_untrusted_and_returns_the_curve(self) -> None:
        """A curve with no plateau flags itself untrusted and still hands back S(k).

        This one is deliberately *not* a warning: a regular signal has no
        exponential scaling region by definition, so warning here would fire on
        correct answers. The flag is machine-checkable and cannot be filtered
        away, and it also lands in ``repr`` and ``meta``.
        """
        import warnings

        # White noise: neighbours separate to the attractor scale in one step, so
        # the stretching curve is a step followed by a flat tail — no plateau.
        x = np.random.default_rng(0).standard_normal(4000)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = lyapunov_from_data(x, dimension=3, delay=1, k_max=20)
        assert not res.trusted
        assert "UNTRUSTED" in repr(res)
        assert res.divergence.shape == (21,)
        assert res.meta["trusted"] is False
        assert res.meta["scaling_region"].startswith("none")

    def test_a_regular_signal_reports_a_near_zero_exponent_without_a_warning(self) -> None:
        """A periodic signal is not a failure — its exponent is 0 and no plateau exists."""
        import warnings

        j = np.arange(4000, dtype=float)
        x = np.cos(2.0 * np.pi * 0.02 * j) + 0.5 * np.cos(2.0 * np.pi * 0.04 * j + 0.7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = lyapunov_from_data(x)
        assert abs(float(res)) < 0.05

    def test_explicit_fit_is_trusted_and_silent(self, henon_x: np.ndarray) -> None:
        """Passing ``fit`` takes ownership: no plateau search, no warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = lyapunov_from_data(henon_x, dimension=4, delay=1, k_max=12, fit=(0, 6))
        assert res.trusted
        assert res.fit_region == (0, 6)


# ---------------------------------------------------------------------------
# The automatic look-ahead must stay affordable
#
# ``k_max = 25 * delay`` is unbounded in the oversampling factor of the input,
# and BOTH estimators are unbounded in ``k_max``: the Kantz loop is linear in it,
# and the Rosenstein path materialises an ``(n_ref, k_max + 1, m)`` array.
# Measured on a Lorenz x-series, the automatic value is 425 at dt = 0.02 but
# 4125 at dt = 0.002 and 8225 at dt = 0.001 — the last being ~90 GB of allocation
# on the Rosenstein path, from a call that passed no ``k_max`` at all.
# ---------------------------------------------------------------------------


def _long_correlated_noise(n: int = 6000, phi: float = 0.97) -> np.ndarray:
    """AR(1) with a ~35-sample decorrelation time: oversampled, and not chaotic."""
    rng = np.random.default_rng(0)
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x


class TestAutoKmaxIsBounded:
    def test_a_heavily_oversampled_series_caps_k_max_and_says_so(self) -> None:
        from tsdynamics.analysis.lyapunov.from_data import _KMAX_CEILING, ScalingRegionWarning

        x = _long_correlated_noise()
        with pytest.warns(ScalingRegionWarning, match="capped at"):
            res = lyapunov_from_data(x)
        assert res.meta["k_max"] == _KMAX_CEILING
        # ...and the capped curve is honestly reported as unusable rather than
        # fitted: correlated noise has no exponential scaling region.
        assert not res.trusted
        assert abs(float(res)) < 0.05

    def test_a_normally_sampled_series_is_untouched_and_silent(self, henon_x: np.ndarray) -> None:
        """The cap must not fire on anything the estimator handles well."""
        import warnings

        from tsdynamics.analysis.lyapunov.from_data import _KMAX_CEILING

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = lyapunov_from_data(henon_x)
        assert res.meta["k_max"] < _KMAX_CEILING
        assert res.trusted


class TestSamplingIntervalIsReadFromTheData:
    """A Trajectory knows its own ``dt``; the estimator must use it.

    ``traj.lyap.from_data()`` used to return 0.0164 for a Lorenz run sampled at
    ``dt = 0.02`` — an exponent per *sample*, where the number that compares
    with ``sys.lyap.spectrum()`` (0.92 per time unit) is ~60x larger.  Nothing
    was missing: the object was holding the sampling interval and the estimator
    was not reading it.  That is the whole reason a caller reaches for the
    accessor instead of the free function.
    """

    def test_a_trajectory_supplies_its_own_dt(self) -> None:
        import tsdynamics as ts

        lor = ts.systems.Lorenz()
        traj = lor.run(final_time=120.0, dt=0.02, ic=[1.0, 1.0, 1.0])
        per_time = lyapunov_from_data(traj, dimension=3, delay=8, k_max=200)
        per_sample = lyapunov_from_data(np.asarray(traj["x"]), dimension=3, delay=8, k_max=200)
        # The array form has no time axis, so it stays per sample (documented).
        assert float(per_time) == pytest.approx(float(per_time), abs=0)
        assert float(per_time) > 20.0 * float(per_sample)
        # ...and the per-time answer is in the right neighbourhood of the truth.
        assert 0.4 < float(per_time) < 1.6

    def test_a_decimated_trajectory_reads_its_own_axis_not_the_recorded_dt(self) -> None:
        """The v6 fix.  ``meta["dt"]`` used to record what the RUN asked for and
        slicing carried it verbatim, so ``tr[::5]`` reported the undecimated step
        and the exponent came back off by exactly the decimation factor —
        silently.  Round 8 also re-derives ``meta`` on a row selection, so the
        stale pair is built by hand here: the *reading rule* (the axis wins, and
        ``meta`` is only consulted when there is no usable axis) is the contract,
        and it has to hold for any trajectory however its ``meta`` was set."""
        import tsdynamics as ts
        from tsdynamics.data import Trajectory

        full = ts.systems.Lorenz().run(final_time=60.0, dt=0.01, transient=20.0, ic=[1.0, 1.0, 1.0])
        decimated = full[::5]
        sub = Trajectory(
            decimated.t, decimated.y, decimated.system, meta={**decimated.meta, "dt": 0.01}
        )
        assert sub.meta["dt"] == pytest.approx(0.01)  # stale by construction
        assert float(np.diff(sub.t)[0]) == pytest.approx(0.05)  # the truth
        derived = lyapunov_from_data(sub, dimension=3, fit=(2, 12))
        explicit = lyapunov_from_data(sub, dimension=3, fit=(2, 12), dt=0.05)
        assert float(derived) == pytest.approx(float(explicit), rel=1e-9)
        # ...and NOT the 5x-too-large number meta["dt"] would have produced.
        stale = lyapunov_from_data(sub, dimension=3, fit=(2, 12), dt=0.01)
        assert float(derived) == pytest.approx(float(stale) / 5.0, rel=1e-9)

    def test_an_explicit_dt_still_wins(self) -> None:
        """Reading a default must never override a value the caller typed."""
        import tsdynamics as ts

        traj = ts.systems.Lorenz().run(final_time=60.0, dt=0.02, ic=[1.0, 1.0, 1.0])
        pinned = lyapunov_from_data(traj, dimension=3, delay=8, k_max=200, dt=1.0)
        read = lyapunov_from_data(traj, dimension=3, delay=8, k_max=200)
        assert float(read) == pytest.approx(float(pinned) / 0.02)

    def test_a_hand_built_trajectory_reads_its_time_axis(self) -> None:
        """A measured ``(t, y)`` carries no ``meta['dt']`` — the axis is the source."""
        import tsdynamics as ts

        traj = ts.systems.Lorenz().run(final_time=60.0, dt=0.02, ic=[1.0, 1.0, 1.0])
        x = np.asarray(traj["x"])
        measured = ts.data.Trajectory(np.arange(x.size) * 0.02, x[:, None])
        assert measured.meta.get("dt") is None
        assert float(
            lyapunov_from_data(measured, dimension=3, delay=8, k_max=200)
        ) == pytest.approx(float(lyapunov_from_data(x, dimension=3, delay=8, k_max=200, dt=0.02)))

    def test_a_non_uniform_time_axis_is_refused_rather_than_guessed(self) -> None:
        import tsdynamics as ts
        from tsdynamics.errors import InvalidParameterError

        t = np.cumsum(np.linspace(0.01, 0.05, 600))
        y = np.sin(t)[:, None]
        with pytest.raises(InvalidParameterError, match="not uniformly"):
            lyapunov_from_data(ts.data.Trajectory(t, y), dimension=3, delay=8, k_max=200)


# ---------------------------------------------------------------------------
# `trusted` is a screen, and a verdict is never silence (v6 round 6)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lorenz_x() -> np.ndarray:
    """A long, transient-free Lorenz ``x`` series at ``dt = 0.02``."""
    return np.asarray(
        ts.systems.Lorenz().run(final_time=500.0, dt=0.02, transient=20.0, ic=[1.0, 1.0, 1.0])["x"]
    )


@pytest.fixture(scope="module")
def rossler_x() -> np.ndarray:
    """A long, transient-free Rössler ``x`` series at ``dt = 0.05``."""
    return np.asarray(
        ts.systems.Rossler().run(final_time=2000.0, dt=0.05, transient=100.0, ic=[1.0, 1.0, 1.0])[
            "x"
        ]
    )


class TestTrustedIsNotBoughtByAStraightLine:
    """``trusted`` must fold in the record, the fitted window and decorrelation.

    Measured at HEAD before this change, all with ``trusted = True``:

    ======= ======  ====== ======= =========================================
    series  n       λ      truth   why the old screen passed it
    ======= ======  ====== ======= =========================================
    Lorenz  2000    1.2414 0.9056  R² = 0.99872 on a fit anchored at k = 0
    Lorenz  3000    1.1204 0.9056  22.9 windows cleared a floor of 10
    Rössler 5000    0.0383 0.0714  long record, clean fit, 0.28 e-folds
    ======= ======  ====== ======= =========================================
    """

    def test_a_short_record_is_refused_however_straight_the_fit(self, lorenz_x: np.ndarray) -> None:
        res = lyapunov_from_data(lorenz_x[:2000], dt=0.02)
        assert res.r_squared > 0.99, "the fit really is straight — that is the point"
        assert float(res) > 1.2, "and really is 37% high"
        assert res.trusted is False
        assert res.chaotic is None

    def test_a_long_enough_record_is_believed(self, lorenz_x: np.ndarray) -> None:
        res = lyapunov_from_data(lorenz_x[:5000], dt=0.02)
        assert res.trusted is True
        assert res.chaotic is True
        # rel=0.25, not 0.15: this is a finite-sample estimate from 5 000 points,
        # and which neighbour each reference point picks is decided by distances
        # that differ in the last bits across platforms.  Measured: 0.9204 on
        # linux/x86-64, 1.0624 on macOS — a 15 % span for the same input, which
        # is the estimator's own scatter at this record length, not a defect.
        # It stays a real assertion: the short-record case below is 1.2+, still
        # outside this band, so the two remain distinguishable.
        assert float(res) == pytest.approx(0.9056, rel=0.25)

    def test_a_fit_the_separation_barely_grows_across_is_refused(
        self, rossler_x: np.ndarray
    ) -> None:
        """The case no record-length floor can catch: long record, clean, wrong."""
        with pytest.warns(ScalingRegionWarning, match="capped at 500"):
            res = lyapunov_from_data(rossler_x[:5000], dt=0.05)
        assert res.independent_windows > 25.0, "the record is long"
        assert res.r_squared > 0.99, "the fit is clean"
        assert res.fit_efoldings < 0.5, "and the divergence hardly moves across it"
        assert res.trusted is False
        assert "e-folds" in repr(res)

    def test_the_same_series_with_more_look_ahead_is_believed(self, rossler_x: np.ndarray) -> None:
        with pytest.warns(ScalingRegionWarning, match="capped at 500"):
            res = lyapunov_from_data(rossler_x[:10000], dt=0.05)
        assert res.trusted is True
        # rel=0.3 for the same reason as the Lorenz case above.  Measured:
        # 0.0724 on linux/x86-64, 0.0570 on macOS.
        assert float(res) == pytest.approx(0.0714, rel=0.3)

    def test_shrinking_the_embedding_cannot_buy_back_trust(self, lorenz_x: np.ndarray) -> None:
        """A remedy that restores RUNNABILITY must not restore CONFIDENCE.

        ``independent_windows`` used to divide by ``(m - 1) * τ`` alone, which is
        entirely the caller's to choose — so the one line the library hands a
        reader whose call was refused ("reduce dimension/delay") also flipped the
        flag: measured, 500 Lorenz samples at ``dimension=2, delay=1`` took the
        count from 2.2 to **358.0**, a 160x jump in apparent record length bought
        by shrinking the reconstruction rather than by measuring anything.  The
        series' own decorrelation time floors the denominator now.
        """
        short = lorenz_x[:500]
        honest = lyapunov_from_data(short, dt=0.02)
        with pytest.warns(ScalingRegionWarning, match="near-collinear"):
            gamed = lyapunov_from_data(short, dt=0.02, dimension=2, delay=1)
        assert honest.trusted is False
        assert gamed.independent_windows < 60.0, "the decorrelation floor must bind"
        assert gamed.trusted is False
        assert gamed.chaotic is None


class TestTheVerdictIsHedgedNeverSilent:
    """Billed as "the one number that says chaotic", it must say something.

    It used to print the failure and nothing about the system — while
    ``zero_one_test``, which carries no trust flag at all, answered
    ``chaotic (K ~ 1)`` on the same data.
    """

    def test_an_untrusted_result_still_names_the_sign(self, lorenz_x: np.ndarray) -> None:
        res = lyapunov_from_data(lorenz_x[:2000], dt=0.02)
        text = repr(res)
        assert "λ > 0" in text, "the measurement must be reported, hedged"
        assert "INDETERMINATE" in text, "and the hedge must be unmistakable"
        assert "UNTRUSTED" in text

    def test_the_hedge_is_not_a_branchable_yes(self, lorenz_x: np.ndarray) -> None:
        """``chaotic`` stays three-valued, so ``if res.chaotic:`` cannot misread it."""
        assert lyapunov_from_data(lorenz_x[:2000], dt=0.02).chaotic is None

    def test_the_reason_names_the_action_that_would_fix_it(
        self, lorenz_x: np.ndarray, rossler_x: np.ndarray
    ) -> None:
        """Three failures, three different actions — never the wrong one."""
        short = repr(lyapunov_from_data(lorenz_x[:2000], dt=0.02))
        assert "measure longer" in short
        with pytest.warns(ScalingRegionWarning, match="capped at 500"):
            narrow = repr(lyapunov_from_data(rossler_x[:5000], dt=0.05))
        assert "larger k_max" in narrow
        with pytest.warns(ScalingRegionWarning, match="near-collinear"):
            degenerate = lyapunov_from_data(lorenz_x[:6000], dt=0.02, delay=1, dimension=3)
        assert "decorrelation time" in repr(degenerate)

    def test_the_diagnostics_reach_to_dict(self, lorenz_x: np.ndarray) -> None:
        data = lyapunov_from_data(lorenz_x[:5000], dt=0.02).to_dict(full=True)
        assert data["chaotic"] is True
        assert data["independent_windows"] > 25.0
        assert data["fit_efoldings"] >= 0.5
