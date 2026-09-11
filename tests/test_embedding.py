r"""
Tests for the delay-embedding toolkit (stream **A-EMBED**).

The headline acceptance is literature-validated and self-contained: the Rössler
attractor is reconstructed from its :math:`x` component alone (Takens, 1981) and
the correlation dimension of the reconstruction recovers the true value
:math:`D_2 \approx 2.0`.  Source signals are generated with ``scipy`` (independent
of the v2 compile backend and the Rust engine streams), so these tests stay in
the fast tier and exercise only the estimators.

The estimators are pinned against known minimum embedding dimensions — a sine
(circle, :math:`m = 2`) and the Rössler/Lorenz attractors (:math:`m = 3`) — and
Cao's :math:`E_2` is checked to separate determinism from white noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis import embedding as emb
from tsdynamics.data import Trajectory

# ── data generators (scipy — independent of the engine) ─────────────────────────


def _rossler(n=6000, dt=0.05, transient=2000, a=0.2, b=0.2, c=5.7):
    def rhs(_t, u):
        x, y, z = u
        return [-y - z, x + a * y, b + z * (x - c)]

    t_end = (n + transient) * dt
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(
        rhs, (0.0, t_end), [1.0, 1.0, 1.0], t_eval=t_eval, method="DOP853", rtol=1e-9, atol=1e-9
    )
    return sol.t[transient : transient + n], sol.y.T[transient : transient + n]


def _lorenz(n=6000, dt=0.02, transient=2000):
    def rhs(_t, u):
        x, y, z = u
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]

    t_end = (n + transient) * dt
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(
        rhs, (0.0, t_end), [1.0, 1.0, 1.0], t_eval=t_eval, method="DOP853", rtol=1e-9, atol=1e-9
    )
    return sol.y.T[transient : transient + n]


@pytest.fixture(scope="module")
def rossler():
    return _rossler()


@pytest.fixture(scope="module")
def lorenz():
    return _lorenz()


@pytest.fixture(scope="module")
def sine():
    t = np.linspace(0.0, 200.0, 8000)
    return np.sin(2.0 * np.pi * 0.7 * t)


# ── embed: shapes, semantics, multivariate ──────────────────────────────────────


def test_embed_small_exact():
    x = np.arange(10.0)
    y = emb.embed(x, dimension=3, delay=2)
    # rows = 10 - (3-1)*2 = 6; row i = [x[i], x[i+2], x[i+4]]
    assert y.shape == (6, 3)
    np.testing.assert_array_equal(y[0], [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(y[-1], [5.0, 7.0, 9.0])
    # the first column is just the (truncated) original series
    np.testing.assert_array_equal(y[:, 0], x[:6])


def test_embed_dimension_one_is_identity_column():
    x = np.arange(8.0)
    y = emb.embed(x, dimension=1, delay=3)
    assert y.shape == (8, 1)
    np.testing.assert_array_equal(y[:, 0], x)


def test_embed_multivariate_two_channels():
    a = np.arange(12.0)
    b = 100.0 + np.arange(12.0)
    data = np.column_stack([a, b])
    y = emb.embed(data, dimension=2, delay=3)
    # per-channel span = (2-1)*3 = 3; rows = 12 - 3 = 9; cols = 2 + 2 = 4
    assert y.shape == (9, 4)
    np.testing.assert_array_equal(y[0], [a[0], a[3], b[0], b[3]])
    np.testing.assert_array_equal(y[-1], [a[8], a[11], b[8], b[11]])


def test_embed_multivariate_per_channel_params():
    a = np.arange(20.0)
    b = np.arange(20.0) * 2.0
    y = emb.embed([a, b], dimension=[2, 3], delay=[2, 1])
    # spans: chan a (2-1)*2=2, chan b (3-1)*1=2; rows = 20 - 2 = 18; cols = 2 + 3 = 5
    assert y.shape == (18, 5)
    np.testing.assert_array_equal(y[0], [a[0], a[2], b[0], b[1], b[2]])


def test_embed_component_selection_and_errors():
    data = np.column_stack([np.arange(10.0), -np.arange(10.0)])
    y = emb.embed(data, dimension=2, delay=1, component=1)
    np.testing.assert_array_equal(y[:, 0], -np.arange(9.0))
    with pytest.raises(ValueError, match="per-channel"):
        emb.embed(np.arange(10.0), dimension=[2, 2], delay=1)
    with pytest.raises(ValueError, match="too short"):
        emb.embed(np.arange(5.0), dimension=4, delay=3)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            emb.embed(np.arange(10.0), dimension=bad, delay=1)
        with pytest.raises(ValueError):
            emb.embed(np.arange(10.0), dimension=2, delay=bad)


def test_embed_accepts_trajectory(rossler):
    t, y = rossler
    traj = Trajectory(t, y, system=None)
    out = emb.embed(traj, dimension=2, delay=5, component=0)
    assert out.shape == (y.shape[0] - 5, 2)
    np.testing.assert_allclose(out[:, 0], y[: y.shape[0] - 5, 0])


# ── delay selection: autocorrelation & mutual information ────────────────────────


def test_autocorrelation_properties(rossler):
    _, y = rossler
    acf = emb.autocorrelation(y[:, 0], max_delay=40)
    assert acf.shape == (41,)
    assert acf[0] == pytest.approx(1.0)
    assert np.all(np.abs(acf) <= 1.0 + 1e-9)


def test_autocorrelation_constant_raises():
    with pytest.raises(ValueError, match="constant"):
        emb.autocorrelation(np.ones(100), max_delay=10)


def test_mutual_information_self_is_max(rossler):
    _, y = rossler
    mi = emb.mutual_information(y[:, 0], max_delay=40, bins=32)
    assert mi.shape == (41,)
    # I(0) is the series' self-information — the largest value of the curve.
    assert mi[0] == pytest.approx(mi.max())
    assert np.all(mi[1:] <= mi[0] + 1e-9)
    assert np.all(mi >= -1e-9)


def _mi_reference(x, *, max_delay, nbins, base=np.e):
    """Independent histogram MI using ``np.add.at`` (the pre-bincount scatter).

    Mirrors the estimator's binning exactly so the only difference from the
    public function is the joint-histogram construction; both must agree.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=float))
    n = x.size
    lo, hi = float(x.min()), float(x.max())
    edges = np.linspace(lo, hi, nbins + 1)
    codes = np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)
    log = np.log if base == np.e else (lambda v: np.log(v) / np.log(base))
    mi = np.empty(max_delay + 1, dtype=float)
    for tau in range(max_delay + 1):
        a = codes[: n - tau]
        b = codes[tau:] if tau > 0 else codes
        joint = np.zeros((nbins, nbins), dtype=float)
        np.add.at(joint, (a, b), 1.0)
        joint /= joint.sum()
        p_a = joint.sum(axis=1)
        p_b = joint.sum(axis=0)
        mask = joint > 0.0
        outer = p_a[:, None] * p_b[None, :]
        mi[tau] = float(np.sum(joint[mask] * log(joint[mask] / outer[mask])))
    return mi


def test_mutual_information_matches_add_at_reference(rossler):
    """The bincount joint histogram is bit-for-bit equal to the np.add.at scatter.

    Regression guard for the ``np.add.at`` -> ``np.bincount`` rewrite in
    :func:`mutual_information`: the linear-index ``bincount`` must reproduce the
    per-element scatter exactly (same counts, same order), so the resulting MI
    curve is unchanged.
    """
    _, y = rossler
    x = y[:1500, 0]
    nbins, max_delay = 24, 30
    got = np.asarray(emb.mutual_information(x, max_delay=max_delay, bins=nbins))
    ref = _mi_reference(x, max_delay=max_delay, nbins=nbins)
    np.testing.assert_array_equal(got, ref)


def test_mutual_information_bincount_small_exact():
    """A tiny hand-checkable series: bincount MI equals the scatter reference exactly."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0])
    nbins, max_delay = 4, 4
    got = np.asarray(emb.mutual_information(x, max_delay=max_delay, bins=nbins))
    ref = _mi_reference(x, max_delay=max_delay, nbins=nbins)
    np.testing.assert_array_equal(got, ref)
    assert got[0] == pytest.approx(got.max())


def test_optimal_delay_rossler_reasonable(rossler):
    _, y = rossler
    tau_mi = emb.optimal_delay(y[:, 0], method="mi", max_delay=80)
    tau_acf = emb.optimal_delay(y[:, 0], method="acf", max_delay=80)
    # dt = 0.05; a delay of ~0.5–2.5 time units (10–50 samples) is the usual band.
    assert 8 <= tau_mi <= 55
    assert 8 <= tau_acf <= 55
    assert emb.optimal_delay(y[:, 0], method="acf_zero", max_delay=80) >= 1


def test_optimal_delay_unknown_method(rossler):
    _, y = rossler
    with pytest.raises(ValueError, match="unknown method"):
        emb.optimal_delay(y[:, 0], method="nope")


# ── embedding dimension: Cao & FNN against known values ──────────────────────────


def test_fnn_rossler_is_three(rossler):
    _, y = rossler
    x = y[:, 0]
    tau = emb.optimal_delay(x, method="mi", max_delay=80)
    fnn = emb.false_nearest_neighbors(x, delay=tau, max_dim=8, theiler=tau)
    assert fnn.method == "fnn"
    assert int(fnn) == 3, f"FNN dim = {int(fnn)} (E={fnn.fnn_fraction})"
    # The fraction has effectively vanished by d = 3 and stays there.
    assert fnn.fnn_fraction[2] < 0.02
    assert fnn.fnn_fraction[0] > 0.5  # 1-D is almost all false neighbours


def test_fnn_lorenz_is_three(lorenz):
    x = lorenz[:, 0]
    tau = emb.optimal_delay(x, method="mi", max_delay=60)
    fnn = emb.false_nearest_neighbors(x, delay=tau, max_dim=8, theiler=tau)
    assert int(fnn) == 3, f"FNN dim = {int(fnn)} (E={fnn.fnn_fraction})"


def test_fnn_sine_is_two(sine):
    tau = emb.optimal_delay(sine, method="mi", max_delay=80)
    fnn = emb.false_nearest_neighbors(sine, delay=tau, max_dim=6, theiler=tau)
    assert int(fnn) == 2, f"FNN dim = {int(fnn)} (E={fnn.fnn_fraction})"


def test_cao_lorenz_saturates_at_three(lorenz):
    x = lorenz[:, 0]
    tau = emb.optimal_delay(x, method="mi", max_delay=60)
    cao = emb.cao_dimension(x, delay=tau, max_dim=8, theiler=tau)
    assert cao.method == "cao"
    # E1 saturates to ~1; Cao's estimate is the literature value (or one above on
    # borderline samples — it errs toward over-embedding, never under).
    assert int(cao) in (3, 4), f"Cao dim = {int(cao)} (E1={cao.afn_e1})"
    assert cao.afn_e1[-1] > 0.95  # the curve has reached its plateau


def test_cao_e2_discriminates_noise_from_determinism(rossler):
    _, y = rossler
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(6000)

    cao_det = emb.cao_dimension(y[:, 0], delay=8, max_dim=8)
    cao_rnd = emb.cao_dimension(noise, delay=1, max_dim=8)

    # For white noise E2 stays ~1 at every dimension; for the deterministic series
    # it departs strongly from 1 at low dimension (Cao 1997, the E2 test).
    assert np.max(np.abs(cao_rnd.afn_e2 - 1.0)) < 0.1
    assert np.max(np.abs(cao_det.afn_e2 - 1.0)) > 0.3


def test_embedding_dimension_dispatch(rossler):
    _, y = rossler
    x = y[:, 0]
    tau = emb.optimal_delay(x, method="mi", max_delay=80)
    by_cao = emb.embedding_dimension(x, method="cao", delay=tau, max_dim=8, theiler=tau)
    by_fnn = emb.embedding_dimension(x, method="fnn", delay=tau, max_dim=8, theiler=tau)
    assert by_cao.method == "cao" and by_fnn.method == "fnn"
    with pytest.raises(ValueError, match="unknown method"):
        emb.embedding_dimension(x, method="nope")


def test_dimension_estimators_reject_bad_args():
    x = np.sin(np.linspace(0, 50, 2000))
    with pytest.raises(ValueError, match="delay must be"):
        emb.cao_dimension(x, delay=0)
    with pytest.raises(ValueError, match="max_dim must be"):
        emb.cao_dimension(x, max_dim=1)
    with pytest.raises(ValueError, match="too short"):
        emb.false_nearest_neighbors(np.arange(20.0), delay=5, max_dim=8)


# ── headline acceptance: reconstruct Rössler from x only ─────────────────────────


def test_reconstruct_rossler_from_x_only(rossler):
    """Takens reconstruction from x(t) alone recovers the Rössler D_2 ≈ 2.0."""
    from tsdynamics.analysis import correlation_dimension

    _, y = rossler
    x = y[:, 0]

    tau = emb.optimal_delay(x, method="mi", max_delay=80)
    m = int(emb.false_nearest_neighbors(x, delay=tau, max_dim=8, theiler=tau))
    assert m == 3

    reconstructed = emb.embed(x, dimension=m, delay=tau)
    assert reconstructed.shape[1] == 3

    d2 = correlation_dimension(reconstructed, theiler=2 * tau)
    assert 1.75 < float(d2) < 2.25, f"reconstructed Rössler D2 = {float(d2):.3f}, expected ~2.0"


# ── registry integration ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "embed",
        "optimal_delay",
        "mutual_information",
        "cao_dimension",
        "false_nearest_neighbors",
        "embedding_dimension",
    ],
)
def test_estimators_self_register(name):
    assert name in registry.analyses
    assert registry.analyses.get(name) is getattr(emb, name)


@pytest.mark.parametrize(
    "name",
    [
        "embed",
        "optimal_delay",
        "autocorrelation",
        "mutual_information",
        "cao_dimension",
        "false_nearest_neighbors",
        "embedding_dimension",
        "EmbeddingDimension",
    ],
)
def test_public_api_reexported(name):
    assert getattr(ts.analysis, name) is getattr(emb, name)
    # C2 — a type you only ever get *back* is reachable but off the tab surface.
    if name[:1].isupper():
        assert name in ts.analysis.results.__all__
    else:
        assert name in ts.analysis.__all__


# ── v6: the mutual-information noise guard (audit FIX-MI-NOISE) ─────────────────
#
# ``optimal_delay(method="mi")`` used to take the first *shape* minimum of
# I(tau).  On a chaotic map I(tau) decays monotonically to the histogram
# estimator's independence floor and then wobbles there; the first of those
# wobbles was read as "the" first minimum, and ``embedding_dimension`` reported
# ``max_dim`` instead of the true dimension.  The fix rejects minima at the
# floor; the fallback then splits on *why* no minimum was found.


def _henon_series(n=20000, transient=1000, a=1.4, b=0.3):
    """The x-component of a Henon orbit (minimum embedding dimension 2)."""
    x, y = 0.1, 0.1
    out = np.empty(n + transient)
    for i in range(n + transient):
        x, y = 1.0 - a * x * x + y, b * x
        out[i] = x
    return out[transient:]


@pytest.fixture(scope="module")
def henon_series():
    return _henon_series()


def test_mi_curve_of_a_map_has_no_significant_minimum(henon_series):
    """Truth: I(tau) for the Henon map decays monotonically to the noise floor.

    Independently of the selection rule, the curve is checked to have (a) no
    genuine dip in its informative part and (b) an interior *shape* minimum out
    in its floor region.  That is exactly the configuration the pre-fix rule
    mis-read, so it anchors the regression rather than restating the fix.
    """
    from tsdynamics.analysis.embedding.delay import _first_local_min, _mi_noise_floor

    mi = emb.mutual_information(henon_series, max_delay=50)
    curve = np.asarray(mi, dtype=float)
    floor = _mi_noise_floor(int(mi.meta["bins"]), int(mi.meta["n_samples"]))

    shape_min = _first_local_min(curve)
    assert shape_min is not None, "no interior shape minimum — regression premise gone"
    # ... and it sits down at the estimator's independence floor, i.e. it is noise.
    assert curve[shape_min] < 2.0 * floor
    # The informative part of the curve (above the floor) is strictly decreasing,
    # so there is no real dip for the criterion to find.
    informative = curve[curve > 2.0 * floor]
    assert np.all(np.diff(informative) < 0.0)


def test_optimal_delay_is_one_for_a_map(henon_series):
    """A fully decorrelated series gets tau = 1, not a floor-level artefact."""
    assert int(emb.optimal_delay(henon_series, method="mi", max_delay=50)) == 1


def test_henon_embedding_dimension_is_two(henon_series):
    """Headline: the automatic delay -> dimension chain recovers m = 2 for Henon.

    Pre-fix the delay came back as 23 (a noise-floor wobble) and both estimators
    returned ``max_dim``; the true minimum embedding dimension of the Henon
    attractor is 2.
    """
    tau = int(emb.optimal_delay(henon_series, method="mi", max_delay=50))
    assert int(emb.cao_dimension(henon_series, delay=tau, max_dim=8, theiler=tau)) == 2
    assert int(emb.false_nearest_neighbors(henon_series, delay=tau, max_dim=8, theiler=tau)) == 2


def test_lorenz_embedding_dimension_is_three(lorenz):
    """The same chain on a densely sampled flow keeps the genuine first minimum.

    The companion to the Henon case: here I(tau) *does* dip well above the noise
    floor, the guard must not reject it, and m = 3 must survive.
    """
    x = lorenz[:, 0]
    mi = emb.mutual_information(x, max_delay=60)
    from tsdynamics.analysis.embedding.delay import _mi_noise_floor

    floor = _mi_noise_floor(int(mi.meta["bins"]), int(mi.meta["n_samples"]))
    tau = int(emb.optimal_delay(x, method="mi", max_delay=60))
    assert tau > 1
    assert np.asarray(mi)[tau] > 5.0 * floor, "the genuine Lorenz dip is far above the floor"
    assert int(emb.false_nearest_neighbors(x, delay=tau, max_dim=8, theiler=tau)) == 3


def test_oversampled_flow_falls_back_to_the_longest_lag():
    """A monotone curve that is still informative at max_delay gets max_delay.

    The other half of the fallback: an oversampled Lorenz (dt = 0.002) does not
    decorrelate within 50 samples, so I(50) is still far above the floor.  Taking
    tau = 1 there would be the *opposite* error to the Henon one, so the two
    branches are pinned separately.
    """
    from tsdynamics.analysis.embedding.delay import _mi_noise_floor

    x = _lorenz(n=8000, dt=0.002)[:, 0]
    mi = emb.mutual_information(x, max_delay=50)
    curve = np.asarray(mi, dtype=float)
    floor = _mi_noise_floor(int(mi.meta["bins"]), int(mi.meta["n_samples"]))
    assert np.all(np.diff(curve[1:]) < 0.0), "premise: the curve is monotone over the window"
    assert curve[-1] > 2.0 * floor, "premise: still informative at the longest lag"
    assert int(emb.optimal_delay(x, method="mi", max_delay=50)) == 50


def test_mi_noise_floor_matches_the_chi_squared_bias():
    """The floor is the Miller-Madow bias, checked against a Monte-Carlo draw.

    For independent uniforms the plug-in mutual information over a ``B x B``
    table has expectation ``(B-1)**2 / (2N)``.  Checked directly against the
    estimator on independent noise, so the constant is not merely asserted.
    """
    from tsdynamics.analysis.embedding.delay import _mi_noise_floor

    rng = np.random.default_rng(0)
    n, bins = 20000, 32
    x = rng.uniform(0.0, 1.0, n)
    # Lags 1.. of white noise are independent pairs, so I(tau>0) is pure bias.
    curve = np.asarray(emb.mutual_information(x, max_delay=30, bins=bins))[1:]
    predicted = _mi_noise_floor(bins, n)
    assert abs(float(curve.mean()) - predicted) < 0.15 * predicted


def test_embedding_rejects_a_system_with_a_named_error():
    """A System handed to a data-first embedding routine names itself and the fix.

    One shared builder writes this text (CONTRACT §5.6), so the four doors below
    differ only in the name they carry — and a map is told to ``run(steps=...)``
    while a flow is told to ``run(200.0, dt=...)``, because that is the horizon
    word each family actually has.
    """
    from tsdynamics.errors import InvalidInputError

    for call, name, run in (
        (lambda: emb.optimal_delay(ts.systems.Lorenz()), "optimal_delay", "200.0"),
        (lambda: emb.embedding_dimension(ts.systems.Henon()), "embedding_dimension", "20000"),
        (lambda: emb.mutual_information(ts.systems.Lorenz()), "mutual_information", "200.0"),
        (
            lambda: emb.embed(ts.systems.Lorenz(), dimension=3, delay=1),
            "embed",
            "200.0",
        ),
    ):
        with pytest.raises(InvalidInputError) as excinfo:
            call()
        text = str(excinfo.value)
        assert text.startswith(f"{name}() needs data, and got a system")
        assert f"traj = system.run({run}" in text
        assert f"    ts.analysis.{name}(traj)" in text


# ---------------------------------------------------------------------------
# Omitted / misspelled parameters (stream v6 API-FOOTGUNS)
# ---------------------------------------------------------------------------
#
# ``embed`` is the flagship data-to-phase-space bridge, and it used to answer the
# two most likely first calls with raw binder errors: ``embed(x)`` with *missing 2
# required positional arguments*, and ``embed(x, dim=3)`` with *unexpected keyword
# argument 'dim'* — neither of which mentions that the library has estimators for
# exactly those two numbers, nor that ``dimension``/``delay`` are the canonical
# spellings the frozen glossary settled on.


def test_embed_estimates_both_parameters_when_neither_is_given(lorenz):
    """``embed(x)`` runs, and picks the delay and dimension the estimators would."""
    x = lorenz[:, 0]
    result = emb.embed(x)
    expected_delay = int(emb.optimal_delay(x))
    expected_dim = int(emb.embedding_dimension(x, delay=expected_delay))
    assert result.meta["delay"] == expected_delay
    assert result.meta["dimension"] == expected_dim
    assert result.shape == (x.size - (expected_dim - 1) * expected_delay, expected_dim)


def test_embed_estimated_lorenz_reconstruction_is_three_dimensional(lorenz):
    """The estimate is the *right* one: a scalar Lorenz record unfolds at m = 3."""
    assert emb.embed(lorenz[:, 0]).meta["dimension"] == 3


def test_embed_records_which_parameters_it_estimated(lorenz):
    """An estimated parameter is recorded, so a reconstruction says how it was made."""
    x = lorenz[:, 0]
    both = emb.embed(x)
    assert (both.meta["dimension_auto"], both.meta["delay_auto"]) == (True, True)
    given_dim = emb.embed(x, dimension=4)
    assert (given_dim.meta["dimension_auto"], given_dim.meta["delay_auto"]) == (False, True)
    assert given_dim.meta["dimension"] == 4
    given_both = emb.embed(x, dimension=4, delay=3)
    assert (given_both.meta["dimension_auto"], given_both.meta["delay_auto"]) == (False, False)


def test_embed_delay_is_estimated_before_the_dimension(lorenz):
    """The dimension is estimated *at* the resolved delay, not at an arbitrary lag.

    Cao's ratio is computed at a fixed delay, so estimating the dimension first
    would evaluate it at a lag that is then moved out from under it.
    """
    x = lorenz[:, 0]
    tau = int(emb.optimal_delay(x))
    assert emb.embed(x).meta["dimension"] == int(emb.embedding_dimension(x, delay=tau))
    # ...and a caller-supplied delay is the one used, not the estimated one
    assert emb.embed(x, delay=3).meta["dimension"] == int(emb.embedding_dimension(x, delay=3))


@pytest.mark.parametrize(
    "bad,canonical", [("dim", "dimension"), ("m", "dimension"), ("tau", "delay"), ("lag", "delay")]
)
def test_embed_banned_spellings_name_the_canonical_parameter(bad, canonical):
    """``embed(x, dim=3)`` names ``dimension=``; it is not a bare TypeError.

    The glossary bans these spellings, so they cannot simply be accepted — but a
    banned spelling must be *answered*, not met with Python's binder message.
    """
    from tsdynamics.errors import InvalidParameterError

    x = np.sin(np.linspace(0.0, 60.0, 400))
    with pytest.raises(InvalidParameterError) as excinfo:
        emb.embed(x, **{bad: 3})
    message = str(excinfo.value)
    assert f"{bad}=3 → {canonical}=3" in message
    assert "embed(data, dimension, delay" in message


def test_embed_unknown_keyword_says_it_is_not_a_parameter():
    """A keyword that is not a renamed one is still refused with the signature."""
    from tsdynamics.errors import InvalidParameterError

    x = np.sin(np.linspace(0.0, 60.0, 400))
    with pytest.raises(InvalidParameterError, match="nonsense= is not a parameter of embed"):
        emb.embed(x, nonsense=1)


def test_embed_multivariate_refuses_to_guess_per_channel_parameters(lorenz):
    """Auto-selection is univariate by construction, and says so.

    One series has one delay and one dimension; a bundle has one of each *per
    channel*, and inventing them from a single estimate would be a fiction.
    """
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError) as excinfo:
        emb.embed(lorenz)
    message = str(excinfo.value)
    assert "multivariate" in message
    assert "dimension=[3, 3]" in message
    # ...but selecting a channel gets the estimated univariate reconstruction
    assert emb.embed(lorenz, component=0).meta["delay_auto"] is True


def test_embed_keeps_the_glossary_spellings_in_its_signature():
    """The fix must not smuggle a banned spelling into the signature (glossary §2)."""
    import inspect

    params = inspect.signature(emb.embed).parameters
    assert "dimension" in params
    assert "delay" in params
    assert not {"dim", "m", "tau", "lag"} & set(params)


def test_embed_estimator_failure_advice_follows_the_diagnosis():
    """A constant series is not a *short* series, and is not told that it is.

    Adversarial follow-up: the estimator-failure hint was written for the one
    case that motivated it (too few samples) and applied unconditionally, so a
    500-sample constant series was told "series is constant; mutual information
    is undefined" and, in the very next sentence, "500 samples is not enough for
    the estimator to work with".  The second sentence contradicts the first and
    sends the reader to collect data that will fail identically.
    """
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError) as excinfo:
        emb.embed(np.ones(500))
    message = str(excinfo.value)
    assert "constant" in message
    assert "not enough" not in message
    assert "longer record" not in message
    assert "Pass the value explicitly" in message

    # ...while a genuinely short series still gets the length advice
    with pytest.raises(InvalidParameterError) as short:
        emb.embed(np.sin(np.arange(20) * 0.3))
    assert "not enough" in str(short.value)
