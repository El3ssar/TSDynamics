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
    assert getattr(ts, name) is getattr(emb, name)
    assert name in ts.analysis.__all__
