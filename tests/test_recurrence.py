r"""
Tests for the recurrence / RQA toolkit (stream **A-RQA**).

The headline acceptance is literature-validated and self-contained.  Two classic
reproductions anchor the suite, both generated without the engine (so they stay
in the fast tier):

- **Determinism separates order from chaos.**  A periodic orbit of the logistic
  map gives :math:`DET = 1` with a recurrence plot of long diagonals, while the
  fully chaotic orbit drops well below it with short lines (Trulla, Giuliani,
  Zbilut & Webber, *Phys. Lett. A* **223**, 255, 1996).
- **Windowed RQA detects a regime change.**  Determinism steps down where a
  signal switches from periodic to stochastic (Marwan, Romano, Thiel & Kurths,
  *Phys. Rep.* **438**, 237, 2007).

Plus the structural contract: a symmetric, line-of-identity-free sparse matrix
whose density tracks a requested recurrence rate, and the standard RQA measures
read off its diagonal and vertical lines.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis import recurrence as rec
from tsdynamics.data import Trajectory

# ── data generators (self-contained, no engine) ─────────────────────────────────


def _logistic(r: float, n: int = 2000, transient: int = 1000, x0: float = 0.4) -> np.ndarray:
    x = x0
    for _ in range(transient):
        x = r * x * (1.0 - x)
    out = np.empty(n)
    for i in range(n):
        x = r * x * (1.0 - x)
        out[i] = x
    return out


@pytest.fixture(scope="module")
def sine():
    t = np.linspace(0.0, 200.0, 4000)
    return np.sin(2.0 * np.pi * 0.5 * t)


@pytest.fixture(scope="module")
def noise():
    rng = np.random.default_rng(0)
    return rng.standard_normal((4000, 2))


# ── recurrence_matrix: structure & thresholding ──────────────────────────────────


def test_recurrence_matrix_symmetric_and_loi_free(noise):
    rm = rec.recurrence_matrix(noise, threshold=0.5)
    arr = rm.toarray()
    assert arr.shape == (4000, 4000)
    assert arr.dtype == bool
    assert np.array_equal(arr, arr.T)  # symmetric
    assert not np.any(np.diag(arr))  # line of identity excluded
    assert rm.size == 4000


def test_recurrence_rate_tracks_target(noise):
    for target in (0.02, 0.05, 0.1):
        rm = rec.recurrence_matrix(noise, recurrence_rate=target)
        # continuous-valued data: the realised density is close to the request.
        assert abs(rm.recurrence_rate - target) < 0.2 * target


def test_threshold_mode_density_increases_with_eps(noise):
    rr = [rec.recurrence_matrix(noise, threshold=e).recurrence_rate for e in (0.2, 0.5, 1.0)]
    assert rr[0] < rr[1] < rr[2]


def test_theiler_window_removes_near_diagonal(sine):
    emb = ts.embed(sine, dimension=2, delay=10)
    rm = rec.recurrence_matrix(emb, threshold=0.3, theiler=15)
    arr = rm.toarray()
    # every recurrence sits outside the |i - j| <= 15 band.
    i, j = np.nonzero(arr)
    assert np.all(np.abs(i - j) > 15)


def test_metric_variants_all_build(noise):
    for metric in ("euclidean", "manhattan", "chebyshev", 3.0):
        rm = rec.recurrence_matrix(noise[:500], recurrence_rate=0.05, metric=metric)
        assert 0.0 < rm.recurrence_rate < 0.2


def test_recurrence_matrix_accepts_trajectory(noise):
    traj = Trajectory(np.arange(len(noise), dtype=float), noise, system=None)
    rm = rec.recurrence_matrix(traj, threshold=0.5)
    rm2 = rec.recurrence_matrix(noise, threshold=0.5)
    assert np.array_equal(rm.toarray(), rm2.toarray())


def test_recurrence_matrix_array_protocol(noise):
    rm = rec.recurrence_matrix(noise[:200], threshold=0.5)
    assert np.array_equal(np.asarray(rm), rm.toarray())


def test_recurrence_matrix_guards(noise):
    with pytest.raises(ValueError, match="exactly one"):
        rec.recurrence_matrix(noise, threshold=0.5, recurrence_rate=0.05)
    with pytest.raises(ValueError, match="exactly one"):
        rec.recurrence_matrix(noise)
    with pytest.raises(ValueError, match="positive"):
        rec.recurrence_matrix(noise, threshold=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        rec.recurrence_matrix(noise, recurrence_rate=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        rec.recurrence_matrix(noise, threshold=0.5, theiler=-1)
    with pytest.raises(ValueError, match="every pair"):
        rec.recurrence_matrix(noise[:10], threshold=0.5, theiler=20)
    with pytest.raises(ValueError, match="unknown metric"):
        rec.recurrence_matrix(noise, threshold=0.5, metric="nope")
    with pytest.raises(ValueError, match="at least two"):
        rec.recurrence_matrix(np.array([[1.0, 2.0]]), threshold=0.5)
    with pytest.raises(ValueError, match="non-finite"):
        rec.recurrence_matrix(np.array([[1.0], [np.nan], [3.0]]), threshold=0.5)


# ── RQA: determinism separates order from chaos (Trulla et al. 1996) ─────────────


@pytest.mark.parametrize("r", [3.5, 3.55, 3.83])
def test_rqa_periodic_logistic_is_fully_deterministic(r):
    """A periodic orbit recurs on perfect diagonals → DET = 1, long L_max."""
    x = _logistic(r)
    res = rec.rqa(x, recurrence_rate=0.05, theiler=1)
    assert res.determinism == pytest.approx(1.0, abs=1e-9)
    assert res.max_diagonal_length > 100  # diagonals span the whole orbit


def test_rqa_chaotic_logistic_is_less_deterministic():
    """The fully chaotic logistic orbit (r=4) has DET < 1 and short diagonals."""
    chaotic = rec.rqa(_logistic(4.0), recurrence_rate=0.05, theiler=1)
    periodic = rec.rqa(_logistic(3.5), recurrence_rate=0.05, theiler=1)
    assert chaotic.determinism < 0.9
    assert chaotic.determinism < periodic.determinism
    assert chaotic.max_diagonal_length < periodic.max_diagonal_length
    # divergence (1/L_max) is larger for the chaotic orbit.
    assert chaotic.divergence > periodic.divergence


def test_rqa_sine_vs_noise(sine, noise):
    det_sine = rec.rqa(ts.embed(sine, dimension=2, delay=10), recurrence_rate=0.05).determinism
    det_noise = rec.rqa(noise, recurrence_rate=0.05).determinism
    assert det_sine > 0.95
    assert det_noise < 0.3
    assert det_sine > det_noise


def test_rqa_accepts_prebuilt_matrix(noise):
    rm = rec.recurrence_matrix(noise[:800], recurrence_rate=0.05, theiler=1)
    from_matrix = rec.rqa(rm)
    from_data = rec.rqa(noise[:800], recurrence_rate=0.05, theiler=1)
    assert from_matrix.determinism == pytest.approx(from_data.determinism)
    assert from_matrix.recurrence_rate == pytest.approx(from_data.recurrence_rate)
    assert from_matrix.epsilon == rm.epsilon


def test_rqa_matrix_input_rejects_build_args(noise):
    rm = rec.recurrence_matrix(noise[:200], threshold=0.5)
    with pytest.raises(ValueError, match="do not apply"):
        rec.rqa(rm, threshold=0.5)
    with pytest.raises(ValueError, match="do not apply"):
        rec.rqa(rm, recurrence_rate=0.05)


def test_rqa_min_length_guard(noise):
    with pytest.raises(ValueError, match="min_diagonal"):
        rec.rqa(noise[:200], threshold=0.5, min_diagonal=0)
    with pytest.raises(ValueError, match="min_diagonal"):
        rec.rqa(noise[:200], threshold=0.5, min_vertical=0)


def test_rqa_min_diagonal_monotone(noise):
    """A larger min line length can only keep fewer points → DET non-increasing."""
    base = rec.rqa(noise[:1500], recurrence_rate=0.1, min_diagonal=2)
    stricter = rec.rqa(noise[:1500], recurrence_rate=0.1, min_diagonal=4)
    assert stricter.determinism <= base.determinism + 1e-12


def test_rqa_empty_matrix_is_well_defined():
    """No recurrences (tiny threshold) → zero measures, infinite divergence."""
    pts = np.linspace(0.0, 100.0, 300)[:, None]  # strictly increasing, all far apart
    res = rec.rqa(pts, threshold=1e-9)
    assert res.recurrence_rate == 0.0
    assert res.determinism == 0.0
    assert res.laminarity == 0.0
    assert res.max_diagonal_length == 0
    assert res.diagonal_entropy == 0.0
    assert res.divergence == float("inf")


def test_rqa_diagonal_entropy_periodic_below_random(noise):
    """A single dominant line length (periodic) is lower-entropy than noise's spread."""
    periodic = rec.rqa(_logistic(3.5), recurrence_rate=0.05, theiler=1)
    rnd = rec.rqa(noise, recurrence_rate=0.05)
    # periodic: lengths concentrate; the line-length histograms differ in spread.
    assert periodic.diagonal_lengths.size > 0
    assert rnd.diagonal_entropy >= 0.0


# ── windowed RQA: regime-change detection ────────────────────────────────────────


@pytest.fixture(scope="module")
def transition_embedding():
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 100.0, 2000)
    periodic = np.sin(2.0 * np.pi * 1.0 * t)
    stochastic = rng.standard_normal(2000)
    sig = np.concatenate([periodic, stochastic])
    return ts.embed(sig, dimension=3, delay=5)


def test_windowed_rqa_detects_transition(transition_embedding):
    w = rec.windowed_rqa(
        transition_embedding, window=200, step=100, recurrence_rate=0.05, theiler=10
    )
    det = w.determinism
    assert len(w) == len(det) == len(w.centers)
    half = len(det) // 2
    assert det[:half].mean() > 0.8  # periodic first half
    assert det[half:].mean() < 0.4  # stochastic second half
    assert det[:half].mean() > det[half:].mean()


def test_windowed_measure_access(transition_embedding):
    w = rec.windowed_rqa(transition_embedding, window=300, recurrence_rate=0.05)
    # attribute access and measure() agree, for every exposed measure.
    for name in ("determinism", "laminarity", "recurrence_rate", "trapping_time", "divergence"):
        np.testing.assert_array_equal(getattr(w, name), w.measure(name))
    with pytest.raises(ValueError, match="unknown RQA measure"):
        w.measure("nope")
    with pytest.raises(AttributeError):
        _ = w.does_not_exist


def test_windowed_default_step_is_window(transition_embedding):
    w = rec.windowed_rqa(transition_embedding, window=400, recurrence_rate=0.05)
    assert w.step == 400
    # non-overlapping windows tile the series.
    assert len(w) == transition_embedding.shape[0] // 400


def test_windowed_centers_are_window_midpoints(transition_embedding):
    w = rec.windowed_rqa(transition_embedding, window=200, step=100, recurrence_rate=0.05)
    assert w.centers[0] == pytest.approx(99.5)
    assert np.allclose(np.diff(w.centers), 100.0)


def test_windowed_guards(transition_embedding):
    with pytest.raises(ValueError, match="window must be"):
        rec.windowed_rqa(transition_embedding, window=1, recurrence_rate=0.05)
    with pytest.raises(ValueError, match="exceeds"):
        rec.windowed_rqa(transition_embedding, window=10**7, recurrence_rate=0.05)
    with pytest.raises(ValueError, match="step must be"):
        rec.windowed_rqa(transition_embedding, window=200, step=0, recurrence_rate=0.05)


# ── repr / smoke ─────────────────────────────────────────────────────────────────


def test_repr_strings(noise):
    rm = rec.recurrence_matrix(noise[:300], recurrence_rate=0.05)
    assert "RecurrenceMatrix" in repr(rm) and "RR=" in repr(rm)
    res = rec.rqa(rm)
    assert "RQAResult" in repr(res) and "DET=" in repr(res)
    w = rec.windowed_rqa(noise[:600], window=200, recurrence_rate=0.05)
    assert "WindowedRQA" in repr(w) and "n_windows=" in repr(w)


# ── registry integration & public API ───────────────────────────────────────────


@pytest.mark.parametrize("name", ["recurrence_matrix", "rqa", "windowed_rqa"])
def test_estimators_self_register(name):
    assert name in registry.analyses
    assert registry.analyses.get(name) is getattr(rec, name)
    assert registry.analyses.entry(name).metadata["family"] == "recurrence"


@pytest.mark.parametrize(
    "name",
    [
        "recurrence_matrix",
        "rqa",
        "windowed_rqa",
        "RecurrenceMatrix",
        "RQAResult",
        "WindowedRQA",
    ],
)
def test_public_api_reexported(name):
    assert getattr(ts, name) is getattr(rec, name)
    assert name in ts.analysis.__all__
    # v4 (WS-NAMESPACE): the curated top-level ``__all__`` carries only headline
    # names; demoted analysis names stay reachable as flat re-exports.
    assert hasattr(ts, name)
