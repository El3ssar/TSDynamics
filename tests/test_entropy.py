"""Entropy & complexity quantifiers (stream A-ENT).

Validation strategy:

* **LZ76** against the canonical Kaspar–Schuster value (``c = 6`` for
  ``0001101001000101``) and analytic edge cases; factor boundaries cross-checked
  for self-consistency on thousands of random strings.
* **Sample / approximate entropy** against an independent brute-force reference
  implementation (two implementations must agree to machine precision).
* **Permutation entropy** against a hand-computed Bandt–Pompe example plus
  analytic limits (ordered → 0, white noise → 1, monotone-transform invariance).
* The composable :func:`entropy` is checked to reproduce the named wrappers, and
  the information measures are checked on explicit probability vectors.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis.entropy import (
    AddConstant,
    Dispersion,
    OrdinalPatterns,
    Renyi,
    Shannon,
    Tsallis,
    UniqueValues,
)

# The composable function ``entropy`` shadows the subpackage attribute (the same
# griffe-safe pattern as ``orbits.orbit_diagram``), so reach the module via
# sys.modules rather than attribute access.
E = importlib.import_module("tsdynamics.analysis.entropy")

# canonical Kaspar–Schuster example and its known exhaustive-history parse
_KS = "0001101001000101"
_KS_FACTORS = [0, 1, 4, 6, 9, 13]


# ---------------------------------------------------------------------------
# LZ76 complexity
# ---------------------------------------------------------------------------
class TestLZ76:
    def test_kaspar_schuster_canonical(self):
        assert ts.lz76_complexity(_KS, symbolize=None) == 6.0

    def test_factors_match_known_parse(self):
        assert E.lz76_factors(_KS, symbolize=None) == _KS_FACTORS

    @pytest.mark.parametrize(
        ("s", "expected"),
        [
            ("", 0.0),
            ("a", 1.0),
            ("aaaaaaaa", 2.0),  # constant → first symbol + the rest
            ("abcdef", 6.0),  # all distinct → n
            ("0101010101", 3.0),  # 0 | 1 | 01010101
        ],
    )
    def test_analytic_cases(self, s, expected):
        assert ts.lz76_complexity(s, symbolize=None) == expected

    def test_complexity_equals_factor_count(self):
        rng = np.random.default_rng(7)
        for _ in range(500):
            n = int(rng.integers(1, 50))
            s = "".join(rng.choice(list("01"), size=n))
            c = ts.lz76_complexity(s, symbolize=None)
            f = E.lz76_factors(s, symbolize=None)
            assert len(f) == c
            assert f == sorted(f) and len(set(f)) == len(f)
            assert not f or (f[0] == 0 and f[-1] < n)

    def test_relabeling_invariance(self):
        # LZ76 depends only on the equality structure of symbols, not their names.
        s = "0010110100"
        relabeled = s.replace("0", "X").replace("1", "Y")
        assert ts.lz76_complexity(s, symbolize=None) == ts.lz76_complexity(
            relabeled, symbolize=None
        )

    def test_entropy_is_normalized_complexity(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(2000)
        h = ts.lz76_entropy(x)
        c_norm = ts.lz76_complexity(x, normalize=True)
        assert h == pytest.approx(c_norm)

    def test_binarize_median_threshold(self):
        x = np.array([1.0, 5.0, 2.0, 8.0, 3.0])  # median = 3
        b = E.binarize(x, "median")
        assert b.tolist() == [0, 1, 0, 1, 0]  # strictly greater than the median

    def test_auto_symbolize_passes_integers_through(self):
        codes = np.array([0, 1, 0, 1, 1, 0])
        assert ts.lz76_complexity(codes, symbolize=None) == ts.lz76_complexity(
            codes, symbolize="auto"
        )

    def test_float_requires_symbolization(self):
        with pytest.raises(ValueError, match="symbolise"):
            ts.lz76_complexity(np.array([0.1, 0.2, 0.3]), symbolize=None)

    def test_random_more_complex_than_periodic(self):
        rng = np.random.default_rng(0)
        periodic = np.tile([0, 1, 0, 1], 500)
        random = rng.integers(0, 2, size=2000)
        assert ts.lz76_complexity(random, symbolize=None) > ts.lz76_complexity(
            periodic, symbolize=None
        )


class TestLZ76Provider:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            ts.lz76_complexity("0101", symbolize=None, provider="nope")

    def test_lzcomplexity_bridge_matches_native(self):
        lz = pytest.importorskip("lzcomplexity")  # noqa: F841
        rng = np.random.default_rng(11)
        for _ in range(20):
            s = "".join(rng.choice(list("012"), size=int(rng.integers(2, 60))))
            assert ts.lz76_complexity(s, symbolize=None, provider="lzcomplexity") == (
                ts.lz76_complexity(s, symbolize=None, provider="native")
            )


# ---------------------------------------------------------------------------
# Permutation entropy
# ---------------------------------------------------------------------------
class TestPermutationEntropy:
    def test_monotone_increasing_is_zero(self):
        assert ts.permutation_entropy(np.arange(1000)) == pytest.approx(0.0)

    def test_monotone_decreasing_is_zero(self):
        assert ts.permutation_entropy(np.arange(1000)[::-1]) == pytest.approx(0.0)

    def test_white_noise_approaches_one(self):
        rng = np.random.default_rng(0)
        assert ts.permutation_entropy(rng.random(20000), dimension=4) > 0.99

    def test_bandt_pompe_hand_example(self):
        # Bandt & Pompe (2002): x = (4,7,9,10,6,11,3), m=3, tau=1
        # ordinal patterns → {(0,1,2):2, (2,0,1):2, (1,0,2):1}
        x = np.array([4, 7, 9, 10, 6, 11, 3], dtype=float)
        h = ts.permutation_entropy(x, dimension=3, delay=1, normalize=False)
        assert h == pytest.approx(1.521928, abs=1e-5)
        hn = ts.permutation_entropy(x, dimension=3, delay=1, normalize=True)
        assert hn == pytest.approx(1.521928 / np.log2(6), abs=1e-5)

    def test_invariant_under_monotone_transform(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(3000)
        pe_x = ts.permutation_entropy(x, dimension=4)
        pe_fx = ts.permutation_entropy(np.exp(x), dimension=4)  # strictly increasing map
        assert pe_x == pytest.approx(pe_fx, abs=1e-12)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            ts.permutation_entropy(np.arange(2), dimension=4)

    def test_weighted_monotone_is_zero(self):
        assert ts.weighted_permutation_entropy(np.arange(1000)) == pytest.approx(0.0)

    def test_weighted_constant_is_zero(self):
        assert ts.weighted_permutation_entropy(np.ones(500)) == 0.0

    def test_composable_matches_named(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(2000)
        composed = ts.entropy(x, outcomes=OrdinalPatterns(3, 1), measure=Shannon(2), normalize=True)
        assert composed == pytest.approx(ts.permutation_entropy(x, 3, 1))


# ---------------------------------------------------------------------------
# Dispersion entropy
# ---------------------------------------------------------------------------
class TestDispersionEntropy:
    def test_constant_is_zero(self):
        assert ts.dispersion_entropy(np.ones(500)) == pytest.approx(0.0)

    def test_white_noise_high(self):
        rng = np.random.default_rng(0)
        assert ts.dispersion_entropy(rng.standard_normal(20000), c=6, dimension=2) > 0.9

    def test_noise_exceeds_sine(self):
        rng = np.random.default_rng(1)
        t = np.linspace(0, 100, 5000)
        sine = np.sin(t)
        noise = rng.standard_normal(5000)
        assert ts.dispersion_entropy(noise) > ts.dispersion_entropy(sine)

    def test_composable_matches_named(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(3000)
        composed = ts.entropy(x, outcomes=Dispersion(6, 2, 1), measure=Shannon(2), normalize=True)
        assert composed == pytest.approx(ts.dispersion_entropy(x, 6, 2, 1))

    def test_cardinality(self):
        assert Dispersion(6, 2).cardinality == 36
        assert Dispersion(4, 3).cardinality == 64

    def test_classes_are_equal_width_partition(self):
        # Rostaghi & Azami: class k ⟺ c·NCDF(x) ∈ [k-1, k); the mapping is
        # monotone, bounded in [1, c], and balanced for a normal sample.
        from scipy.stats import norm

        rng = np.random.default_rng(3)
        x = rng.standard_normal(60000)
        z = Dispersion(c=6)._classes(x)
        assert z.min() >= 1 and z.max() <= 6
        # monotone in the standardised value
        order = np.argsort(x)
        assert np.all(np.diff(z[order]) >= 0)
        # explicit equal-width formula reproduced
        expected = np.clip(np.floor(6 * norm.cdf((x - x.mean()) / x.std())).astype(int) + 1, 1, 6)
        assert np.array_equal(z, expected)
        # the inner classes are well populated (no systematic half-class shift)
        counts = np.bincount(z, minlength=7)[1:]
        assert counts[1:-1].min() > 0.10 * x.size

    def test_reference_pipeline(self):
        # Independent re-derivation of the dispersion-pattern entropy pipeline.
        from scipy.stats import norm

        rng = np.random.default_rng(9)
        x = rng.standard_normal(2000)
        c, m = 5, 2
        y = norm.cdf((x - x.mean()) / x.std())
        z = np.clip(np.floor(c * y).astype(int) + 1, 1, c) - 1  # 0-based digits
        labels = z[:-1] * c + z[1:]  # mixed-radix, m=2
        p = np.bincount(labels, minlength=c**m)
        p = p[p > 0] / p.sum()
        h_ref = -np.sum(p * np.log2(p)) / np.log2(c**m)
        assert ts.dispersion_entropy(x, c=c, dimension=m) == pytest.approx(h_ref)


# ---------------------------------------------------------------------------
# Sample & approximate entropy (vs brute-force reference)
# ---------------------------------------------------------------------------
def _ref_sampen(x, m, r):
    x = np.asarray(x, float)
    n = x.size
    nt = n - m

    def count(mm):
        c = 0
        for i in range(nt):
            for j in range(nt):
                if i == j:
                    continue
                if np.max(np.abs(x[i : i + mm] - x[j : j + mm])) <= r:
                    c += 1
        return c

    b, a = count(m), count(m + 1)
    return -np.log(a / b)


def _ref_apen(x, m, r):
    x = np.asarray(x, float)
    n = x.size

    def phi(mm):
        nt = n - mm + 1
        s = 0.0
        for i in range(nt):
            c = sum(1 for j in range(nt) if np.max(np.abs(x[i : i + mm] - x[j : j + mm])) <= r)
            s += np.log(c / nt)
        return s / nt

    return phi(m) - phi(m + 1)


class TestSampleEntropy:
    def test_matches_bruteforce_reference(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(80)
        r = 0.2 * x.std()
        assert ts.sample_entropy(x, dimension=2, r=r) == pytest.approx(
            _ref_sampen(x, 2, r), abs=1e-12
        )

    def test_default_r_is_explicit_fraction(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(80)
        assert ts.sample_entropy(x, dimension=2) == pytest.approx(
            _ref_sampen(x, 2, 0.2 * x.std()), abs=1e-12
        )

    def test_constant_series_is_zero(self):
        assert ts.sample_entropy(np.ones(100)) == pytest.approx(0.0)

    def test_noise_exceeds_sine(self):
        rng = np.random.default_rng(1)
        t = np.linspace(0, 60, 1500)
        assert ts.sample_entropy(rng.standard_normal(1500)) > ts.sample_entropy(np.sin(t))

    def test_no_m_match_raises(self):
        # strictly increasing → every length-2 template is unique; r too small
        with pytest.raises(ValueError, match="no length-m"):
            ts.sample_entropy(np.arange(7.0), dimension=2, r=1e-9)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            ts.sample_entropy(np.array([1.0, 2.0]), dimension=2)


class TestApproximateEntropy:
    def test_matches_bruteforce_reference(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal(80)
        r = 0.2 * x.std()
        assert ts.approximate_entropy(x, dimension=2, r=r) == pytest.approx(
            _ref_apen(x, 2, r), abs=1e-12
        )

    def test_noise_exceeds_sine(self):
        rng = np.random.default_rng(1)
        t = np.linspace(0, 60, 1500)
        assert ts.approximate_entropy(rng.standard_normal(1500)) > ts.approximate_entropy(np.sin(t))


# ---------------------------------------------------------------------------
# Multiscale
# ---------------------------------------------------------------------------
class TestMultiscale:
    def test_coarse_grain_hand_example(self):
        assert E.coarse_grain([1, 2, 3, 4, 5, 6], 2).tolist() == [1.5, 3.5, 5.5]
        # a partial trailing window is dropped
        assert E.coarse_grain([1, 2, 3, 4, 5], 2).tolist() == [1.5, 3.5]

    def test_coarse_grain_scale_one_is_identity(self):
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert np.array_equal(E.coarse_grain(x, 1), x)

    def test_white_noise_decays_with_scale(self):
        rng = np.random.default_rng(0)
        mse = ts.multiscale_entropy(rng.standard_normal(6000), scales=6)
        assert mse.size == 6
        assert mse[0] > mse[-1]

    def test_explicit_scale_iterable(self):
        rng = np.random.default_rng(0)
        mse = ts.multiscale_entropy(rng.standard_normal(4000), scales=[1, 2, 4])
        assert mse.size == 3

    def test_generic_over_entropy_fn(self):
        rng = np.random.default_rng(0)
        # permutation entropy takes no r — multiscale must not inject one
        mpe = ts.multiscale_entropy(
            rng.standard_normal(4000), scales=3, entropy_fn=ts.permutation_entropy, dimension=3
        )
        assert mpe.size == 3 and np.all(np.isfinite(mpe))

    def test_fixed_r_across_scales(self):
        # The tolerance is taken once from the original series; passing it
        # explicitly must reproduce the default behaviour.
        rng = np.random.default_rng(2)
        x = rng.standard_normal(3000)
        auto = ts.multiscale_entropy(x, scales=4)
        fixed = ts.multiscale_entropy(x, scales=4, r=0.15 * x.std())
        assert np.allclose(auto, fixed)


# ---------------------------------------------------------------------------
# Composable core: measures and estimators
# ---------------------------------------------------------------------------
class TestInformationMeasures:
    def test_shannon_uniform(self):
        assert Shannon(2).apply(np.array([0.5, 0.5])) == pytest.approx(1.0)
        assert Shannon(2).apply(np.array([1.0])) == pytest.approx(0.0)
        assert Shannon(2).maximum(8) == pytest.approx(3.0)

    def test_shannon_base_change(self):
        p = np.array([0.2, 0.3, 0.5])
        assert Shannon(np.e).apply(p) == pytest.approx(Shannon(2).apply(p) * np.log(2))

    def test_renyi_hartley_and_collision(self):
        p = np.array([0.5, 0.3, 0.2])
        assert Renyi(0, 2).apply(p) == pytest.approx(np.log2(3))  # Hartley (log support size)
        assert Renyi(2, 2).apply(np.array([0.5, 0.5])) == pytest.approx(1.0)

    def test_renyi_limit_is_shannon(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        assert Renyi(1.0, 2).apply(p) == pytest.approx(Shannon(2).apply(p))

    def test_tsallis(self):
        assert Tsallis(2).apply(np.array([0.5, 0.5])) == pytest.approx(0.5)
        p = np.array([0.1, 0.2, 0.3, 0.4])
        assert Tsallis(1.0).apply(p) == pytest.approx(Shannon(np.e).apply(p))


class TestEstimators:
    def test_add_constant_is_a_distribution(self):
        counts = np.array([3, 0, 1, 0])
        p = AddConstant(1.0).probabilities(counts)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(p > 0)  # smoothing removes zeros
        # Laplace: (n_i + 1) / (N + K)
        assert p[0] == pytest.approx(4 / 8)

    def test_mle_empty_raises(self):
        with pytest.raises(ValueError, match="no outcomes"):
            E.MLE().probabilities(np.zeros(4, dtype=int))

    def test_unique_values_outcome_space(self):
        space = UniqueValues()
        counts = space.counts(np.array([0, 0, 1, 2, 2, 2]))
        assert sorted(counts.tolist()) == [1, 2, 3]
        assert space.cardinality == 3

    def test_normalized_entropy_unit_for_uniform(self):
        # A series whose ordinal patterns are exactly uniform → normalized = 1.
        # Construct counts directly via AmplitudeBinning on a uniform grid.
        space = E.AmplitudeBinning(4)
        x = np.tile([0.1, 1.1, 2.1, 3.1], 50)  # each bin equally often
        h = ts.entropy(x, outcomes=space, normalize=True)
        assert h == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input coercion / Trajectory interop
# ---------------------------------------------------------------------------
class TestInput:
    def test_two_d_requires_component(self):
        with pytest.raises(ValueError, match="component"):
            ts.permutation_entropy(np.random.default_rng(0).random((500, 3)))

    def test_two_d_component_selects_column(self):
        rng = np.random.default_rng(0)
        data = rng.random((2000, 3))
        assert ts.permutation_entropy(data, component=1) == pytest.approx(
            ts.permutation_entropy(data[:, 1])
        )

    def test_trajectory_named_component(self):
        # A Trajectory carrying variable names resolves component by name.
        lor = ts.Lorenz()
        t = np.linspace(0, 10, 3000)
        y = np.random.default_rng(0).standard_normal((3000, 3))
        traj = ts.Trajectory(t, y, lor)
        by_name = ts.permutation_entropy(traj, component="y")
        by_index = ts.permutation_entropy(y[:, 1])
        assert by_name == pytest.approx(by_index)


# ---------------------------------------------------------------------------
# Registry self-registration
# ---------------------------------------------------------------------------
def test_entropy_analyses_registered():
    from tsdynamics import registry

    for name in ("permutation_entropy", "sample_entropy", "lz76_complexity", "entropy"):
        assert name in registry.analyses
        assert registry.analyses.get(name) is getattr(ts, name)
