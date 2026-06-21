"""Tests for the object-side topical accessor layer (stream WS-ACCESSORS).

The accessors are a *purely additive* convenience surface: each one delegates to
the canonical free function in :mod:`tsdynamics.analysis` (or constructs the
matching derived wrapper) with the system bound, adding **zero behaviour**.  The
tests below assert exactly that contract:

* the topical namespaces (``lyap`` / ``dims`` / ``recurrence`` / ``chaos`` /
  ``surrogate`` / ``entropy``) and the first-class verbs (``fixed_points`` /
  ``poincare`` / ``tangent`` / ``project`` / ``ensemble`` / ``stroboscope``) are
  present and grouped (tab-completion discoverability),
* an accessor is cached on the instance (``sys.lyap is sys.lyap``),
* an accessor result is *identical* to the free-function result on the same
  input, and
* the derived-builder verbs return the correct wrapper type, byte-identical to
  the hand-built wrapper.

These run on cheap systems (Hénon / logistic for maps, Rössler / Lorenz for
flows) so they stay in the fast tier.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.derived import (
    EnsembleSystem,
    PoincareMap,
    ProjectedSystem,
    StroboscopicMap,
    TangentSystem,
)
from tsdynamics.systems import Henon, Lorenz, Rossler

# The accessors delegate to engine-backed analyses (they run / iterate the
# system), so this module needs the compiled extension; the import both gates
# the module and auto-tags it ``engine`` (see tests/_engine_marker.py).
pytest.importorskip("tsdynamics._rust")


TOPICAL = ("lyap", "dims", "recurrence", "chaos", "surrogate", "entropy")
VERBS = ("fixed_points", "poincare", "tangent", "project", "ensemble", "stroboscope")


# --------------------------------------------------------------------------- #
# discoverability / grouping
# --------------------------------------------------------------------------- #


def test_topical_accessors_present():
    """Every topical accessor namespace is reachable from a system."""
    lor = Lorenz()
    for name in TOPICAL:
        assert hasattr(lor, name), name


def test_first_class_verbs_present():
    """Every first-class verb is reachable from a system."""
    lor = Lorenz()
    for name in VERBS:
        assert hasattr(lor, name), name


def test_accessors_grouped_not_flat():
    """The analyses live grouped under accessors, not as ~60 flat methods.

    ``rqa`` is a recurrence estimator — it must be reached via ``sys.recurrence``
    (the grouping), never hung directly on the system.
    """
    lor = Lorenz()
    assert hasattr(lor.recurrence, "rqa")
    assert not hasattr(lor, "rqa")
    assert hasattr(lor.dims, "correlation")
    assert not hasattr(lor, "correlation_dimension")


# --------------------------------------------------------------------------- #
# caching identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", TOPICAL)
def test_accessor_cached(name):
    """``sys.<group> is sys.<group>`` — one accessor instance per system."""
    lor = Lorenz()
    assert getattr(lor, name) is getattr(lor, name)


def test_accessor_holds_its_system():
    """The accessor binds the very system it was reached from."""
    lor = Lorenz()
    assert lor.lyap._system is lor
    # distinct systems get distinct accessors
    other = Lorenz()
    assert lor.lyap is not other.lyap


# --------------------------------------------------------------------------- #
# delegation identity — system-bound analyses
# --------------------------------------------------------------------------- #


def test_lyap_spectrum_identical():
    """``sys.lyap.spectrum`` == ``ts.lyapunov_spectrum(sys)`` (byte-identical)."""
    h = Henon()
    viaaccessor = np.asarray(h.lyap.spectrum(n=2000, ic=[0.1, 0.1]))
    viafree = np.asarray(ts.lyapunov_spectrum(h, n=2000, ic=[0.1, 0.1]))
    assert np.array_equal(viaaccessor, viafree)


def test_lyap_maximal_identical():
    """``sys.lyap.maximal`` == ``ts.max_lyapunov(sys)``."""
    h = Henon()
    assert h.lyap.maximal(ic=[0.1, 0.1], seed=0) == ts.max_lyapunov(h, ic=[0.1, 0.1], seed=0)


def test_chaos_gali_identical():
    """``sys.chaos.gali(k)`` == ``ts.gali(sys, k)``."""
    h = Henon()
    g_acc = h.chaos.gali(2, ic=[0.1, 0.1])
    g_free = ts.gali(h, 2, ic=[0.1, 0.1])
    assert float(g_acc.final) == float(g_free.final)
    assert g_acc.k == g_free.k


def test_chaos_expansion_entropy_identical():
    """``sys.chaos.expansion_entropy`` == ``ts.expansion_entropy(sys)``.

    Run on a bounded ``region`` so the estimator's orbit-box sampler does not
    wander off the Hénon attractor — the accessor binds the system and forwards
    ``region`` and the kwargs unchanged.
    """
    h = Henon()
    region = ([-1.5, -0.4], [1.5, 0.4])  # (lo, hi) vectors bounding the attractor
    e_acc = h.chaos.expansion_entropy(region, seed=0)
    e_free = ts.expansion_entropy(h, region, seed=0)
    assert float(e_acc.entropy) == float(e_free.entropy)


def test_fixed_points_verb_identical():
    """``sys.fixed_points()`` == ``ts.fixed_points(sys)`` (Lorenz equilibria)."""
    lor = Lorenz()
    fp_acc = lor.fixed_points(seed=0)
    fp_free = ts.fixed_points(lor, seed=0)
    assert len(fp_acc) == len(fp_free)
    for a, b in zip(fp_acc, fp_free, strict=True):
        assert np.array_equal(np.asarray(a.x), np.asarray(b.x))


# --------------------------------------------------------------------------- #
# delegation identity — data-consuming analyses (explicit data)
# --------------------------------------------------------------------------- #


def _henon_data():
    return Henon().iterate(steps=3000, ic=[0.1, 0.1])


def test_dims_correlation_identical():
    """``sys.dims.correlation(data)`` == ``ts.correlation_dimension(data)``."""
    data = _henon_data()
    d_acc = Henon().dims.correlation(data)
    d_free = ts.correlation_dimension(data)
    assert float(d_acc) == float(d_free)
    assert np.array_equal(d_acc.x, d_free.x)
    assert np.array_equal(d_acc.y, d_free.y)


def test_recurrence_rqa_identical():
    """``sys.recurrence.rqa(data, ...)`` == ``ts.rqa(data, ...)``."""
    data = _henon_data()
    r_acc = Henon().recurrence.rqa(data, recurrence_rate=0.05)
    r_free = ts.rqa(data, recurrence_rate=0.05)
    assert r_acc.determinism == r_free.determinism
    assert r_acc.laminarity == r_free.laminarity


def test_entropy_permutation_identical():
    """``sys.entropy.permutation(data)`` == ``ts.permutation_entropy(data)``."""
    series = _henon_data().y[:, 0]
    assert Henon().entropy.permutation(series, 3, 1) == ts.permutation_entropy(series, 3, 1)


def test_surrogate_test_identical():
    """``sys.surrogate.test(data, ...)`` == ``ts.surrogate_test(data, ...)``."""
    series = _henon_data().y[:, 0]
    s_acc = Henon().surrogate.test(series, n=9, seed=1)
    s_free = ts.surrogate_test(series, n=9, seed=1)
    assert s_acc.p_value == s_free.p_value


def test_chaos_zero_one_identical():
    """``sys.chaos.zero_one(data)`` == ``ts.zero_one_test(data)``."""
    series = _henon_data().y[:, 0]
    assert Henon().chaos.zero_one(series, seed=0) == ts.zero_one_test(series, seed=0)


# --------------------------------------------------------------------------- #
# data accessors — implicit-run path
# --------------------------------------------------------------------------- #


def test_data_accessor_autorun_matches_manual_run():
    """Omitting ``data`` runs the system; identical to a manual run + free fn."""
    h = Henon()
    auto = h.dims.correlation(run_kwargs={"n": 3000, "ic": [0.1, 0.1]})
    manual = ts.correlation_dimension(h.run(n=3000, ic=[0.1, 0.1]))
    assert float(auto) == float(manual)


# --------------------------------------------------------------------------- #
# first-class derived-builder verbs
# --------------------------------------------------------------------------- #


def test_poincare_builds_poincare_map():
    """``sys.poincare(section=, at=)`` builds the same ``PoincareMap`` as the class."""
    ros = Rossler()
    pm = ros.poincare(section="y", at=0.0, direction=+1)
    assert isinstance(pm, PoincareMap)
    pm_hand = ts.PoincareMap(ros, (1, 0.0), direction=+1)
    assert pm.plane == pm_hand.plane
    assert pm._offset == pm_hand._offset
    assert pm.direction == pm_hand.direction


def test_poincare_explicit_plane():
    """An explicit ``plane=`` tuple bypasses the friendly spelling."""
    ros = Rossler()
    pm = ros.poincare(plane=(1, 0.5))
    assert pm.plane == (1, 0.5)


def test_poincare_requires_section_or_plane():
    """``poincare()`` with neither ``section`` nor ``plane`` raises clearly."""
    with pytest.raises(ValueError, match="section.*plane|plane"):
        Rossler().poincare()


def test_tangent_builds_tangent_system():
    """``sys.tangent(k)`` builds a ``TangentSystem``."""
    assert isinstance(Lorenz().tangent(3), TangentSystem)


def test_project_builds_projected_system():
    """``sys.project(...)`` builds a ``ProjectedSystem`` (names or a sequence)."""
    lor = Lorenz()
    p_args = lor.project("x", "z")
    p_seq = lor.project(["x", "z"])
    assert isinstance(p_args, ProjectedSystem)
    assert p_args.components == p_seq.components
    p_hand = ts.ProjectedSystem(lor, ("x", "z"))
    assert p_args.components == p_hand.components


def test_ensemble_builds_ensemble_system():
    """``sys.ensemble(states)`` builds an ``EnsembleSystem``."""
    states = np.random.default_rng(0).random((5, 3))
    assert isinstance(Lorenz().ensemble(states), EnsembleSystem)


def test_stroboscope_builds_stroboscopic_map():
    """``sys.stroboscope(period)`` builds a ``StroboscopicMap``."""
    strobe = Lorenz().stroboscope(2.0 * np.pi)
    assert isinstance(strobe, StroboscopicMap)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #


def test_accessor_repr():
    """The accessor repr names its kind and its system (helps in a notebook)."""
    assert repr(Lorenz().lyap) == "LyapunovAccessor(Lorenz)"
    assert repr(Henon().dims) == "DimensionsAccessor(Henon)"
