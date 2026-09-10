"""Tests for the object-side topical accessor layer (stream WS-ACCESSORS).

The accessors are a *purely additive* convenience surface: each one delegates to
the canonical free function in :mod:`tsdynamics.analysis` (or constructs the
matching derived wrapper) with the system bound, adding **zero behaviour**.  The
tests below assert exactly that contract:

* the topical namespaces (``lyap`` / ``dims`` / ``recurrence`` / ``chaos``)
  and the first-class verbs (``fixed_points`` /
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

import inspect

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
from tsdynamics.families._accessors import (
    ACCESSOR_DELEGATIONS,
    ChaosAccessor,
    DimensionsAccessor,
    LyapunovAccessor,
    RecurrenceAccessor,
    subject_kind,
)
from tsdynamics.systems import Henon, Lorenz, Rossler

# The accessors delegate to engine-backed analyses (they run / iterate the
# system), so this module needs the compiled extension; the import both gates
# the module and auto-tags it ``engine`` (see tests/_engine_marker.py).
pytest.importorskip("tsdynamics._rust")


TOPICAL = ("lyap", "dims", "recurrence", "chaos")
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
    ``region`` and the kwargs unchanged.  The region is one ``(lo, hi)`` bound
    per state component, the one reading every region door in the library uses.
    """
    h = Henon()
    region = [(-1.5, 1.5), (-0.4, 0.4)]  # one (lo, hi) bound per state component
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


def test_copies_builds_ensemble_system():
    """``sys.copies(states)`` builds an ``EnsembleSystem`` (the LAZY wrapper)."""
    states = np.random.default_rng(0).random((5, 3))
    assert isinstance(Lorenz().copies(states), EnsembleSystem)


def test_ensemble_runs_the_batch_on_every_family():
    """``.ensemble`` means one thing everywhere: run the batch → ``(n, dim)`` finals."""
    ics = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.001]])
    finals = Lorenz().ensemble(ics, final_time=1.0, dt=0.05)
    assert isinstance(finals, np.ndarray)
    assert finals.shape == (2, 3)

    hen = ts.systems.Henon()
    map_finals = hen.ensemble(np.array([[0.1, 0.1], [0.2, 0.2]]), steps=50)
    assert isinstance(map_finals, np.ndarray)
    assert map_finals.shape == (2, 2)

    ou = ts.systems.OrnsteinUhlenbeck()
    sde_finals = ou.ensemble(np.zeros((3, 1)), final_time=1.0, dt=0.01, seed=0)
    assert isinstance(sde_finals, np.ndarray)
    assert sde_finals.shape == (3, 1)


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


# --------------------------------------------------------------------------- #
# the delegation contract (stream v6 api-core)
# --------------------------------------------------------------------------- #
#
# ``sys.chaos.zero_one()`` used to pre-run the system with the *family's*
# ``run()`` defaults (dt = 0.01) and hand the resulting oversampled trajectory to
# the SYSTEM-first ``zero_one_test`` as if it were measured data.  Successive
# samples were then heavily correlated and the 0-1 test collapsed: Lorenz —
# unambiguously chaotic — measured K = -0.026 through the accessor against
# K = 0.999 through the free function.  Plausible number, qualitatively wrong
# answer, no warning.
#
# The tests below close the whole class of bug rather than that one instance:
# every accessor method is checked *programmatically* against the signature of
# the free function it delegates to.


def test_every_accessor_method_is_declared():
    """Every public accessor method appears in ``ACCESSOR_DELEGATIONS``.

    The delegation table is what the agreement tests below iterate, so a new
    accessor method must be registered (and thereby checked) to pass.
    """
    for cls in (LyapunovAccessor, ChaosAccessor, DimensionsAccessor, RecurrenceAccessor):
        declared = set(ACCESSOR_DELEGATIONS[cls.__name__])
        public = {
            name for name, obj in vars(cls).items() if callable(obj) and not name.startswith("_")
        }
        assert public == declared, f"{cls.__name__}: {public ^ declared} undeclared/stale"


def test_declared_free_functions_exist_and_are_the_public_ones():
    """Each declared target resolves to the same object the top level exports."""
    for methods in ACCESSOR_DELEGATIONS.values():
        for free_name in methods.values():
            free = getattr(ts.analysis, free_name)
            assert callable(free)
            assert getattr(ts, free_name) is free


@pytest.mark.parametrize(
    ("accessor", "method", "free_name"),
    [
        (acc, meth, free)
        for acc, methods in ACCESSOR_DELEGATIONS.items()
        for meth, free in methods.items()
    ],
)
def test_accessor_first_argument_matches_the_free_function(accessor, method, free_name):
    """The accessor hands the free function the subject *its signature asks for*.

    A ``system``-first free function drives the system itself (with the horizon
    defaults its own method needs); a ``data``-first one consumes a measured
    point set.  The accessor's routing is read off the signature, so it can never
    disagree — this asserts the two agree for every registered pair.
    """
    free = getattr(ts.analysis, free_name)
    kind = subject_kind(free)
    first = next(iter(inspect.signature(free).parameters))
    assert kind == ("system" if first == "system" else "data")

    # A ``system``-first accessor method must NOT accept ``run_kwargs``: pre-running
    # the system is exactly the mistake that produced the zero_one defect.
    acc_cls = {
        c.__name__: c
        for c in (LyapunovAccessor, ChaosAccessor, DimensionsAccessor, RecurrenceAccessor)
    }[accessor]
    acc_params = inspect.signature(getattr(acc_cls, method)).parameters
    if kind == "system":
        assert "run_kwargs" not in acc_params, (
            f"{accessor}.{method} takes run_kwargs but {free_name} drives the system itself"
        )


def test_chaos_zero_one_agrees_with_the_free_function_on_a_flow():
    """``lor.chaos.zero_one()`` == ``ts.zero_one_test(lor)`` — the regression.

    Independent truth: the Lorenz attractor at the standard parameters is chaotic,
    so the Gottwald-Melbourne indicator must land near 1.  Before the fix the
    accessor returned K = -0.026 ("regular") while the free function returned
    K = 0.999 ("chaotic").
    """
    lor = Lorenz(ic=[1.0, 1.0, 1.0])
    free = ts.zero_one_test(lor, component=0)
    acc = lor.chaos.zero_one(component=0)
    assert float(acc) == float(free)
    assert float(acc) > 0.9  # chaotic, the literature answer


def test_chaos_zero_one_agrees_with_the_free_function_on_a_map():
    """The same agreement on a discrete map, both branches of the K threshold."""
    chaotic = ts.systems.Logistic(params={"r": 4.0}, ic=[0.4])
    periodic = ts.systems.Logistic(params={"r": 3.2}, ic=[0.4])
    for sys_, expect_chaos in ((chaotic, True), (periodic, False)):
        free = float(ts.zero_one_test(sys_, n=3000, component=0))
        acc = float(sys_.chaos.zero_one(n=3000, component=0))
        assert acc == free
        assert (acc > 0.5) is expect_chaos


def test_chaos_zero_one_rejects_run_kwargs():
    """``run_kwargs`` is gone: the free function owns the sampling grid.

    Pre-running the system is exactly the mistake that produced the wrong K, so
    asking for it is refused loudly rather than silently ignored.
    """
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError, match="drives the system itself"):
        Lorenz().chaos.zero_one(run_kwargs={"final_time": 200.0})


def test_data_first_accessors_still_accept_a_measured_series():
    """The ``data``-first accessors keep delegating a supplied series verbatim."""
    data = _henon_data()
    h = Henon()
    assert float(h.dims.correlation(data)) == float(ts.correlation_dimension(data))
    assert h.recurrence.rqa(data, recurrence_rate=0.05).determinism == (
        ts.rqa(data, recurrence_rate=0.05).determinism
    )


# --------------------------------------------------------------------------- #
# The delegation contract, checked BEHAVIOURALLY (v6 api-core follow-up)
# --------------------------------------------------------------------------- #
#
# ``test_accessor_first_argument_matches_the_free_function`` above asserts
# ``subject_kind(free) == ("system" if first == "system" else "data")`` — which is
# the definition of ``subject_kind``, i.e. a tautology that would still pass if
# ``_delegate`` handed every free function the wrong subject.  The test below
# closes the loop for real: it replaces each free function with a recorder and
# asserts what the accessor ACTUALLY passed as the first argument.


def _record_call(monkeypatch, free_name):
    """Patch ``ts.analysis.<free_name>`` with a recorder and return the record.

    The recorder borrows the real function's ``__signature__`` because
    ``_delegate`` routes on it (:func:`subject_kind` reads the first parameter's
    name) — a stub with a differently named first argument would silently be
    treated as ``data``-first and the test would prove nothing.
    """
    seen: dict[str, object] = {}
    real_sig = inspect.signature(getattr(ts.analysis, free_name))

    def _recorder(subject, *args, **kwargs):
        seen["subject"] = subject
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "sentinel"

    _recorder.__signature__ = real_sig  # type: ignore[attr-defined]
    monkeypatch.setattr(ts.analysis, free_name, _recorder, raising=True)
    return seen


@pytest.mark.parametrize(
    ("accessor", "method", "free_name"),
    [
        (acc, meth, free)
        for acc, methods in ACCESSOR_DELEGATIONS.items()
        for meth, free in methods.items()
    ],
)
def test_accessor_actually_passes_the_declared_subject(monkeypatch, accessor, method, free_name):
    """Every accessor hands its free function the subject that function asks for.

    ``system``-first free functions must receive the *bound system itself* (they
    drive the integration with the horizon their own method needs — pre-running
    the system is precisely the mistake that made ``chaos.zero_one`` call Lorenz
    regular).  ``data``-first ones must receive the measured point set: verbatim
    when the caller supplies one, and a freshly-run Trajectory otherwise.
    """
    system = Henon(ic=[0.1, 0.1])
    bound = getattr(
        {
            "lyap": system.lyap,
            "chaos": system.chaos,
            "dims": system.dims,
            "recurrence": system.recurrence,
        }[
            {
                "LyapunovAccessor": "lyap",
                "ChaosAccessor": "chaos",
                "DimensionsAccessor": "dims",
                "RecurrenceAccessor": "recurrence",
            }[accessor]
        ],
        method,
    )
    # The expectation is read from the ANALYSIS layer's own signature, not from
    # ``subject_kind`` — otherwise a broken ``subject_kind`` would move the
    # expectation with the behaviour and the test would pass through the bug.
    system_first = next(iter(inspect.signature(getattr(ts.analysis, free_name)).parameters)) == (
        "system"
    )

    # 1. no data supplied
    seen = _record_call(monkeypatch, free_name)
    assert bound() == "sentinel"
    if system_first:
        assert seen["subject"] is system, f"{accessor}.{method} did not pass the system"
    else:
        assert isinstance(seen["subject"], (np.ndarray, ts.Trajectory)), (
            f"{accessor}.{method} passed {type(seen['subject'])!r}, not a point set"
        )

    # 2. an accessor that exposes ``data`` forwards it verbatim, whatever the kind
    #    (``lyap.spectrum`` / ``lyap.maximal`` take none; ``chaos.gali`` /
    #    ``chaos.expansion_entropy`` spend their first positional slot on ``k`` /
    #    ``region``, so there is nothing to forward there).
    acc_params = list(inspect.signature(bound).parameters)
    if acc_params and acc_params[0] == "data":
        data = np.asarray(_henon_data(), dtype=float)
        seen = _record_call(monkeypatch, free_name)
        assert bound(data) == "sentinel"
        assert seen["subject"] is data, f"{accessor}.{method} did not forward the supplied data"


# --------------------------------------------------------------------------- #
# the data-consuming accessors, bound to a Trajectory
# --------------------------------------------------------------------------- #
#
# ``dims`` / ``recurrence`` / ``lyap.from_data`` consume a *measured point set*.
# They used to be reachable only from a SYSTEM — that is, the discoverable path
# existed only on the object half of them refuse, and the user actually holding a
# trajectory (the normal case) was pushed onto the flat functions.  The same
# accessor classes are now bound to a ``Trajectory`` as well.

TRAJECTORY_TOPICAL = ("dims", "recurrence", "lyap")


@pytest.fixture(scope="module")
def _lorenz_traj():
    """A short Lorenz run — the measured series the accessors below consume."""
    return Lorenz().run(final_time=60.0, dt=0.02, ic=[1.0, 1.0, 1.0])


@pytest.mark.parametrize("name", TRAJECTORY_TOPICAL)
def test_trajectory_carries_the_data_consuming_accessors(name, _lorenz_traj):
    """``traj.dims`` / ``traj.recurrence`` / ``traj.lyap`` exist and are cached."""
    acc = getattr(_lorenz_traj, name)
    assert acc is getattr(_lorenz_traj, name), f"traj.{name} is not cached"
    assert acc._system is _lorenz_traj


def test_trajectory_does_not_grow_the_system_only_accessor():
    """``chaos`` is system-first throughout, so it is NOT hung on a trajectory."""
    traj = Lorenz().run(final_time=5.0, dt=0.05, ic=[1.0, 1.0, 1.0])
    assert not hasattr(traj, "chaos")


def test_trajectory_dims_is_identical_to_the_free_function(_lorenz_traj):
    """The accessor adds zero behaviour: same series in, same number out."""
    radii = np.logspace(-0.5, 0.8, 10)
    assert _lorenz_traj.dims.correlation(radii=radii) == ts.analysis.correlation_dimension(
        _lorenz_traj, radii=radii
    )


def test_trajectory_recurrence_is_identical_to_the_free_function(_lorenz_traj):
    """Same for the recurrence accessor."""
    short = _lorenz_traj[:400]
    assert short.recurrence.rqa(recurrence_rate=0.05).determinism == (
        ts.analysis.rqa(short, recurrence_rate=0.05).determinism
    )


def test_trajectory_lyap_from_data_is_identical_to_the_free_function(_lorenz_traj):
    """``lyap.from_data`` is the one Lyapunov estimator a bare series supports."""
    # Decimated: the raw dt=0.02 series is oversampled for this estimator and
    # warns (the suite runs under ``filterwarnings = error``).
    thin = _lorenz_traj[::4]
    a = thin.lyap.from_data(dimension=3, delay=5, k_max=60)
    b = ts.analysis.lyapunov_from_data(thin, dimension=3, delay=5, k_max=60)
    assert np.allclose(np.asarray(a.ordinate), np.asarray(b.ordinate))
    assert float(a.estimate) == float(b.estimate)


@pytest.mark.parametrize("method", ["spectrum", "maximal"])
def test_a_system_first_method_on_a_trajectory_raises_and_names_the_fix(method, _lorenz_traj):
    """A trajectory has no right-hand side; the error says so and names the spelling."""
    with pytest.raises(ts.errors.InvalidParameterError) as exc:
        getattr(_lorenz_traj.lyap, method)()
    msg = str(exc.value)
    assert "measured data" in msg
    assert f"system.lyap.{method}()" in msg


def test_run_kwargs_on_a_trajectory_accessor_is_refused_not_ignored(_lorenz_traj):
    """There is nothing to run, so silently dropping the window would be a lie."""
    with pytest.raises(ts.errors.InvalidParameterError, match="nothing to run"):
        _lorenz_traj.dims.correlation(run_kwargs={"final_time": 10.0})


def test_the_system_side_of_the_accessors_is_unchanged(_lorenz_traj):
    """Binding the accessors to a trajectory must not move the system behaviour."""
    lor = Lorenz()
    radii = np.logspace(-0.5, 0.8, 10)
    assert lor.dims.correlation(data=_lorenz_traj, radii=radii) == (
        _lorenz_traj.dims.correlation(radii=radii)
    )
    # ... and a system-first method still drives the system itself.
    assert isinstance(lor.lyap.spectrum(final_time=20.0, ic=[1.0, 1.0, 1.0]), ts.LyapunovSpectrum)


def test_trajectory_accessor_cache_survives_a_pickle_round_trip(_lorenz_traj):
    """``__slots__`` needs the cache slot restored explicitly (it is not pickled)."""
    import pickle

    restored = pickle.loads(pickle.dumps(_lorenz_traj))
    assert restored.dims is restored.dims
    assert restored._accessor_cache == {"dims": restored.dims}
