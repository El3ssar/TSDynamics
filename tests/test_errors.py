"""WS-ERRORS — the TSDynamicsError hierarchy + value-naming + silent-footgun gate.

These tests pin three guarantees the v4 error standard makes:

1. The hierarchy exists and the leaf classes multiply-inherit from the stdlib
   type a caller would already be catching, so ``except ValueError`` /
   ``except TypeError`` keep working (the break is purely additive).
2. Rejected values get a *value-naming* message: it names the offending value,
   states the rule or options, and (where one exists) suggests a fix.
3. The headline silent footguns now **raise** instead of returning garbage:
   ``final_time <= 0``, ``dt <= 0``, a typo'd system attribute, an unknown
   constructor parameter, and a System handed where a measured series is wanted.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.errors import (
    BackendError,
    ConvergenceError,
    InvalidInputError,
    InvalidParameterError,
    TSDynamicsError,
    invalid_value,
)
from tsdynamics.utils.grids import make_output_grid

# ---------------------------------------------------------------------------
# 1. The hierarchy (and the multiple-inheritance contract)
# ---------------------------------------------------------------------------


def test_errors_module_is_reachable():
    """``tsdynamics.errors`` is importable as a submodule."""
    import tsdynamics.errors as errs

    assert errs.TSDynamicsError is TSDynamicsError


@pytest.mark.parametrize(
    ("cls", "stdlib"),
    [
        (InvalidParameterError, ValueError),
        (InvalidInputError, TypeError),
        (ConvergenceError, RuntimeError),
        (BackendError, RuntimeError),
    ],
)
def test_leaf_classes_multiply_inherit(cls, stdlib):
    """Every leaf is a TSDynamicsError *and* the expected stdlib exception."""
    assert issubclass(cls, TSDynamicsError)
    assert issubclass(cls, stdlib)
    err = cls("boom")
    assert isinstance(err, TSDynamicsError)
    assert isinstance(err, stdlib)


def test_invalid_parameter_still_caught_by_value_error():
    """A plain ``except ValueError`` must still catch InvalidParameterError."""
    with pytest.raises(ValueError):
        raise InvalidParameterError("bad")


def test_invalid_input_still_caught_by_type_error():
    """A plain ``except TypeError`` must still catch InvalidInputError."""
    with pytest.raises(TypeError):
        raise InvalidInputError("bad")


# ---------------------------------------------------------------------------
# 2. The value-naming message builder
# ---------------------------------------------------------------------------


def test_invalid_value_rule_form_names_value():
    err = invalid_value("final_time", -5, rule="must be > 0")
    assert isinstance(err, InvalidParameterError)
    assert str(err) == "final_time must be > 0, got -5"


def test_invalid_value_options_form_lists_choices():
    err = invalid_value("backend", "gpu", options=["interp", "jit"])
    assert "gpu" in str(err)
    assert "interp" in str(err) and "jit" in str(err)


def test_invalid_value_appends_hint():
    err = invalid_value("dt", 0.0, rule="must be > 0", hint="use a small positive step")
    assert str(err).endswith("use a small positive step")


# ---------------------------------------------------------------------------
# 3a. Silent footguns — the output grid (dt<=0 / backwards window)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_grid_rejects_nonpositive_dt(dt):
    with pytest.raises(InvalidParameterError) as ei:
        make_output_grid(0.0, 10.0, dt)
    assert "dt" in str(ei.value) and repr(dt) in str(ei.value)
    assert isinstance(ei.value, ValueError)  # back-compat


@pytest.mark.parametrize(("t0", "tf"), [(0.0, -5.0), (0.0, 0.0), (5.0, 1.0)])
def test_grid_rejects_backwards_window(t0, tf):
    with pytest.raises(InvalidParameterError) as ei:
        make_output_grid(t0, tf, 0.1)
    assert "final_time" in str(ei.value)
    assert isinstance(ei.value, ValueError)


def test_grid_happy_path_unchanged():
    grid = make_output_grid(0.0, 1.0, 0.5)
    assert np.allclose(grid, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# 3b. Silent footguns — end-to-end through a real integrate()
# ---------------------------------------------------------------------------


def test_integrate_negative_final_time_raises():
    """Was a silent one-step garbage Trajectory; now raises with the value named."""
    lor = ts.systems.Lorenz()
    with pytest.raises(InvalidParameterError) as ei:
        lor.run(final_time=-5, dt=0.01)
    assert "final_time" in str(ei.value) and "-5" in str(ei.value)


def test_integrate_zero_dt_raises():
    """Was a bare ZeroDivisionError from the grid helper; now a domain error."""
    lor = ts.systems.Lorenz()
    with pytest.raises(InvalidParameterError) as ei:
        lor.run(final_time=10, dt=0.0)
    assert "dt" in str(ei.value)
    # the old failure mode was a ZeroDivisionError — assert we no longer leak it
    assert not isinstance(ei.value, ZeroDivisionError)


def test_integrate_happy_path_still_works():
    lor = ts.systems.Lorenz()
    traj = lor.run(final_time=2.0, dt=0.1)
    assert traj.y.shape[1] == 3
    assert traj.y.shape[0] > 1


# ---------------------------------------------------------------------------
# 3c. Silent footguns — typo'd attribute / unknown constructor parameter
# ---------------------------------------------------------------------------


def test_typo_attribute_raises_naming_value():
    """``lor.sigmaa = 99`` was silently stored; now rejected like with_params."""
    lor = ts.systems.Lorenz()
    with pytest.raises(InvalidParameterError) as ei:
        lor.sigmaa = 99
    msg = str(ei.value)
    assert "sigmaa" in msg and "sigma" in msg
    assert isinstance(ei.value, ValueError)
    # and the real parameter is untouched
    assert lor.sigma != 99


def test_setting_a_real_parameter_still_works():
    lor = ts.systems.Lorenz()
    lor.sigma = 12.0
    assert lor.sigma == 12.0


def test_private_and_known_attributes_pass_through():
    """Internal step state (``_``-prefixed) and class attrs are never rejected."""
    lor = ts.systems.Lorenz()
    lor._scratch = object()  # private — allowed
    lor.ic = [1.0, 2.0, 3.0]  # real instance attribute — allowed
    assert np.allclose(lor.ic, [1.0, 2.0, 3.0])


def test_unknown_constructor_parameter_raises():
    with pytest.raises(InvalidParameterError) as ei:
        ts.systems.Lorenz(params={"sigmaa": 9.0})
    assert "sigmaa" in str(ei.value) and "Declared" in str(ei.value)
    assert isinstance(ei.value, ValueError)


# ---------------------------------------------------------------------------
# 3d. Silent footguns — a System handed where a measured series is wanted
# ---------------------------------------------------------------------------


def test_data_analysis_on_a_system_is_not_silent():
    """A data-level analysis handed a System rejects it loudly, never silently.

    The half of the standard that is *live* today: the wrong input raises rather
    than being coerced into a meaningless number.  The value-naming half (a
    ``TSDynamicsError`` whose message names the System and the fix) is not yet
    met by the shared array coercion — it is tracked executably by the strict
    ``xfail`` row ``open-wrong-type-input-message`` in
    ``tests/test_polish_standards.py``.
    """
    lor = ts.systems.Lorenz()
    with pytest.raises(TypeError):
        ts.analysis.lyapunov_from_data(lor)


def test_reject_system_names_the_system_and_the_fix():
    """The shared guard answers with a value-naming ``InvalidInputError``.

    This is the guard the data-first coercions are to call first (see the
    ``open-wrong-type-input-message`` row in ``tests/test_polish_standards.py``):
    it names the offending system, the analysis, and the two-line fix.
    """
    from tsdynamics.analysis._common import reject_system

    with pytest.raises(InvalidInputError) as ei:
        reject_system(ts.systems.Lorenz(), analysis="correlation_dimension")
    msg = str(ei.value)
    assert "Lorenz" in msg
    assert "system" in msg.lower()
    assert "correlation_dimension()" in msg
    # §5.6's own wording: the recipe is the run-it-first line, family-shaped.
    assert "traj = system.run(" in msg
    # Additive: an existing `except TypeError` still catches it.
    assert isinstance(ei.value, TypeError)


@pytest.mark.parametrize(
    "factory,front_door",
    [
        (lambda: ts.systems.Lorenz(), "run(final_time=100.0, dt=0.01)"),
        (lambda: ts.systems.Henon(), "run(steps=10000)"),
        (lambda: ts.systems.MackeyGlass(), "run(final_time=100.0, dt=0.01)"),
        (
            lambda: ts.systems.Rossler().poincare("y", 0.0),
            "run(steps=10000)",
        ),
    ],
    ids=["ode", "map", "dde", "derived"],
)
def test_reject_system_shows_the_front_door_that_system_has(factory, front_door):
    """The suggested fix names a method the offending object actually has.

    Since v6 that method is ``run`` on every family — which is exactly why this
    gate matters more, not less: the message used to name ``integrate`` /
    ``iterate`` / ``trajectory``, three verbs that no longer exist, and the
    horizon word still differs per family (``final_time`` for a flow, ``steps``
    for a map or a section).
    """
    from tsdynamics.analysis._common import reject_system

    system = factory()
    with pytest.raises(InvalidInputError) as ei:
        reject_system(system, analysis="rqa")
    message = str(ei.value)
    # §5.6 hands back the family-shaped RUN line; the horizon word still differs
    # (a flow takes a time, a map a count), which is what this gate is about.
    assert "traj = system.run(" in message
    assert "dt=" in message if "final_time" in front_door else True
    assert callable(system.run)


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((10, 3)),
        np.linspace(0.0, 1.0, 10),
        [1.0, 2.0, 3.0],
        [[1.0, 2.0], [3.0, 4.0]],
        (0.0, 1.0),
        None,
    ],
    ids=["points", "series", "list", "nested-list", "tuple", "none"],
)
def test_reject_system_passes_measured_data_through(data):
    """Genuine measured data (and ``None``) is never mistaken for a system."""
    from tsdynamics.analysis._common import is_system, reject_system

    assert not is_system(data)
    reject_system(data, analysis="rqa")  # must not raise


def test_reject_system_passes_a_trajectory_through():
    """A ``Trajectory`` carries a ``.system`` reference but *is* measured data."""
    from tsdynamics.analysis._common import is_system, reject_system

    traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
    assert not is_system(traj)
    reject_system(traj, analysis="rqa")  # must not raise


def test_data_analysis_on_a_series_still_works():
    """The same estimator on a genuine measured series recovers the known exponent.

    The point of this test is that rejecting a ``System`` must not also reject
    real measured data.  It is anchored on a series with a *published* answer
    rather than on "a number came back": the Henon map at ``a=1.4, b=0.3`` has
    maximal Lyapunov exponent ``lambda_1 ~= 0.419`` per iterate (Henon 1976),
    and the scalar ``x`` series must reproduce it to a few percent.

    It deliberately does **not** use a random walk.  A stochastic series has no
    deterministic Lyapunov exponent at all, so an estimator is *right* to refuse
    one — asserting a finite value there would pin the wrong behaviour.
    """
    orbit = ts.systems.Henon().run(steps=6000, ic=[0.1, 0.1])
    x = np.asarray(orbit.y)[1000:, 0]  # drop the transient, keep one scalar channel

    result = ts.analysis.lyapunov_from_data(x, dimension=2, delay=1)

    assert result.trusted, "the Henon x series must yield a genuine scaling region"
    assert np.isfinite(float(result))
    assert abs(float(result) - 0.419) < 0.05, (
        f"maximal exponent {float(result):.4f} disagrees with the literature 0.419"
    )


def test_catch_all_via_base_class():
    """A user can catch everything deliberate through the one base class."""
    lor = ts.systems.Lorenz()
    with pytest.raises(TSDynamicsError):
        lor.run(final_time=-1.0, dt=0.1)


# ---------------------------------------------------------------------------
# 4. The #1 documented pitfall — a numeric routine inside a symbolic kernel
#
# `math.*` / `np.*` inside `_equations` / `_drift` / `_diffusion` / `_step` used
# to surface as a bare "RuntimeError: Symbol cannot be evaluated." with no file,
# no line and no fix.  The guard that had always existed in `lower_map` is now
# shared by all four families, so every one gives the same actionable error.
#
# These tests were missing entirely when the guard was widened; without them a
# refactor could drop the diagnosis and only the raw exception type would move.
# ---------------------------------------------------------------------------


def _lowered(system):
    """Lower ``system`` with the right family entry point (bypassing the cache)."""
    from tsdynamics.engine import compile as _compile

    if hasattr(type(system), "_drift"):
        return _compile.lower_sde(system)
    if system.is_discrete:
        return _compile.lower_map(system)
    if hasattr(type(system), "delays") or "Delay" in type(system).__mro__[1].__name__:
        return _compile.lower_dde(system)
    return _compile.lower_ode(system)


def test_numeric_call_in_an_ode_kernel_names_the_call_and_the_fix():
    """``math.sin`` in ``_equations`` is diagnosed, not left as a raw RuntimeError."""
    import math

    from tsdynamics.engine.compile import TapeCompileError

    class _MathODE(ts.ContinuousSystem):
        params = {"s": 10.0}
        dim = 2
        default_ic = [1.0, 1.0]

        @staticmethod
        def _equations(y, t, *, s):
            return [s * y(1), math.sin(y(0))]

    with pytest.raises(TapeCompileError) as excinfo:
        _lowered(_MathODE())
    message = str(excinfo.value)

    assert "_MathODE" in message, "the message must name the system"
    assert "_equations" in message, "the message must name the kernel"
    assert "math.sin" in message, "the message must name the offending call"
    assert "symengine" in message, "the message must point at the replacement"
    assert "test_errors.py" in message, "the message must locate the source line"


def test_numeric_call_in_an_sde_diffusion_names_the_kernel():
    """The guard reaches ``_diffusion``, not only the drift."""
    from tsdynamics.engine.compile import TapeCompileError

    class _NumpySDE(ts.StochasticSystem):
        params = {"mu": 1.0}
        dim = 1
        default_ic = [1.0]

        @staticmethod
        def _drift(y, t, *, mu):
            return [-mu * y(0)]

        @staticmethod
        def _diffusion(y, t, *, mu):
            return [np.sqrt(y(0) ** 2 + 1.0)]

    with pytest.raises(TapeCompileError) as excinfo:
        _lowered(_NumpySDE())
    message = str(excinfo.value)

    assert "_diffusion" in message, "the message must name the failing kernel, not `_drift`"
    assert "np.sqrt" in message
    assert "symengine" in message


def test_missing_structural_params_is_not_diagnosed_as_a_numeric_call():
    """The *other* documented pitfall gets its own diagnosis, not the wrong one.

    A control parameter reaches the kernel as a SymEngine symbol, so ``range(N)``
    fails with an integer-coercion ``TypeError``.  The fix is
    ``_structural_params``; answering that with "use symengine.sin instead of
    math.sin" would send the user in entirely the wrong direction, which is what
    the shared guard's catch-all branch originally did.
    """
    from tsdynamics.engine.compile import TapeCompileError

    class _NoStructural(ts.ContinuousSystem):
        params = {"N": 4, "F": 8.0}
        dim = 4
        default_ic = [1.0, 1.0, 1.0, 1.0]

        @staticmethod
        def _equations(y, t, *, N, F):
            return [y((i + 1) % N) - y(i - 1) + F for i in range(N)]

    with pytest.raises(TapeCompileError) as excinfo:
        _lowered(_NoStructural())
    message = str(excinfo.value)

    assert "_structural_params" in message, "the message must name the actual fix"
    assert "'N'" in message, "the message must name the integer-valued candidate"
    # The wrong diagnosis must not be offered for this failure.
    assert "Use the SymEngine equivalents" not in message, (
        "an integer-coercion failure is not a numeric-routine failure; "
        "offering the SymEngine-replacement advice here misdiagnoses it"
    )


# ---------------------------------------------------------------------------
# 4. The System guard is wired into EVERY data-first analysis (registry-driven)
# ---------------------------------------------------------------------------
#
# The guard existing is not the point — it shipped once with zero callers, and
# the user-visible defect (`TypeError: float() argument must be ... not
# 'Lorenz'`) was unchanged.  These tests walk the *registry* so a newly
# registered analysis cannot quietly rejoin the un-guarded set.
#
# `analysis.dimensions` and `analysis.embedding` are guarded by their own owner
# and carry their own coverage; this file owns the rest.
_GUARDED_PACKAGES = (
    "tsdynamics.analysis.basins",
    "tsdynamics.analysis.chaos",
    "tsdynamics.analysis.fixedpoints",
    "tsdynamics.analysis.lyapunov",
    "tsdynamics.analysis.orbits",
    "tsdynamics.analysis.recurrence",
)

#: Data-first analyses: the first positional argument is a measured series, a
#: point set, or an already computed result — never a live system.  Each maps to
#: the extra keyword arguments needed to reach its own body.
_DATA_FIRST: dict[str, dict] = {
    "basin_entropy": {},
    "kaplan_yorke_dimension": {},
    "lyapunov_from_data": {},
    "estimate_period": {},
    "recurrence_matrix": {},
    "resilience": {"attractor_id": 1},
    "rqa": {},
    "tipping_points": {},
    "uncertainty_exponent": {},
    "wada_property": {},
    "windowed_rqa": {"window": 10},
}

#: System-first analyses: a ``System`` is exactly what they want.
_SYSTEM_FIRST = frozenset(
    {
        "attractors",
        "basin_fractions",
        "basins",
        "continuation",
        "expansion_entropy",
        "fixed_points",
        "gali",
        "lyapunov_spectrum",
        "max_lyapunov",
        "orbit_diagram",
        "periodic_orbits",
    }
)

#: Dual-convention analyses: documented to take *either* a system or data, so the
#: guard must NOT fire for them.
_DUAL = frozenset({"poincare_section", "return_map", "zero_one_test"})


def _guarded_registry_names() -> set[str]:
    """Every registered analysis owned by the packages this file guards."""
    from tsdynamics import registry

    return {
        name
        for name in registry.analyses.names()
        if registry.analyses.get(name).__module__.startswith(_GUARDED_PACKAGES)
    }


def test_every_guarded_analysis_is_classified():
    """A new analysis in these packages must be classified before it can ship.

    Registry-driven on purpose: the failure mode being defended against is an
    analysis joining the library with no thought about what its first argument
    is, which is exactly how the guard came to have zero callers.
    """
    classified = set(_DATA_FIRST) | _SYSTEM_FIRST | _DUAL
    registered = _guarded_registry_names()
    assert registered - classified == set(), (
        "unclassified analyses — add each to _DATA_FIRST, _SYSTEM_FIRST or _DUAL "
        "in tests/test_errors.py and wire reject_system() if it is data-first"
    )
    assert classified - registered == set(), "stale classification entries"


@pytest.mark.parametrize("name", sorted(_DATA_FIRST))
def test_data_first_analysis_rejects_a_system_with_a_usable_message(name):
    """Every data-first analysis names the system and the fix, not NumPy's coercion.

    The pre-fix behaviour leaked ``TypeError: float() argument must be a string
    or a real number, not 'Lorenz'`` — a message that names neither the mistake
    nor the remedy.
    """
    from tsdynamics import registry

    fn = registry.analyses.get(name)
    system = ts.systems.Lorenz()
    with pytest.raises(InvalidInputError) as ei:
        fn(system, **_DATA_FIRST[name])
    msg = str(ei.value)
    assert "Lorenz" in msg, "the message must name the offending system"
    assert "system" in msg.lower(), "the message must say what went wrong"
    assert f"{name}()" in msg, "the message must name the analysis called"
    assert "float() argument" not in msg
    # additive: an existing `except TypeError` still catches it
    assert isinstance(ei.value, TypeError)


@pytest.mark.parametrize("name", sorted(_DATA_FIRST))
def test_data_first_analysis_message_shows_a_runnable_next_step(name):
    """The remedy names a call the offending object (or the library) really has."""
    from tsdynamics import registry
    from tsdynamics.analysis._common import front_door

    fn = registry.analyses.get(name)
    system = ts.systems.Lorenz()
    with pytest.raises(InvalidInputError) as ei:
        fn(system, **_DATA_FIRST[name])
    msg = str(ei.value)
    # either the generic "run it and pass the trajectory" recipe ...
    assert front_door(system)  # the helper still answers for this family
    if "traj = system.run(" in msg:
        assert callable(system.run)
        return
    # ... or an analysis-specific recipe naming a real public entry point.
    named = [w for w in ("basins", "lyapunov_spectrum", "continuation") if w in msg]
    assert named, f"{name}: message offers no runnable next step:\n{msg}"
    for w in named:
        assert callable(getattr(ts.analysis, w))


@pytest.mark.parametrize("name", sorted(_DUAL))
def test_dual_convention_analysis_still_accepts_a_system(name):
    """The guard must not fire where a ``System`` is a documented input.

    ``poincare_section`` / ``return_map`` / ``zero_one_test`` take either a
    system or data; guarding them would break the documented calling convention,
    so this is the counterweight to the tests above.
    """
    from tsdynamics import registry
    from tsdynamics.analysis._common import is_system

    fn = registry.analyses.get(name)
    calls = {
        "poincare_section": dict(plane=("y", 0.0, "up"), crossings=5, seed=0),
        # an extremum return map needs a flow (a map has no continuous extrema)
        "return_map": dict(n=20, final_time=200.0, dt=0.02, components=2),
        "zero_one_test": dict(n=500, components=0),
    }
    system = ts.systems.Henon() if name == "zero_one_test" else ts.systems.Rossler()
    assert is_system(system)  # the guard would fire if one were wired in
    fn(system, **calls[name])  # must not raise
