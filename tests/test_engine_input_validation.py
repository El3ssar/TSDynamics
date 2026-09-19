"""Input validation and typed errors at the engine FFI boundary (v6 WP1).

Two contracts are pinned here, both of which the compiled bridge used to break:

**1. Tolerances are validated.**  ``rtol``/``atol`` reached the adaptive
controllers unchecked, so two silent failure modes shipped:

- a *negative* tolerance (a sign typo) made the per-component error scale
  ``sc = atol + rtol·|u|`` negative, the scaled error negative, and the accept
  test ``err <= 1`` vacuously true — the controller degenerated to *always
  accept* and returned an answer bit-identical to ``rtol = 1e6``, four orders
  less accurate, with no diagnostic at all;
- ``rtol = atol = 0`` (and ``NaN``) made the error test unsatisfiable, so the
  step size collapsed at ``t0`` and the run was reported as
  ``ConvergenceError: integration diverged … step size collapsed to 0 at t = 0``
  — a blow-up of a model that was never integrated.

Both are now rejected as :class:`~tsdynamics.errors.InvalidParameterError`, at
*every* surface that takes tolerances, with a message that names the value.

**2. Engine divergence is a typed** :class:`~tsdynamics.errors.ConvergenceError`.
The bridge used to raise a bare ``RuntimeError`` for ``EngineError::Diverged``,
so the most common failure on the most common path violated the WS-ERRORS
contract and forced two call sites to sniff the message text for ``"diverg"``.
The mapping now lives in the binding layer, so every family inherits it.
"""

import math

import numpy as np
import pytest

# The whole module is about the compiled bridge's boundary behaviour.
_rust = pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts  # noqa: E402
from tsdynamics._engine import run as engine_run  # noqa: E402
from tsdynamics._engine.problem import build_problem  # noqa: E402
from tsdynamics._utils.grids import make_output_grid  # noqa: E402
from tsdynamics.errors import ConvergenceError, InvalidParameterError  # noqa: E402

# The tolerance pairs no adaptive controller can act on.
BAD_TOLERANCES = [
    {"rtol": -1.0},
    {"atol": -1e-9},
    {"rtol": float("nan")},
    {"atol": float("nan")},
    {"rtol": float("inf")},
    {"rtol": 0.0, "atol": 0.0},
]


# ---------------------------------------------------------------------------
# Test systems
# ---------------------------------------------------------------------------


class _Blowup(ts.ContinuousSystem):
    """``dx/dt = x³`` — finite-time blow-up from any ``x0 > 0``."""

    dim = 2
    params: dict[str, float] = {}

    @staticmethod
    def _equations(y, t):
        return [y(0) ** 3, y(1) ** 3]


class _BlowupMap(ts.DiscreteMap):
    """``x ← a x³`` — escapes to infinity from ``|x0| > a^{-1/2}``."""

    dim = 1
    params = {"a": 10.0}

    @staticmethod
    def _step(x, a):
        return [a * x[0] ** 3]

    @staticmethod
    def _jacobian(x, a):
        return [[3 * a * x[0] ** 2]]


class _BlowupDDE(ts.DelaySystem):
    """A delayed cubic — blows up in finite time from a constant past of 2."""

    dim = 1
    params = {"tau": 1.0}

    @staticmethod
    def _equations(y, t, tau):
        return [y(0) ** 3 + y(0, t - tau) ** 3]


class _BlowupSDE(ts.StochasticSystem):
    """A cubic drift with weak additive noise — the drift blows up."""

    dim = 1
    params: dict[str, float] = {}

    @staticmethod
    def _drift(y, t):
        return [y(0) ** 3]

    @staticmethod
    def _diffusion(y, t):
        return [0.1]


# ---------------------------------------------------------------------------
# 1. Tolerance validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tol", BAD_TOLERANCES, ids=lambda t: repr(sorted(t.items())))
def test_integrate_rejects_inadmissible_tolerances(tol):
    """The dense ODE path names the offending tolerance instead of integrating."""
    with pytest.raises(InvalidParameterError) as exc:
        ts.systems.Lorenz().run(final_time=1.0, dt=0.01, ic=[1.0, 1.0, 1.0], **tol)
    msg = str(exc.value)
    name = "rtol" if "rtol" in tol else "atol"
    assert name in msg, msg
    # The message quotes what it received, per the value-naming standard.
    assert any(tok in msg for tok in ("-1", "NaN", "inf", "0")), msg


@pytest.mark.parametrize("tol", BAD_TOLERANCES, ids=lambda t: repr(sorted(t.items())))
def test_every_tolerance_surface_rejects_inadmissible_tolerances(tol):
    """Not just ``integrate``: the ensemble, DDE, events and stepper agree.

    The guard lives in one place (the bridge's validated ``Tolerances`` newtype,
    which is the only way to build a solver), so a surface cannot forget it.
    """
    lorenz = ts.systems.Lorenz()
    ics = np.array([[1.0, 1.0, 1.0], [1.1, 1.0, 1.0]])

    with pytest.raises(InvalidParameterError):
        engine_run.ensemble(lorenz, ics, final_time=1.0, dt=0.01, **tol)

    with pytest.raises(InvalidParameterError):
        lorenz.run(final_time=1.0, dt=0.01, ic=[1.0, 1.0, 1.0], events=[("z", 20.0)], **tol)

    # The resumable stepper builds its engine handle lazily on the first ``step``,
    # so that is where its tolerances are validated.
    stepped = ts.systems.Lorenz()
    stepped.reinit([1.0, 1.0, 1.0], **tol)
    with pytest.raises(InvalidParameterError):
        stepped.step(0.01)

    with pytest.raises(InvalidParameterError):
        ts.systems.MackeyGlass().run(final_time=5.0, dt=0.5, **tol)


@pytest.mark.parametrize("tol", [{"atol": 0.0}, {"rtol": 0.0}, {}])
def test_one_sided_error_control_is_still_accepted(tol):
    """``atol = 0`` (pure relative) and ``rtol = 0`` (pure absolute) are valid.

    Only the *pair* being zero is unsatisfiable; rejecting either alone would
    break standard error control that SciPy also accepts.
    """
    traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.01, ic=[1.0, 1.0, 1.0], **tol)
    assert np.isfinite(traj.y).all()


def test_negative_rtol_no_longer_silently_matches_a_huge_one():
    """The measured symptom: ``rtol=-1`` used to be bit-identical to ``rtol=1e6``.

    ``rtol=1e6`` stays *legal* — it is merely a loose request, so it is accepted
    rather than rejected; ``rtol=-1`` is inadmissible and is rejected.  That
    accept/reject split is what this test pins.

    The ``max_step=dt`` on the loose leg is deliberate and is **not** a
    re-baselining.  Before v6 the stepper was forced to land on every output
    sample, so ``dt`` silently doubled as a step ceiling and an absurdly loose
    ``rtol`` could not blow the step up.  Since v6 ``dt`` is sampling-only, so
    "I do not care about accuracy" now genuinely means the controller grows the
    step without bound — and an unbounded explicit step on a chaotic flow
    diverges, correctly and loudly.  ``max_step=dt`` states the old implicit
    bound explicitly and reproduces the old behaviour exactly (verified: the run
    is finite and on the attractor).  The assertion is unchanged; only the
    *mechanism* that bounds the step is now written down instead of accidental.
    """
    lorenz = ts.systems.Lorenz()
    loose = lorenz.run(final_time=5.0, dt=0.01, ic=[1.0, 1.0, 1.0], rtol=1e6, max_step=0.01)
    assert np.isfinite(loose.y).all()
    with pytest.raises(InvalidParameterError):
        lorenz.run(final_time=5.0, dt=0.01, ic=[1.0, 1.0, 1.0], rtol=-1.0)


def test_zero_tolerance_is_not_reported_as_a_divergence():
    """``rtol = atol = 0`` used to raise ``ConvergenceError`` at ``t = 0``."""
    with pytest.raises(InvalidParameterError):
        ts.systems.Lorenz().run(final_time=1.0, dt=0.01, ic=[1.0, 1.0, 1.0], rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 2. Divergence is a typed ConvergenceError on every family
# ---------------------------------------------------------------------------


def test_ode_divergence_raises_convergence_error():
    with pytest.raises(ConvergenceError):
        _Blowup().run(final_time=100.0, dt=0.01, ic=[2.0, 2.0])


def test_stepper_divergence_raises_convergence_error():
    system = _Blowup()
    system.reinit([2.0, 2.0])
    with pytest.raises(ConvergenceError):
        for _ in range(100_000):
            system.step(0.01)


def test_map_divergence_raises_convergence_error():
    with pytest.raises(ConvergenceError):
        _BlowupMap().run(steps=200, ic=[2.0])


def test_dde_divergence_raises_convergence_error():
    with pytest.raises(ConvergenceError):
        _BlowupDDE().run(final_time=50.0, dt=0.1, ic=[2.0])


def test_sde_divergence_raises_convergence_error():
    with pytest.raises(ConvergenceError):
        _BlowupSDE().run(final_time=50.0, dt=0.01, ic=[3.0], seed=0)


def test_raw_ffi_divergence_is_typed_not_a_bare_runtime_error():
    """The mapping lives in the binding, so even a direct ``_rust`` call is typed.

    This is the load-bearing assertion: fixing it at the source is what lets the
    Python call sites catch the *type* instead of sniffing ``"diverg"`` out of a
    message.
    """
    problem = build_problem(_Blowup())
    with pytest.raises(ConvergenceError) as exc:
        _rust.integrate_dense(
            *problem.tape.to_arrays(),
            np.array([2.0, 2.0]),
            problem.params_vec(),
            np.linspace(0.0, 100.0, 200),
            "rk45",
            1e-6,
            1e-9,
            math.inf,  # max_step: no ceiling (v6)
            False,  # dense: keep the landing march (v6)
            False,
        )
    assert type(exc.value) is ConvergenceError


def test_a_jit_failure_would_not_be_mislabelled_as_divergence():
    """A non-divergence ``RuntimeError`` must stay distinguishable.

    ``ConvergenceError`` is a *subclass* of ``RuntimeError``, so the narrowed
    ``except ConvergenceError`` at the map / map-Lyapunov call sites still lets an
    unrelated engine ``RuntimeError`` through untouched.
    """
    assert issubclass(ConvergenceError, RuntimeError)
    assert not isinstance(RuntimeError("cranelift said no"), ConvergenceError)


# ---------------------------------------------------------------------------
# 3. Output-grid / cadence guards
# ---------------------------------------------------------------------------


def test_empty_output_grid_is_rejected():
    """An empty ``t_eval`` returned ``(0, dim)``, contradicting "first row is the IC"."""
    problem = build_problem(ts.systems.Lorenz())
    with pytest.raises(ValueError, match="t_eval"):
        _rust.integrate_dense(
            *problem.tape.to_arrays(),
            np.array([1.0, 1.0, 1.0]),
            problem.params_vec(),
            np.array([], dtype=float),
            "rk45",
            1e-6,
            1e-9,
            math.inf,  # max_step: no ceiling (v6)
            False,  # dense: keep the landing march (v6)
            False,
        )


@pytest.mark.parametrize("dt", [float("inf"), float("nan"), 0.0, -1.0])
def test_make_output_grid_rejects_non_finite_dt(dt):
    """``if not dt > 0`` caught 0 / negative / NaN but *not* ``+inf``."""
    with pytest.raises(InvalidParameterError, match="dt"):
        make_output_grid(0.0, 1.0, dt)


@pytest.mark.parametrize("final_time", [float("inf"), float("nan")])
def test_make_output_grid_rejects_non_finite_final_time(final_time):
    """``final_time=inf`` used to reach ``np.arange`` ("Maximum allowed size exceeded")."""
    with pytest.raises(InvalidParameterError, match="final_time"):
        make_output_grid(0.0, final_time, 0.01)


def test_make_output_grid_rejects_non_finite_start():
    with pytest.raises(InvalidParameterError, match="t0"):
        make_output_grid(float("inf"), 1.0, 0.01)


@pytest.mark.parametrize("dt", [float("inf"), float("nan")])
def test_integrate_rejects_non_finite_dt(dt):
    """End to end: ``dt=inf`` used to yield a two-sample "trajectory"."""
    with pytest.raises(InvalidParameterError):
        ts.systems.Lorenz().run(final_time=1.0, dt=dt, ic=[1.0, 1.0, 1.0])


def test_integrate_rejects_non_finite_final_time():
    with pytest.raises(InvalidParameterError):
        ts.systems.Lorenz().run(final_time=float("inf"), dt=0.01, ic=[1.0, 1.0, 1.0])


@pytest.mark.parametrize("dt", [float("inf"), float("nan")])
def test_step_rejects_non_finite_dt(dt):
    """The stepping seam builds its two-node grid through the same helper."""
    system = ts.systems.Rossler()
    system.reinit([1.0, 1.0, 1.0])
    with pytest.raises(InvalidParameterError, match="dt"):
        system.step(dt)


# ---------------------------------------------------------------------------
# 4. A bad start state is a bad input, not a divergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_initial_state_is_rejected_not_called_a_divergence(bad):
    """It used to surface as "diverged … non-finite state at t = 0"."""
    problem = build_problem(ts.systems.Lorenz())
    with pytest.raises(ValueError, match="finite") as exc:
        _rust.integrate_dense(
            *problem.tape.to_arrays(),
            np.array([bad, 1.0, 1.0]),
            problem.params_vec(),
            np.linspace(0.0, 1.0, 11),
            "rk45",
            1e-6,
            1e-9,
            math.inf,  # max_step: no ceiling (v6)
            False,  # dense: keep the landing march (v6)
            False,
        )
    assert not isinstance(exc.value, ConvergenceError)


# ---------------------------------------------------------------------------
# 5. Every inadmissible *value* raises the same typed error
# ---------------------------------------------------------------------------
#
# The bridge draws a deliberate line between `EngineError::BadShape` (the
# *geometry* of the call — a slice of the wrong length, an empty grid — which a
# caller fixes by reshaping) and `EngineError::InvalidParameter` (the *value* of
# something passed, which reshaping cannot fix). Only the second maps to
# `InvalidParameterError`.
#
# Three guards sat on the wrong side of that line: a non-finite entry in the
# initial state, a non-finite entry in the output grid, and a descending grid.
# All three are values, and all three used to raise a bare `ValueError` while
# the guards *beside* them — non-finite `t0`, non-finite `dt` — raised the typed
# `InvalidParameterError`. A caller writing `except InvalidParameterError` to
# handle bad input therefore caught some of its own mistakes and not others.


def _lorenz_ffi_args(*, ic, t_eval):
    problem = build_problem(ts.systems.Lorenz())
    return (
        *problem.tape.to_arrays(),
        np.asarray(ic, dtype=float),
        problem.params_vec(),
        np.asarray(t_eval, dtype=float),
        "rk45",
        1e-6,
        1e-9,
        math.inf,  # max_step: no ceiling (v6)
        False,  # dense: keep the landing march (v6)
        False,
    )


@pytest.mark.parametrize(
    ("what", "kwargs"),
    [
        ("non-finite state", {"ic": [float("nan"), 1.0, 1.0], "t_eval": np.linspace(0, 1, 11)}),
        ("infinite state", {"ic": [float("inf"), 1.0, 1.0], "t_eval": np.linspace(0, 1, 11)}),
        ("non-finite grid time", {"ic": [1.0, 1.0, 1.0], "t_eval": [0.0, float("inf")]}),
        ("descending grid", {"ic": [1.0, 1.0, 1.0], "t_eval": [0.0, 1.0, 0.5]}),
    ],
)
def test_an_inadmissible_value_is_an_invalid_parameter_not_a_bare_value_error(what, kwargs):
    """Each of these is a *value* the caller got wrong, so each is typed."""
    with pytest.raises(InvalidParameterError):
        _rust.integrate_dense(*_lorenz_ffi_args(**kwargs))


def test_an_empty_grid_stays_a_shape_error():
    """The counter-case, so the dividing line is pinned from both sides.

    An empty ``t_eval`` really is geometry — the caller fixes it by passing a
    grid with samples in it — so it must *not* be reclassified along with the
    value guards above.
    """
    with pytest.raises(ValueError) as exc:
        _rust.integrate_dense(*_lorenz_ffi_args(ic=[1.0, 1.0, 1.0], t_eval=[]))
    assert not isinstance(exc.value, InvalidParameterError)


def test_the_stepper_agrees_with_the_dense_path_on_a_bad_state():
    """``reinit``/``set_state`` reject the same values through the same type.

    Validation is *deferred*: ``reinit`` and ``set_state`` only record the point
    (a cold system builds no engine stepper until it is stepped), so the guard
    fires on the first ``step``. What matters — and what this pins — is that when
    it does fire it is the same :class:`InvalidParameterError` the dense path
    raises, not a divergence report about a run that never started.
    """
    cold = ts.systems.Lorenz()
    cold.reinit([float("nan"), 1.0, 1.0])
    with pytest.raises(InvalidParameterError, match="finite"):
        cold.step(0.01)

    warm = ts.systems.Lorenz()
    warm.reinit([1.0, 1.0, 1.0])
    warm.step(0.01)
    warm.set_state([float("inf"), 1.0, 1.0])
    with pytest.raises(InvalidParameterError, match="finite"):
        warm.step(0.01)
