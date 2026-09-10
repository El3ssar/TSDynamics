"""
Tests for continuous ODE systems (``ContinuousSystem`` subclasses).

Instantiation tests are registry-driven and cover every built-in system
(fast, no JiT compilation).  A curated subset is integration-tested in the
``slow`` tier; the exhaustive every-system compile sweep runs nightly under
``-m full``.

Scope note: the integration sweeps here are **shape and finiteness smoke tests**
— they prove a system integrates and returns an array of the right shape, and
deliberately nothing more.  Whether the resulting orbit is the attractor the
system claims (bounded, non-degenerate, recurrent, with the right sign of
leading exponent) is asserted for every catalogue system by
``tests/test_catalogue_dynamics.py``.  Do not read a green sweep here as
evidence that a kernel is correct: ``isfinite`` at ``final_time=2.0`` is cleared
by an orbit heading for ``1e19``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import INTEGRATION_SAMPLE

from tsdynamics import registry

# ---------------------------------------------------------------------------
# Instantiation — fast, every registered ODE system
# ---------------------------------------------------------------------------


def test_ode_instantiation(ode_entry) -> None:
    sys = ode_entry.cls()
    assert sys.dim is not None and sys.dim > 0
    assert len(sys.params) == len(type(sys).params)


def test_ode_params_as_attributes(ode_entry) -> None:
    sys = ode_entry.cls()
    for key in sys.params:
        assert hasattr(sys, key), f"{ode_entry.name} missing attribute for param {key!r}"


# ---------------------------------------------------------------------------
# Lorenz96 / KuramotoSivashinsky / MultiChua — variable-dim constructors
# (these were broken before the structural-params fix)
# ---------------------------------------------------------------------------


def test_lorenz96_default_constructor() -> None:
    import tsdynamics as ts

    lor = ts.Lorenz96()
    assert lor.dim == 20
    assert lor.params["N"] == 20
    assert lor.params["f"] == 8.0


def test_lorenz96_dim_follows_n() -> None:
    import tsdynamics as ts

    lor = ts.Lorenz96(N=12)
    assert lor.dim == 12
    assert lor.params["N"] == 12


def test_kuramoto_sivashinsky_default_ic_is_zero_mean() -> None:
    import tsdynamics as ts

    ks = ts.KuramotoSivashinsky(N=8, L=8.0)
    assert ks.dim == 8
    assert ks.ic is not None
    assert ks.ic.shape == (8,)
    assert abs(ks.ic.mean()) < 1e-10


def test_kuramoto_sivashinsky_default_ic_is_reproducible() -> None:
    """Two instances with the same (N, L) must yield byte-identical default ICs."""
    import tsdynamics as ts

    a = ts.KuramotoSivashinsky(N=32, L=22.0).ic
    b = ts.KuramotoSivashinsky(N=32, L=22.0).ic
    np.testing.assert_array_equal(a, b)


def test_kuramoto_sivashinsky_default_ic_has_target_rms() -> None:
    """Broadband IC is normalised to RMS=0.5 (the documented target)."""
    import tsdynamics as ts

    ks = ts.KuramotoSivashinsky(N=64, L=22.0)
    rms = float(np.sqrt(np.mean(ks.ic**2)))
    assert rms == pytest.approx(0.5, rel=1e-9)


def test_kuramoto_sivashinsky_rejects_small_n() -> None:
    import tsdynamics as ts

    with pytest.raises(ValueError):
        ts.KuramotoSivashinsky(N=4)


def test_multichua_dim_follows_n_circuits() -> None:
    import tsdynamics as ts

    mc = ts.MultiChua(n_circuits=4)
    assert mc.dim == 12
    assert mc.params["n_circuits"] == 4


# ---------------------------------------------------------------------------
# Constructor parameter keywords
#
# The five variable-dimension systems own a custom ``__init__`` (it has to
# resolve ``dim`` from a structural parameter).  A custom ``__init__`` must not
# *shadow* the constructor-keyword front door — every declared parameter has to
# stay reachable as a plain keyword, and an unknown one has to be rejected by
# name.  ``MultiChua`` used to accept only ``n_circuits``, so
# ``MultiChua(alpha=10.0)`` raised ``TypeError: unexpected keyword argument``.
# ---------------------------------------------------------------------------

_CUSTOM_INIT_SYSTEMS = (
    "Lorenz96",
    "KuramotoSivashinsky",
    "MultiChua",
    "GrayScott",
    "SwiftHohenberg",
)


@pytest.mark.parametrize("name", _CUSTOM_INIT_SYSTEMS)
def test_custom_init_systems_accept_every_declared_parameter(name: str) -> None:
    """Each declared parameter of a custom-``__init__`` system is a keyword."""
    entry = registry.get(name)
    for key, default in entry.params.items():
        # Perturb structurally-safe parameters only: a wrong N would change dim.
        value = default
        instance = entry.cls(**{key: value})
        assert instance.params[key] == value, f"{name}: {key}= did not reach params"


@pytest.mark.parametrize("name", _CUSTOM_INIT_SYSTEMS)
def test_custom_init_systems_reject_an_unknown_parameter_by_name(name: str) -> None:
    """A misspelled parameter raises ``InvalidParameterError`` naming the valid ones."""
    from tsdynamics.errors import InvalidParameterError

    entry = registry.get(name)
    real = next(iter(entry.params))
    typo = real + "x"
    with pytest.raises(InvalidParameterError) as excinfo:
        entry.cls(**{typo: 1.0})
    message = str(excinfo.value)
    assert typo in message, f"{name}: message does not name the offending parameter"
    assert real in message, f"{name}: message does not list the declared parameters"


def test_multichua_circuit_parameters_are_constructor_keywords() -> None:
    """The worked example: ``MultiChua(2, alpha=10.0)`` (used to be a TypeError)."""
    import tsdynamics as ts

    mc = ts.MultiChua(2, alpha=10.0, kappa=0.5)
    assert mc.dim == 6
    assert mc.params["alpha"] == 10.0
    assert mc.params["kappa"] == 0.5
    # Untouched parameters keep their class defaults.
    assert mc.params["beta"] == ts.MultiChua.params["beta"]


# ---------------------------------------------------------------------------
# Integration — slow tier: curated representative sample
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", INTEGRATION_SAMPLE)
def test_ode_integration_shape_and_finiteness(name: str) -> None:
    sys = registry.get(name).cls()
    traj = sys.integrate(final_time=2.0, dt=0.1, rtol=1e-5, atol=1e-7)
    assert traj.t.ndim == 1
    assert traj.y.ndim == 2
    assert traj.y.shape[0] == traj.t.shape[0]
    assert traj.y.shape[1] == sys.dim
    assert np.all(np.isfinite(traj.y))


# ---------------------------------------------------------------------------
# Integration — full tier: every registered ODE system (nightly)
# ---------------------------------------------------------------------------


@pytest.mark.full
@pytest.mark.slow  # belt-and-braces: `-m "not slow"` alone must never trigger 120 compiles
def test_ode_full_integration_sweep(ode_entry) -> None:
    import zlib

    from _sampling import HARD_TO_INTEGRATE, HEAVY_FIELD_CATEGORIES

    if ode_entry.name in HARD_TO_INTEGRATE:
        pytest.skip(HARD_TO_INTEGRATE[ode_entry.name])
    if ode_entry.category in HEAVY_FIELD_CATEGORIES:
        pytest.skip(f"{ode_entry.name}: high-dim PDE field — covered by the viz field tests")
    sys = ode_entry.cls()
    # Deterministic per-system IC: systems with a default_ic use it; the rest
    # get a fixed (seeded) draw so the sweep is reproducible, not flaky.
    np.random.seed(zlib.crc32(ode_entry.name.encode()) & 0xFFFFFFFF)
    ic = sys.resolve_ic(None)
    # Each system carries its own _default_method (stiff systems default to an
    # implicit solver), so the plain default path must integrate them all.
    traj = sys.integrate(ic=ic, final_time=2.0, dt=0.1, rtol=1e-5, atol=1e-7)
    assert traj.y.shape[1] == sys.dim
    assert np.all(np.isfinite(traj.y))


# ---------------------------------------------------------------------------
# Integration behaviour one-offs — slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ode_time_starts_at_zero() -> None:
    import tsdynamics as ts

    traj = ts.Lorenz().integrate(final_time=1.0, dt=0.1)
    assert traj.t[0] == pytest.approx(0.0)


@pytest.mark.slow
def test_ode_custom_ic_stored() -> None:
    import tsdynamics as ts

    ic = [1.0, 1.0, 1.0]
    lor = ts.Lorenz()
    lor.integrate(final_time=1.0, dt=0.1, ic=ic)
    np.testing.assert_array_almost_equal(lor.ic, ic)


@pytest.mark.slow
def test_ode_random_ic_stored_when_none_supplied() -> None:
    import tsdynamics as ts

    r = ts.Rossler()
    assert r.ic is None
    r.integrate(final_time=0.5, dt=0.1)
    assert r.ic is not None
    assert r.ic.shape == (3,)


@pytest.mark.slow
def test_ode_dop853_integrator() -> None:
    import tsdynamics as ts

    traj = ts.Lorenz().integrate(final_time=2.0, dt=0.1, method="dop853")
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_lorenz96_integrates() -> None:
    """Lorenz-96 with non-default N — was broken before the structural-params fix."""
    import tsdynamics as ts

    lor = ts.Lorenz96(N=10, f=8.0)
    traj = lor.integrate(final_time=2.0, dt=0.1)
    assert traj.y.shape == (traj.t.shape[0], 10)
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_kuramoto_sivashinsky_integrates() -> None:
    """KS with default IC — was broken before the structural-params fix."""
    import tsdynamics as ts

    ks = ts.KuramotoSivashinsky(N=8, L=8.0)
    traj = ks.integrate(final_time=2.0, dt=0.1)
    assert traj.y.shape == (traj.t.shape[0], 8)
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
@pytest.mark.parametrize("L", [22.0, 30.0, 60.0])
def test_kuramoto_sivashinsky_large_l_is_nontrivial(L: float) -> None:
    """
    Regression test for the "horizontal stripes" bug: with the old default
    IC (``0.01 cos(2π x / L)``), KS at ``L >= 30`` stayed near zero for the
    whole integration window because only the lowest, marginally unstable
    Fourier mode was excited.  The broadband default IC drives the system
    into the chaotic attractor for every ``L`` tested.
    """
    import tsdynamics as ts

    N = max(64, 4 * int(np.ceil(L)))
    ks = ts.KuramotoSivashinsky(N=N, L=L)
    traj = ks.integrate(final_time=120.0, dt=0.5, rtol=1e-6, atol=1e-9)
    y_post = traj.y[traj.t > 60.0]
    assert np.all(np.isfinite(y_post))
    temporal_std = float(np.sqrt(y_post.var(axis=0)).mean())
    assert temporal_std > 0.5, (
        f"L={L}: temporal_std={temporal_std:.3e} suggests near-frozen dynamics "
        f"(horizontal stripes regression). Expected >0.5 on the KS attractor."
    )
    assert y_post.max() - y_post.min() > 2.0


@pytest.mark.slow
def test_multichua_integrates() -> None:
    """MultiChua with default n_circuits — was broken before the structural-params fix."""
    import tsdynamics as ts

    mc = ts.MultiChua()
    traj = mc.integrate(final_time=2.0, dt=0.1, ic=0.1 * np.ones(mc.dim))
    assert traj.y.shape == (traj.t.shape[0], 9)
    assert np.all(np.isfinite(traj.y))
