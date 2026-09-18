"""Backend-neutral variational core (stream C-DERIV).

Covers the extended-variational lowering (:mod:`tsdynamics.derived._variational`)
and :class:`~tsdynamics.derived.tangent.TangentSystem` as the one Lyapunov engine
every family delegates to.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics._engine.compile import clear_tape_cache, tape_cache_stats
from tsdynamics.derived._variational import (
    build_variational_tape,
    build_variational_tape_cached,
    embed_extended,
    split_extended,
)
from tsdynamics.derived.tangent import TangentSystem
from tsdynamics.families import ContinuousSystem


class LinOsc(ContinuousSystem):
    """Overdamped linear oscillator ``x'=y, y'=-k x - c y``.

    Jacobian ``[[0, 1], [-k, -c]]`` is constant, so the Lyapunov spectrum equals
    the real parts of its eigenvalues ``(-c ± sqrt(c²-4k)) / 2``.  With the
    defaults below the spectrum is exactly ``[-1, -2]`` — a closed-form oracle
    for the variational machinery, with off-diagonal Jacobian coupling.
    """

    params = {"k": 2.0, "c": 3.0}
    dim = 2
    variables = ("x", "y")

    @staticmethod
    def _equations(y, t, k, c):
        return [y(1), -k * y(0) - c * y(1)]


# ---------------------------------------------------------------------------
# Extended-tape lowering and packing
# ---------------------------------------------------------------------------


def test_embed_split_roundtrip() -> None:
    x = np.array([1.0, 2.0, 3.0])
    w = np.arange(6.0).reshape(3, 2)  # (dim=3, k=2)
    z = embed_extended(x, w)
    assert z.shape == (9,)
    x2, w2 = split_extended(z, 3, 2)
    np.testing.assert_array_equal(x2, x)
    np.testing.assert_array_equal(w2, w)


def test_variational_tape_shape_and_rhs() -> None:
    from tsdynamics._engine.compile import eval_tape

    s = LinOsc()
    k = 2
    tape = build_variational_tape(s, k)
    assert tape.dim == s.dim * (k + 1) == 6

    # Extended RHS at x=[1, 0.5], W = I_2: [y1, -k y0 - c y1, J@w0, J@w1].
    x = np.array([1.0, 0.5])
    w = np.eye(2)
    z = embed_extended(x, w)
    p = np.array([s.params["k"], s.params["c"]])  # control-name order
    out = eval_tape(tape, z, p, 0.0)

    kk, cc = 2.0, 3.0
    J = np.array([[0.0, 1.0], [-kk, -cc]])
    base = np.array([x[1], -kk * x[0] - cc * x[1]])
    expected = np.concatenate([base, (J @ w[:, 0]), (J @ w[:, 1])])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_variational_tape_partial_k() -> None:
    s = LinOsc()
    tape = build_variational_tape(s, 1)
    assert tape.dim == s.dim * 2 == 4


# ---------------------------------------------------------------------------
# Backend-neutral ODE Lyapunov via the reference engine (no JiTCODE, no wheel)
# ---------------------------------------------------------------------------


def test_backend_neutral_linear_spectrum_reference() -> None:
    """The reference (pure-Python) variational path reproduces the analytic spectrum."""
    tang = TangentSystem(LinOsc(), k=2, backend="reference")
    spec = ts.analysis.lyapunov_spectrum(
        tang, final_time=40.0, dt=0.25, transient=5.0, ic=[1.0, 0.5]
    )
    np.testing.assert_allclose(spec, [-1.0, -2.0], atol=0.02)


def test_backend_neutral_partial_spectrum_reference() -> None:
    """Only the leading exponent, via k=1 deviation vector."""
    tang = TangentSystem(LinOsc(), k=1, backend="reference")
    spec = ts.analysis.lyapunov_spectrum(
        tang, final_time=40.0, dt=0.25, transient=5.0, ic=[1.0, 0.5]
    )
    assert spec.shape == (1,)
    np.testing.assert_allclose(spec, [-1.0], atol=0.02)


def test_backend_neutral_deviations_orthonormal() -> None:
    tang = TangentSystem(LinOsc(), k=2, backend="reference")
    tang.reinit([1.0, 0.5])
    for _ in range(10):
        tang.step(0.25)
    q = tang.deviations()
    assert q.shape == (2, 2)
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-10)


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ODE tangent backend"):
        TangentSystem(ts.systems.Lorenz(), backend="bogus")


def test_jitcode_backend_is_rejected() -> None:
    """The retired ``jitcode`` variational backend is no longer a valid choice."""
    with pytest.raises(ValueError, match="unknown ODE tangent backend"):
        TangentSystem(ts.systems.Lorenz(), k=3, backend="jitcode")


# ---------------------------------------------------------------------------
# TangentSystem is the one engine: families delegate to it (identical results)
# ---------------------------------------------------------------------------


def test_map_family_delegates_to_tangent() -> None:
    """Family map ``lyapunov_spectrum`` is exactly ``TangentSystem.lyapunov_spectrum``."""
    via_family = ts.analysis.lyapunov_spectrum(ts.systems.Henon(), n=4000, ic=[0.1, 0.1])
    via_tangent = ts.analysis.lyapunov_spectrum(
        TangentSystem(ts.systems.Henon(), k=2), n=4000, ic=[0.1, 0.1]
    )
    np.testing.assert_array_equal(via_family, via_tangent)


def test_map_partial_spectrum_via_k() -> None:
    """``k`` is the canonical name (v4 glossary); ``n_exp`` was silently swallowed."""
    spec = ts.analysis.lyapunov_spectrum(ts.systems.Henon(), n=4000, ic=[0.1, 0.1], k=1)
    assert spec.shape == (1,)
    assert 0.3 < spec[0] < 0.5  # leading Hénon exponent ≈ 0.42


def test_tangent_lyapunov_records_meta() -> None:
    """A RUN records its own provenance — ``system.meta`` is gone in v6 (§8.3)."""
    m = ts.systems.Henon()
    tang = TangentSystem(m, k=2)
    spec = ts.analysis.lyapunov_spectrum(tang, n=2000, ic=[0.1, 0.1])
    assert spec.meta["analysis"] == "lyapunov_spectrum"
    assert spec.meta["k"] == 2


class _StiffLinOsc(ContinuousSystem):
    """Linear oscillator that *defaults to the stiff BDF kernel*.

    Same constant-Jacobian closed-form oracle as :class:`LinOsc` (spectrum
    ``[-1, -2]``), but with ``_default_method = "bdf"`` so the tangent flow is
    driven onto an *implicit* engine kernel.  Before the fix the extended
    variational tape was lowered with ``jacobian=False``, so the implicit kernel
    found no Jacobian tape and ``lyapunov_spectrum`` *raised*; this gives a fast,
    closed-form regression for that path without the cost of a genuinely stiff
    catalogue system.
    """

    params = {"k": 2.0, "c": 3.0}
    dim = 2
    variables = ("x", "y")
    _default_method = "bdf"

    @staticmethod
    def _equations(y, t, k, c):
        return [y(1), -k * y(0) - c * y(1)]


def test_variational_tape_emits_jacobian() -> None:
    """The extended variational tape carries its own Jacobian block.

    Regression guard for the P0 bug: a base flow whose ``_default_method`` is an
    implicit kernel (``"bdf"``) integrates the *pre-built* extended ODEProblem,
    and ``engine.run.integrate`` does not rebuild a pre-built problem
    ``with_jacobian=True`` — so the Jacobian must be present on the tape itself.
    """
    tape = build_variational_tape(LinOsc(), 2)
    assert tape.has_jacobian


def test_stiff_default_ode_lyapunov_does_not_raise() -> None:
    """A stiff-defaulted (``bdf``) flow's Lyapunov spectrum integrates cleanly.

    Fast closed-form proxy for the catalogue stiff systems (Oregonator,
    KuramotoSivashinsky, Duffing, the stiff Sprott jerks): the implicit kernel
    now has the extended Jacobian it needs, so the spectrum is finite, descending
    and matches the analytic ``[-1, -2]`` instead of raising.  Exercises the
    *engine* implicit path (the one the bug broke), so it skips without the
    compiled extension.
    """
    pytest.importorskip("tsdynamics._rust")
    tang = TangentSystem(_StiffLinOsc(), k=2, backend="interp")
    spec = ts.analysis.lyapunov_spectrum(
        tang, final_time=40.0, dt=0.25, transient=5.0, ic=[1.0, 0.5]
    )
    assert np.all(np.isfinite(spec))
    assert spec[0] >= spec[1]  # descending (QR order)
    np.testing.assert_allclose(spec, [-1.0, -2.0], atol=0.05)


def test_structural_change_rebuilds_extended_tape(monkeypatch) -> None:
    """A live structural-parameter change re-lowers the cached extended tape.

    The extended tape bakes in structural parameters; a stale cache would carry
    the wrong dimension/structure.  Reinitialising after a structural change must
    take the rebuild branch (the per-instance cache is keyed on the structural
    values).

    The *rebuild* is asserted by counting the builder call, not by object
    identity: since the ``perf/variational-tape-cache`` stream the build itself is
    memoised process-wide, so re-lowering the same math legitimately hands back the
    same shared ``Tape`` object.  What must still hold is that the branch runs — a
    genuine structural change would then key a different cache entry and get a
    structurally different tape.

    The key is ``(k, structural_values)``: the extended tape carries the state
    PLUS ``k`` tangent vectors, so two values of ``k`` are two different tapes.
    Keyed on the structural values alone (as it was before v6 round 9) a live
    ``tang.k = 3`` silently reused the ``k=2`` tape — see
    :func:`test_changing_k_rebuilds_the_extended_tape` below, which is the
    failing-first half of this pair.
    """
    from tsdynamics.derived import tangent as tangent_mod

    builds = {"n": 0}
    real = tangent_mod.build_variational_tape_cached

    def counting(system, k):
        builds["n"] += 1
        return real(system, k)

    monkeypatch.setattr(tangent_mod, "build_variational_tape_cached", counting)

    tang = TangentSystem(LinOsc(), k=2, backend="reference")
    tang.reinit([1.0, 0.5])
    first_tape = tang._ext_tape
    first_key = tang._ext_tape_key
    assert builds["n"] == 1

    # No structural params on LinOsc → the structural half of the key is the
    # empty tuple, and the tape is reused across reinit (a control/IC change is
    # not a structural change).
    assert first_key == (2, ())
    tang.reinit([0.2, 0.3])
    assert tang._ext_tape is first_tape
    assert tang._ext_tape_key == first_key
    assert builds["n"] == 1  # no rebuild

    # Simulate a structural change by poking the cached key stale; the next
    # reinit must take the rebuild branch (key mismatch path).
    tang._ext_tape_key = (2, (("N", 99),))
    tang.reinit([0.2, 0.3])
    assert builds["n"] == 2
    assert tang._ext_tape_key == (2, ())


def test_changing_k_rebuilds_the_extended_tape() -> None:
    """Changing ``k`` on a live ``TangentSystem`` re-lowers the extended tape.

    The extended variational tape carries the base state PLUS ``k`` tangent
    vectors, so ``k`` is part of the math the tape encodes, not a runtime knob
    read off the system.  Before v6 round 9 the per-instance cache was keyed on
    the structural parameters alone, so raising ``k`` after a ``reinit`` reused
    the narrower tape and ``split_extended`` failed with ``cannot reshape array
    of size 6 into shape (3,3)``.

    Asserted on the observable — the shape of :meth:`TangentSystem.deviations`
    — rather than on the private key, so the test survives a change of key
    representation.
    """
    tang = TangentSystem(ts.systems.Lorenz(), k=2, backend="reference")
    tang.reinit([1.0, 1.0, 1.0])
    assert tang.deviations().shape == (3, 2)

    tang.k = 3
    tang.reinit([1.0, 1.0, 1.0])
    assert tang.deviations().shape == (3, 3)


@pytest.mark.slow
def test_oregonator_stiff_lyapunov_finite_descending() -> None:
    """The genuinely-stiff Oregonator Lyapunov spectrum is finite and descending.

    End-to-end guard for the named P0 system: ``ts.systems.Oregonator()`` defaults to the
    implicit ``bdf`` kernel, so ``lyapunov_spectrum`` drives the extended
    variational ODE onto that kernel.  Before the fix this *raised* (no Jacobian
    tape); it must now return a finite, descending spectrum.  ``final_time`` is
    kept modest for speed — correctness of the *values* is covered by the
    closed-form oscillator oracles above; here we only assert it does not raise
    and is well-formed.

    Why the tolerance is pinned instead of taking the v6 library default
    -------------------------------------------------------------------
    The **extended variational** system for a stiff flow packs the base state and
    the deviation vectors into one error-weight vector.  For the Oregonator that
    spans ~16 decades: the base ``z`` reaches ``2.4e3`` while the strongly
    contracting tangent direction (``lambda_3 ~ -3e3``) decays to ``~3e-11``
    inside a single ``dt=0.01`` chunk.  A single global ``atol`` cannot serve
    both, and at ``atol=1e-12`` the BDF step collapses (``ConvergenceError`` at
    ``t~7.8``).

    That is a **pre-existing weakness of the stiff extended-variational path**,
    not something the v6 tolerance bump introduced:

    * The value was never converged at any tolerance — ``lambda_3`` measures
      -2984 / -2904 / -7405 / -8213 / -10001 at ``rtol=1e-6 … 1e-9``, a 3.4x
      spread. This test asserts only shape/finiteness/ordering for that reason.
    * Two of the four catalogue ODEs with an implicit ``_default_method``
      (``SprottL``, ``SprottJerk``) already raise ``ConvergenceError`` here at
      **both** the old and the new tolerance, and a third (``SprottP``) returns
      unrelated numbers at each.
    * The Oregonator's *flow* path, by contrast, is unambiguously **better** at
      the v6 default: at ``T=100`` its error against SciPy ``Radau`` at
      ``rtol=1e-12`` drops 1.89e-2 -> 4.85e-5, a **390x** improvement.

    So the flow keeps the library default and this variational guard pins the
    tolerance the stiff path can actually take.  Fixing the underlying
    ill-conditioning (per-block error weights for the variational lowering) is
    its own piece of work.
    """
    pytest.importorskip("tsdynamics._rust")
    spec = ts.analysis.lyapunov_spectrum(
        ts.systems.Oregonator(),
        final_time=6.0,
        dt=0.01,
        transient=2.0,
        ic=[1.0, 1.0, 1.0],
        rtol=1e-6,
        atol=1e-9,
    )
    assert spec.shape == (3,)
    assert np.all(np.isfinite(spec))
    assert spec[0] >= spec[1] >= spec[2]  # descending (QR order)


@pytest.mark.slow
def test_ode_family_delegates_to_tangent_engine() -> None:
    """Family ODE ``lyapunov_spectrum`` reproduces the literature Lorenz spectrum.

    The family method *is* ``TangentSystem(self).lyapunov_spectrum`` on the engine
    variational path; ``final_time`` is long enough that the finite-time estimate
    has converged to the canonical Lorenz spectrum ``[0.906, 0, -14.57]``.
    """
    spec = ts.analysis.lyapunov_spectrum(
        ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]), final_time=240.0, dt=0.1, transient=40.0
    )
    # Leading exponent ≈ 0.906, middle ≈ 0, third ≈ -14.57 (Lorenz 1963 / Sprott).
    assert abs(spec[0] - 0.906) < 0.06
    assert abs(spec[1]) < 0.06
    assert abs(spec[2] + 14.57) < 0.6


@pytest.mark.slow
def test_backend_neutral_lorenz_spectrum_reference() -> None:
    """The engine variational path reproduces the Lorenz spectrum on the reference backend."""
    ref = ts.analysis.lyapunov_spectrum(
        TangentSystem(ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]), k=3, backend="reference"),
        final_time=120.0,
        dt=0.1,
        transient=40.0,
    )
    assert abs(ref[0] - 0.906) < 0.1
    assert abs(ref[1]) < 0.1
    assert abs(ref[2] + 14.57) < 1.0


# ---------------------------------------------------------------------------
# The variational lowering is memoised (stream perf/variational-tape-cache)
# ---------------------------------------------------------------------------


def _tape_fields_equal(a, b) -> bool:
    """Whether two lowered tapes are field-for-field identical (the wire contract)."""
    return (
        np.array_equal(a.ops, b.ops)
        and np.array_equal(a.a, b.a)
        and np.array_equal(a.b, b.b)
        and np.array_equal(a.imm, b.imm)
        and np.array_equal(a.outputs, b.outputs)
        and np.array_equal(a.jac_outputs, b.jac_outputs)
        and a.n_state == b.n_state
        and a.n_param == b.n_param
    )


def test_variational_tape_cache_hits_on_a_repeat_build() -> None:
    """A second build of the same (system, k) is served from the cache."""
    clear_tape_cache()
    sysm = LinOsc()
    build_variational_tape_cached(sysm, 2)
    first = tape_cache_stats()
    build_variational_tape_cached(sysm, 2)
    second = tape_cache_stats()
    assert second["hits"] == first["hits"] + 1
    assert second["misses"] == first["misses"]


def test_variational_tape_cache_k_change_is_a_miss() -> None:
    """A different number of deviation vectors is a structurally different tape.

    ``k`` sets how many blocks of ``dim`` tangent equations the extended system
    carries, so it MUST be a key part — serving a ``k=1`` tape to a ``k=2`` caller
    would hand the engine a state vector of the wrong length.
    """
    clear_tape_cache()
    sysm = LinOsc()
    t1 = build_variational_tape_cached(sysm, 1)
    after_first = tape_cache_stats()
    t2 = build_variational_tape_cached(sysm, 2)
    after_second = tape_cache_stats()
    assert after_second["misses"] == after_first["misses"] + 1
    assert after_second["hits"] == after_first["hits"]
    assert t1.n_state != t2.n_state  # dim*(k+1): 4 vs 6


def test_variational_tape_cache_control_param_change_is_a_hit() -> None:
    """A control-parameter change reuses the tape (parameters feed it live).

    This is the design the other ``lower_*_cached`` helpers share and the reason a
    Lyapunov parameter sweep stays cheap.
    """
    clear_tape_cache()
    build_variational_tape_cached(LinOsc(), 2)
    before = tape_cache_stats()
    build_variational_tape_cached(LinOsc().with_params(k=5.0), 2)
    after = tape_cache_stats()
    assert after["hits"] == before["hits"] + 1
    assert after["misses"] == before["misses"]


def test_variational_tape_cache_monkeypatched_kernel_is_a_miss(monkeypatch) -> None:
    """Redefining ``_equations`` invalidates the entry — never a stale tape."""
    clear_tape_cache()
    build_variational_tape_cached(LinOsc(), 2)
    before = tape_cache_stats()

    original = LinOsc.__dict__["_equations"].__func__

    def patched(y, t, k, c):
        return original(y, t, k, c)

    monkeypatch.setattr(LinOsc, "_equations", staticmethod(patched))
    build_variational_tape_cached(LinOsc(), 2)
    after = tape_cache_stats()
    assert after["misses"] == before["misses"] + 1
    assert after["hits"] == before["hits"]


def test_variational_cached_tape_equals_a_fresh_build() -> None:
    """The memoised tape is field-for-field the tape ``build_variational_tape`` makes."""
    clear_tape_cache()
    for sysm, k in ((LinOsc(), 2), (ts.systems.Lorenz(), 3), (ts.systems.Rossler(), 2)):
        fresh = build_variational_tape(sysm, k)
        cached = build_variational_tape_cached(sysm, k)
        assert _tape_fields_equal(fresh, cached), type(sysm).__name__


def test_variational_spectrum_is_identical_with_the_cache_disabled(monkeypatch) -> None:
    """``TSDYNAMICS_NO_TAPE_CACHE`` gives a value-identical spectrum (the bypass proof)."""
    pytest.importorskip("tsdynamics._rust")
    kw = dict(final_time=40.0, dt=0.1, transient=10.0, ic=[1.0, 1.0, 1.0])

    clear_tape_cache()
    cached = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), **kw)

    monkeypatch.setenv("TSDYNAMICS_NO_TAPE_CACHE", "1")
    clear_tape_cache()
    uncached = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), **kw)

    assert np.array_equal(cached, uncached)


def test_cached_variational_path_keeps_interp_equals_jit() -> None:
    """The memoised tape preserves the ``interp == jit`` bit-for-bit contract."""
    pytest.importorskip("tsdynamics._rust")
    clear_tape_cache()
    kw = dict(final_time=40.0, dt=0.1, transient=10.0, ic=[1.0, 1.0, 1.0])
    jit = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), backend="jit", **kw)
    interp = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), backend="interp", **kw)
    assert np.array_equal(jit, interp)


def test_repeat_ode_lyapunov_reuses_one_variational_tape() -> None:
    """Two ``lyapunov_spectrum`` calls build the variational tape exactly ONCE.

    ``ContinuousSystem.lyapunov_spectrum`` constructs a fresh ``TangentSystem`` per
    call, so before this stream every call re-lowered the extended tape (measured
    ~17 s for a 32-D field system).  Count the *uncached* builder to pin the fix.
    """
    pytest.importorskip("tsdynamics._rust")
    import tsdynamics.derived._variational as var_mod

    clear_tape_cache()
    calls = {"n": 0}
    real = var_mod.build_variational_tape

    def counting(system, k):
        calls["n"] += 1
        return real(system, k)

    kw = dict(final_time=20.0, dt=0.1, transient=5.0, ic=[1.0, 1.0, 1.0])
    original = var_mod.build_variational_tape
    var_mod.build_variational_tape = counting
    try:
        ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), **kw)
        ts.analysis.lyapunov_spectrum(ts.systems.Lorenz(), **kw)
        ts.analysis.lyapunov_spectrum(ts.systems.Lorenz().with_params(rho=29.0), **kw)
    finally:
        var_mod.build_variational_tape = original
    assert calls["n"] == 1, f"expected one variational lowering, got {calls['n']}"


def test_the_old_n_exp_spelling_raises_instead_of_being_swallowed() -> None:
    """``k=1`` used to return THREE exponents — the request vanished silently.

    The v4 glossary renamed the parameter ``n_exp`` -> ``k``, but the family
    methods kept the old name, so ``k=`` fell into ``**integrator_kwargs`` and was
    dropped.  When the free function was later fixed to take ``k``, the two doors
    disagreed: ``ts.analysis.lyapunov_spectrum(sys, k=1)`` gave one exponent and
    the bound ``ts.analysis.lyapunov_spectrum(sys, k=1)`` gave ``dim`` of them, with no error.
    A wrong count returned confidently is worse than a failure, so the old
    spelling now raises and names the replacement.
    """
    import pytest

    lz = ts.systems.Lorenz()
    assert len(ts.analysis.lyapunov_spectrum(lz, final_time=20.0, k=1)) == 1

    with pytest.raises(ts.errors.InvalidInputError, match="unexpected keyword argument"):
        ts.analysis.lyapunov_spectrum(
            lz, final_time=20.0, **{"n_exp": 1}
        )  # the OLD spelling, on purpose
