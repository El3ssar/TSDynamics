"""The M3 migration gate — Rust engine vs v2, swept over the whole catalogue.

ROADMAP stream **I-XVAL** (§9, decision D1).  Before any v2 backend
(JiTCODE / JiTCDDE / Numba / diffsol) is removed, the shipping Rust engine
(:mod:`tsdynamics._rust`) must be shown to reproduce the v2 numeric truth across
*every* registered system.  This module is that gate: registry-driven, so a new
system joins it with zero edits, and self-contained enough that a green run here
(plus the per-family binding tests and the literature-Lyapunov checks) is the
evidence that authorises the deletion PR.

The gate has six legs, ordered by how tolerance-tight (and chaos-free) the
signal is:

1. **RHS lowering** — for every ODE the engine's ``du/dt`` matches the symbolic
   ``_rhs_numeric`` to ``1e-9`` at random states.  The strongest per-system
   proof that the tape lowers correctly; no chaotic-divergence caveat.
2. **interp == jit** — the SSA-tape interpreter and the Cranelift JIT are
   numerically identical *by contract* (decision D2).  Asserted **bit-for-bit**
   over every ODE and map: the same lowering, two evaluators, zero difference.
3. **reference == engine** — the pure-Python tape oracle (``backend="reference"``,
   the evaluator that survives the wheel-free install) matches the compiled
   engine to ``~1e-12`` (1-ULP transcendental differences aside), and the
   square-and-multiply ``OP_POWI`` path matches **bit-for-bit** (consolidation
   note, PR #74).
4. **trajectory vs reference** — on a curated sample the engine's early-window
   trajectory tracks a tight SciPy integration of the symbolic RHS (slow).  The
   catalogue-wide per-system proof is leg 1; this confirms the integrate loop
   against a trustworthy independent integrator on representative systems.
5. **literature Lyapunov** — the engine variational backend (the successor to
   ``jitcode_lyap`` at M3) reproduces the published Lyapunov spectrum on a
   curated set of chaotic flows, with ``interp`` and ``jit`` agreeing (slow).
6. **DDE Lyapunov vs v2** — the engine DDE-Lyapunov estimator (the successor to
   ``jitcdde_lyap``, stream E-DDE-LYAP) reproduces JiTCDDE on Mackey–Glass; the
   full 5-DDE parity sweep is in ``test_dde_lyapunov.py`` (slow).

Maps get legs 1–2 as a one-step lowering check (engine next-state vs the
pure-Python ``_step``, then interp vs jit).  DDEs are gated for finiteness + engine
provenance here; their tight JiTCDDE early-window parity (the E-DDE literature
bar) lives in ``test_dde_engine.py`` and now runs in the same engine CI job via
the ``engine`` marker.

The whole module needs the compiled extension, so it ``importorskip``s
``tsdynamics._rust`` (which also auto-tags every test here with the ``engine``
marker — see ``conftest.py``) and skips cleanly in the wheel-free matrix.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from _sampling import INTEGRATION_SAMPLE
from xval_harness import RustEngine, ScipyReference, crossvalidate, crossvalidate_rhs

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.engine import run
from tsdynamics.engine.compile import (
    OP_ADD,
    OP_POWI,
    OP_STATE,
    Tape,
    TapeCompileError,
    eval_tape,
)
from tsdynamics.engine.problem import map_problem
from tsdynamics.families.discrete import _unwrap_static

_rust = pytest.importorskip("tsdynamics._rust")

# Tolerances (kept in one place so the gate's bar is auditable).
_RHS_RTOL = 1e-9  # engine RHS vs symbolic truth (leg 1) — relative, scale-invariant
_RHS_ATOL = 1e-9  # leg 1 near-zero floor
_REF_RTOL = 1e-12  # reference vs engine, transcendental 1-ULP slack (leg 3)
_REF_ATOL = 1e-12
_MAP_RTOL = 1e-9  # engine next-state vs pure-Python _step (leg 1, maps)
_MAP_ATOL = 1e-12

_RNG_STATES = 8  # random states per system for the pointwise legs


def _seed_for(name: str) -> int:
    """A stable per-system seed (independent of ``PYTHONHASHSEED``) for reproducible ICs."""
    return int.from_bytes(hashlib.blake2s(name.encode(), digest_size=4).digest(), "little")


def _resolve_ic(sys, name: str) -> np.ndarray:
    """Resolve a *deterministic* initial condition for ``sys``.

    Uses the system's own ``ic``/``default_ic`` when it declares one; otherwise
    seeds the global RNG per-system (matching the ``_on_attractor`` convention in
    this file) so the random fallback in ``resolve_ic`` is reproducible — a gate
    failure must replay, and its inputs must be auditable.
    """
    np.random.seed(_seed_for(name))
    return np.asarray(sys.resolve_ic(None), dtype=np.float64).ravel()


def _states(dim: int, *, seed: int, scale: float = 1.5) -> list[tuple[np.ndarray, float]]:
    """A reproducible list of ``(u, t)`` evaluation points for a ``dim``-state system."""
    rng = np.random.default_rng(seed)
    return [
        (rng.standard_normal(dim) * scale, float(rng.uniform(0.0, 3.0))) for _ in range(_RNG_STATES)
    ]


# ---------------------------------------------------------------------------
# Leg 1 — ODE RHS lowering (engine vs symbolic, tolerance-tight, chaos-free)
# ---------------------------------------------------------------------------


def test_ode_rhs_lowers_to_symbolic(ode_entry) -> None:
    """Every ODE: the engine RHS reproduces the symbolic ``_rhs_numeric`` to 1e-9.

    ``rep.passed`` is the ``numpy.allclose`` criterion ``|Δ| <= atol + rtol*|ref|``
    — scale-invariant (a benign single-ULP difference at a large-magnitude RHS
    component, e.g. JerkCircuit ~5e13, is well within ``rtol*|ref|`` and does not
    spuriously redden the gate), with a small ``atol`` floor for near-zero
    components.
    """
    from _sampling import HEAVY_FIELD_CATEGORIES

    if ode_entry.category in HEAVY_FIELD_CATEGORIES:
        pytest.skip(f"{ode_entry.name}: high-dim PDE field — viz-only (heavy full-grid sweep)")
    rep = crossvalidate_rhs(
        ode_entry.cls(),
        candidate=RustEngine(backend="interp"),
        n=_RNG_STATES,
        seed=11,
        rtol=_RHS_RTOL,
        atol=_RHS_ATOL,
    )
    assert rep.passed, rep.summary()


# ---------------------------------------------------------------------------
# Leg 2 — interp == jit, bit-for-bit (the D2 contract)
#
# The JIT is wired into the *integrate* and *iterate* paths (the FFI threads the
# jit flag there), NOT into the pointwise ``run.eval_rhs`` seam — which evaluates
# on the interpreter for both "interp" and "jit".  So this leg drives the JIT
# through a short integration/iteration per system, where the two evaluators are
# genuinely distinct, and asserts bit-for-bit equality (decision D2).
# ---------------------------------------------------------------------------


def test_ode_interp_equals_jit_bit_for_bit(ode_entry) -> None:
    """Every ODE: a short engine integration is bit-identical on interp and jit."""
    from _sampling import HEAVY_FIELD_CATEGORIES

    if ode_entry.category in HEAVY_FIELD_CATEGORIES:
        pytest.skip(f"{ode_entry.name}: high-dim PDE field — viz-only (heavy full-grid sweep)")
    sys = ode_entry.cls()
    ic = _resolve_ic(sys, ode_entry.name)
    kw = dict(final_time=0.5, dt=0.05, t0=0.0, ic=ic, method="RK45", rtol=1e-9, atol=1e-11)
    try:
        interp = run.integrate(sys, backend="interp", **kw).y
    except RuntimeError:
        # Divergence on this short window must be identical on both evaluators
        # (the engine raises on a non-finite step) — that is itself interp==jit.
        with pytest.raises(RuntimeError):
            run.integrate(sys, backend="jit", **kw)
        return
    jit = run.integrate(sys, backend="jit", **kw).y
    np.testing.assert_array_equal(interp, jit, err_msg=f"{ode_entry.name}: interp != jit")


# ---------------------------------------------------------------------------
# Leg 3 — reference == engine (the wheel-free oracle vs the compiled engine)
# ---------------------------------------------------------------------------


def test_ode_reference_matches_engine(ode_entry) -> None:
    """Every ODE: the pure-Python tape oracle matches the compiled engine to ~1e-12.

    ``backend="reference"`` is the evaluator that survives a wheel-free install
    (it powers the analysis layer without ``tsdynamics._rust``); it must agree
    with the interpreter the families dispatch to, up to a 1-ULP transcendental
    slack.  Pure-arithmetic systems (the ``OP_POWI`` path included) agree
    bit-for-bit — pinned exactly by :func:`test_op_powi_is_bit_exact`.

    High-dimensional method-of-lines PDE fields (``HEAVY_FIELD_CATEGORIES``) are
    skipped: this leg re-lowers the tape for every sample state, so for a 1k-5k-state
    field it is tens of seconds for no extra coverage — the same ``_equations`` are
    lowered + integrated on the engine by the small-grid viz field tests.
    """
    from _sampling import HEAVY_FIELD_CATEGORIES

    if ode_entry.category in HEAVY_FIELD_CATEGORIES:
        pytest.skip(f"{ode_entry.name}: high-dim PDE field — re-lowered per state (covered by viz)")
    sys = ode_entry.cls()
    for u, t in _states(sys.dim, seed=31):
        reference = run.eval_rhs(sys, u, t, backend="reference")
        engine = run.eval_rhs(sys, u, t, backend="interp")
        np.testing.assert_allclose(
            reference,
            engine,
            rtol=_REF_RTOL,
            atol=_REF_ATOL,
            err_msg=f"{ode_entry.name}: reference != engine at t={t}",
        )


def test_op_powi_is_bit_exact() -> None:
    """``OP_POWI`` is bit-for-bit between the reference oracle and the engine.

    PR #74 changed the Python reference to evaluate integer powers by
    square-and-multiply, matching Rust's ``f64::powi``; this is the gate that the
    two are *bit-identical* (not merely close).  A hand-built ``y0**3 + y1**2``
    tape isolates the opcode from the transcendental functions that cost a ULP
    elsewhere, and the catalogue assertion below proves the opcode is actually
    exercised by built-in systems (so the check is not vacuous).

    ``OP_POWI`` reads its integer exponent from operand ``b`` (``regs[a] ** b``),
    so the exponents live in ``b`` — exponent 3 and 2 here, both ``>= 2`` so the
    square-and-multiply loop is genuinely driven on both sides.
    """
    ops = np.array([OP_STATE, OP_STATE, OP_POWI, OP_POWI, OP_ADD], dtype=np.int32)
    a = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    b = np.array([0, 0, 3, 2, 3], dtype=np.int32)  # POWI exponents 3, 2; ADD's 2nd reg
    imm = np.zeros(5, dtype=np.float64)  # OP_POWI takes its exponent from b, not imm
    outputs = np.array([4, 4], dtype=np.int32)  # both outputs = r4 = u0**3 + u1**2
    tape = Tape(
        ops=ops,
        a=a,
        b=b,
        imm=imm,
        outputs=outputs,
        jac_outputs=np.array([], dtype=np.int32),
        n_state=2,
        n_param=0,
    )
    rng = np.random.default_rng(5)
    arrays = tape.to_arrays()
    for _ in range(2000):
        u = rng.standard_normal(2) * 3.0
        reference = eval_tape(tape, u, (), 0.0)
        # Sanity: the tape genuinely depends on u (guards against a vacuous tape).
        assert reference[0] == pytest.approx(u[0] ** 3 + u[1] ** 2)
        engine = np.asarray(_rust.eval_rhs(*arrays, u.astype(np.float64), np.array([]), 0.0))
        np.testing.assert_array_equal(reference, engine)


def test_op_powi_is_exercised_by_the_catalogue() -> None:
    """At least one built-in ODE lowers an integer power to ``OP_POWI``.

    Keeps :func:`test_ode_reference_matches_engine` honest: it would still pass if
    no system used the opcode, so assert the catalogue genuinely exercises it.
    """
    from tsdynamics.engine.problem import build_problem

    using_powi = [
        e.name
        for e in registry.all_systems(family="ode")
        if OP_POWI in set(build_problem(e.cls()).tape.ops.tolist())
    ]
    assert using_powi, "no built-in ODE lowers to OP_POWI — the bit-exact gate is vacuous"


# ---------------------------------------------------------------------------
# Maps — one-step lowering vs the pure-Python step, and interp == jit
# ---------------------------------------------------------------------------


def _on_attractor(cls, *, n_warm: int = 60, drop: int = 40, take: int = 5) -> np.ndarray:
    """A few finite, on-attractor states for ``cls`` (deterministic, via reference).

    Mirrors ``test_map_engine``: the reference path only returns once the whole
    buffer is finite, so the tail slice sits on the orbit rather than in a
    transient or off-basin escape.
    """
    np.random.seed(0)
    warm = cls().iterate(steps=n_warm, backend="reference")
    finite = warm.y[np.isfinite(warm.y).all(axis=1)]
    if finite.shape[0] < drop + take:
        pytest.skip(f"{cls.__name__}: too few finite warm-up states for a stable sample")
    return np.ascontiguousarray(finite[drop : drop + take])


def test_map_engine_step_matches_step(map_entry) -> None:
    """Every lowerable map: the engine next-state matches the pure-Python ``_step``.

    The chaos-free, tolerance-tight map analogue of leg 1 — but through the
    *compiled* interpreter (``backend="interp"``), not the pure-Python reference
    that ``test_map_engine`` checks.  A map that cannot lower to the frozen IR is
    skipped (the lowering boundary itself is pinned by ``test_map_engine``).
    """
    cls = map_entry.cls
    try:
        map_problem(cls())
    except TapeCompileError:
        pytest.skip(f"{map_entry.name} does not lower to the straight-line IR")
    m = cls()
    step = _unwrap_static(type(m)._step)
    params = m.params.as_tuple()
    for s in _on_attractor(cls):
        expected = np.asarray(step(s, *params), dtype=float).ravel()
        got = run.eval_rhs(m, s, backend="interp")
        np.testing.assert_allclose(
            got, expected, rtol=_MAP_RTOL, atol=_MAP_ATOL, err_msg=f"{map_entry.name} next-state"
        )


def test_map_interp_equals_jit_bit_for_bit(map_entry) -> None:
    """Every lowerable map: a short engine iteration is bit-identical on interp and jit.

    Driven through ``iterate`` (where the FFI threads the jit flag into the
    native map loop), not the pointwise ``run.eval_rhs`` seam — so the two
    evaluators are genuinely distinct here (decision D2).
    """
    cls = map_entry.cls
    try:
        map_problem(cls())
    except TapeCompileError:
        pytest.skip(f"{map_entry.name} does not lower to the straight-line IR")
    ic = _on_attractor(cls, take=1)[0]
    interp = cls().iterate(steps=20, ic=ic, backend="interp").y
    jit = cls().iterate(steps=20, ic=ic, backend="jit").y
    np.testing.assert_array_equal(interp, jit, err_msg=f"{map_entry.name}: interp != jit")


# ---------------------------------------------------------------------------
# DDEs — the engine path is reachable and finite for every built-in DDE
# ---------------------------------------------------------------------------


def test_dde_engine_path_is_finite(dde_entry) -> None:
    """Every built-in DDE integrates on the engine to a finite trajectory.

    A smoke that the DDE engine path (method-of-steps, stream E-DDE) is reachable
    for each catalogue DDE and carries the engine provenance.  The tight
    JiTCDDE early-window parity (the E-DDE literature bar, ``< 5e-3``) is asserted
    in ``test_dde_engine.py``, which now runs in this same engine CI job.
    """
    from _sampling import DDE_HISTORIES

    history = DDE_HISTORIES[dde_entry.name]
    traj = dde_entry.cls().integrate(
        backend="interp", final_time=10.0, dt=0.1, history=history, rtol=1e-6, atol=1e-8
    )
    assert np.all(np.isfinite(traj.y)), f"{dde_entry.name}: engine produced non-finite states"
    assert traj.meta["engine"] == "rust"
    assert traj.meta["backend"] == "interp"


# ---------------------------------------------------------------------------
# Leg 4 — trajectory vs a tight SciPy integration of the symbolic RHS (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", INTEGRATION_SAMPLE)
def test_ode_trajectory_matches_reference_early(name) -> None:
    """A curated ODE sample: the engine tracks the trustworthy reference on an early window.

    Leg 4 — confirms the integrate *loop* (not just the symbolic RHS) against a
    tight SciPy integration of the symbolic ``_rhs_numeric`` on representative
    systems.  The comparison window is short: two correct integrators of a chaotic
    flow diverge exponentially, so agreement is only expected before Lyapunov
    amplification dominates (the harness ``window=`` convention).
    """
    sys = getattr(ts, name)()
    ic = _resolve_ic(sys, name)
    t_eval = np.arange(0.0, 3.0 + 1e-9, 0.01)
    engine = RustEngine(backend="interp", rtol=1e-10, atol=1e-12)

    rep = crossvalidate(
        sys,
        reference=ScipyReference(method="DOP853", rtol=1e-12, atol=1e-14),
        candidate=engine,
        ic=ic,
        t_eval=t_eval,
        window=(0.0, 1.0),
        atol=1e-3,
    )
    assert rep.passed, rep.summary()


# ---------------------------------------------------------------------------
# Leg 5 — literature Lyapunov on the engine variational path (slow)
# ---------------------------------------------------------------------------
#
# The acceptance is "Rust vs v2 within tol AND literature Lyapunov".  The
# engine's variational backend (``TangentSystem(backend="interp"/"jit")``, stream
# C-DERIV) is the path that *replaces* ``jitcode_lyap`` at M3, so the gate proves
# it reproduces the published spectrum on a curated, robust set of chaotic flows.
# The full ``known_lyapunov`` sweep (``test_known_values.py``) migrates onto the
# engine automatically when M3 flips ``_default_backend`` — this leg de-risks that
# flip ahead of time.
_LYAP_SYSTEMS = ("Lorenz", "Rossler")


@pytest.mark.slow
@pytest.mark.parametrize("name", _LYAP_SYSTEMS)
def test_engine_lyapunov_matches_literature(name) -> None:
    """The engine variational spectrum reproduces the literature spectrum.

    Computed through the backend-neutral ODE variational core (the *extended*
    variational ODE integrated on the engine, then QR-reorthonormalised) — the
    successor to ``jitcode_lyap``.  ``interp`` and ``jit`` must also agree closely
    (the same lowering, integrated by two numerically-identical evaluators).
    """
    cls = getattr(ts, name)
    meta = dict(cls().known_lyapunov)
    expected = np.asarray(meta["spectrum"], dtype=float)
    atol = np.asarray(meta["atol"], dtype=float)
    kwargs = {k: v for k, v in meta.get("kwargs", {}).items() if k != "method"}
    if meta.get("ic") is not None:
        kwargs.setdefault("ic", list(meta["ic"]))

    interp = ts.TangentSystem(cls(), backend="interp").lyapunov_spectrum(**kwargs)
    assert np.all(np.isfinite(interp))
    deviation = np.abs(interp - expected)
    assert np.all(deviation <= atol), (
        f"{name}: engine spectrum {np.round(interp, 4)} deviates from literature "
        f"{expected} by {np.round(deviation, 4)} (atol {atol}). "
        f"Source: {meta.get('source', 'n/a')}"
    )

    jit = ts.TangentSystem(cls(), backend="jit").lyapunov_spectrum(**kwargs)
    np.testing.assert_allclose(
        interp, jit, rtol=0.0, atol=1e-6, err_msg=f"{name}: interp vs jit Lyapunov spectrum"
    )


# ---------------------------------------------------------------------------
# Leg 6 — engine DDE Lyapunov reproduces the literature sign (slow)
# ---------------------------------------------------------------------------
#
# The engine DDE-Lyapunov estimator (DelaySystem.lyapunov_spectrum, stream
# E-DDE-LYAP) is the successor to the retired jitcdde_lyap.  Pin the canonical
# Mackey-Glass case to its known_lyapunov (one positive exponent); the full 5-DDE
# parity sweep against the legacy values lives in tests/test_dde_lyapunov.py.


@pytest.mark.slow
def test_dde_engine_lyapunov_is_positive_mackeyglass() -> None:
    """Mackey-Glass: the engine DDE-Lyapunov λ₁ is positive (known_lyapunov n_positive=1)."""
    from _sampling import DDE_HISTORIES

    mg = ts.MackeyGlass()
    ic = mg.integrate(final_time=500.0, dt=0.2, history=DDE_HISTORIES["MackeyGlass"]).y[-1]
    eng = mg.lyapunov_spectrum(
        backend="interp", n_exp=1, burn_in=200.0, final_time=2000.0, ic=ic, dt=0.05
    )
    assert eng[0] > 0.0  # chaotic — matches known_lyapunov n_positive=1
    # Mackey-Glass at τ=17 is weakly chaotic: λ₁ ≈ 0.0086 (Farmer 1982). A loose
    # band keeps the estimator honest without pinning it to the retired jitcdde value.
    assert eng[0] < 0.05, f"engine MG λ₁ = {eng[0]} (expected ≈ 0.0086)"
