"""N2 regression tests for ODE IR lowering + Rust RHS + Rust DP8 goldens.

Four guarantees:

1. :func:`test_lower_ode_succeeds` — every built-in ``ContinuousSystem``
   lowers via :func:`tsdynamics.base._ode_lowering.lower_ode_to_ir`
   without raising :class:`NotLowerableError`.
2. :func:`test_ode_rhs_matches_symengine` — for each system, evaluate
   the IR-compiled RHS at a batch of random ``(t, y)`` samples and
   compare against the SymEngine-traced RHS (``Lambdify``).  This is
   the strongest functional guarantee available *without* a stepper —
   it catches every opcode-level lowering bug.
3. :func:`test_golden_ode_loadable` — golden trajectories under
   ``tests/native/regression/ode/`` are present, loadable, and non-trivial.
4. :func:`test_golden_trajectory_matches_rust_dp8` — each golden was produced
   by ``ContinuousSystem.integrate(..., method="dop853")``, which maps to the
   Rust **DP8** driver; recomputing with ``method="DP8"`` must recover the
   stored ``(t, y)`` on the uniform grid.

``Duffing``, ``SprottD``, and ``SprottI`` use fixed golden ICs (see
``scripts/generate_ode_goldens.py``) because the catalogue RNG seed lands
outside a viable basin for those systems.

The RHS comparison uses SymEngine's ``Lambdify`` rather than the
JiTCODE-compiled ``.so`` because the latter does not expose its
generated f() symbol cleanly.  Both ``Lambdify`` and our IR lowering
operate on the same SymEngine expression tree returned by
``cls._equations``, so an exact match between them validates the
lowering walker + bytecode encoder + Rust evaluator end-to-end.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import symengine as se

from tsdynamics._native import eval_ode_rhs, eval_ode_rhs_batch
from tsdynamics.base._ir import NotLowerableError
from tsdynamics.base._ode_lowering import lower_ode_to_ir

# Reuse the canonical system list from test_ode_systems so we never drift.
sys.path.insert(0, str(Path(__file__).parent))
from test_ode_systems import ALL_ODE_SYSTEMS  # noqa: E402

# Variable-dim systems need a fixed config — the same one
# ``scripts/generate_ode_goldens.py`` pins, so goldens and lowered IR
# describe the same system.
_VARIABLE_DIM_FACTORIES = {
    "Lorenz96": lambda: importlib.import_module("tsdynamics").Lorenz96(N=8),
    "KuramotoSivashinsky": lambda: importlib.import_module("tsdynamics").KuramotoSivashinsky(
        N=8, L=8.0
    ),
    "MultiChua": lambda: importlib.import_module("tsdynamics").MultiChua(n_circuits=2),
}

GOLDEN_DIR = Path(__file__).parent / "native" / "regression" / "ode"

# Systems with no integrator-friendly random IC; goldens are skipped by
# ``scripts/generate_ode_goldens.py`` for these.  The lowering + RHS
# checks still run because they don't integrate.
GOLDEN_SKIP: frozenset[str] = frozenset()

# Rust DP8 replay of committed goldens — comfortably inside FP noise after a
# full regenerate on the machine that committed the `.npz` files.
GOLDEN_TRAJ_RTOL = 1e-9
GOLDEN_TRAJ_ATOL = 1e-11

# Tight comparison tolerance for IR vs SymEngine.  Both pipelines walk
# the same expression tree; the only sources of disagreement are
# differing floating-point evaluation orders (associativity inside
# variadic ``Add`` / ``Mul``) and the resolved ``exp`` / ``cos`` intrinsic
# implementations.  ``1e-9`` is conservative enough to cover both.
RHS_RTOL = 1e-9
RHS_ATOL = 1e-12

# Systems whose RHS produces values with very large dynamic range
# (exponentials at non-trivial state) and therefore need looser
# absolute tolerance.  ``JerkCircuit`` is the canonical case: the
# ``exp(y / y0)`` term reaches ~5e7 at random ICs, so ~1e-15 relative
# error becomes ~1e-7 absolute.  All other systems honour RHS_ATOL.
LARGE_DYNAMIC_RANGE: dict[str, float] = {
    "JerkCircuit": 1e-6,
}

# Per-class RNG seed so failure messages are reproducible.
RNG_SEED = 1234


# ---------------------------------------------------------------------------
# Per-system helpers — kept module-level so they're easy to import for
# triaging from a REPL.
# ---------------------------------------------------------------------------


def _instantiate(module_path: str, class_name: str):
    factory = _VARIABLE_DIM_FACTORIES.get(class_name)
    if factory is not None:
        return factory()
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls()


def _build_symengine_rhs(inst):
    """Build a callable ``(t, y, params) -> rhs`` from the SymEngine trace.

    Mirrors the placeholder construction in
    :func:`tsdynamics.base._ode_lowering.lower_ode_to_ir` so both
    pipelines see the exact same symbolic expression tree before they
    diverge.  Structural parameters are baked in literally.
    """
    cls = type(inst)
    structural = cls._structural_params
    dim = inst.dim
    var_syms = [se.Symbol(f"_v{i}") for i in range(dim)]
    t_sym = se.Symbol("_t")
    nonstructural = [k for k in inst.params if k not in structural]
    par_syms = {name: se.Symbol(f"_p_{name}") for name in nonstructural}

    kwargs = {
        name: (value if name in structural else par_syms[name])
        for name, value in inst.params.items()
    }

    def y_accessor(i: int) -> se.Symbol:
        return var_syms[i]

    try:
        rhs = cls._equations(y_accessor, t_sym, **kwargs)
    except TypeError:
        rhs = cls._equations(y_accessor, t_sym, *kwargs.values())
    rhs = list(rhs)

    # Lambdify needs explicit input symbol order: (t, v0..vD-1, p0..pP-1).
    args = [t_sym, *var_syms, *par_syms.values()]
    lam = se.Lambdify(args, rhs, real=True)
    return lam, nonstructural


def _sample_ys(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` random states.  ``U[-1, 1)^dim`` is wide enough to
    exercise every branch of every system's RHS but avoids the
    extreme values where ``exp`` / ``log`` overflow.
    """
    return rng.uniform(-1.0, 1.0, size=(n, dim))


# ---------------------------------------------------------------------------
# Parametrisation — single list keeps test reports easy to scan.
# ---------------------------------------------------------------------------

_IDS = [name for _, name in ALL_ODE_SYSTEMS]


@pytest.mark.parametrize(("module_path", "class_name"), ALL_ODE_SYSTEMS, ids=_IDS)
def test_lower_ode_succeeds(module_path: str, class_name: str) -> None:
    """Every built-in ODE must lower to IR without raising."""
    inst = _instantiate(module_path, class_name)
    cls = type(inst)
    try:
        compiled = lower_ode_to_ir(
            cls,
            dim=inst.dim,
            params=dict(inst.params),
            structural_params=cls._structural_params,
        )
    except NotLowerableError as exc:
        pytest.fail(f"{class_name}: lowering failed — {exc}")

    assert compiled.dim == inst.dim
    nonstructural = [k for k in inst.params if k not in cls._structural_params]
    assert compiled.n_params == len(nonstructural)
    assert len(compiled.bytecode) > 0
    assert compiled.param_names == tuple(nonstructural)


@pytest.mark.parametrize(("module_path", "class_name"), ALL_ODE_SYSTEMS, ids=_IDS)
def test_ode_rhs_matches_symengine(module_path: str, class_name: str) -> None:
    """IR-evaluated RHS must match SymEngine ``Lambdify`` on random samples.

    This is the load-bearing N2.a guarantee — it certifies the entire
    lowering pipeline (walker, encoder, decoder, Rust evaluator) for
    every supported opcode.  Single-sample and batched paths share
    the same Rust kernel, so we exercise both to catch any shape /
    striding bug.
    """
    inst = _instantiate(module_path, class_name)
    cls = type(inst)
    compiled = lower_ode_to_ir(
        cls,
        dim=inst.dim,
        params=dict(inst.params),
        structural_params=cls._structural_params,
    )

    lam, nonstructural = _build_symengine_rhs(inst)
    param_vals = np.array([inst.params[k] for k in nonstructural], dtype=float)
    atol = LARGE_DYNAMIC_RANGE.get(class_name, RHS_ATOL)

    rng = np.random.default_rng(RNG_SEED + hash(class_name) % 2**16)
    n_samples = 16
    ys = _sample_ys(inst.dim, n_samples, rng)
    ts = rng.uniform(-5.0, 5.0, size=n_samples).astype(np.float64)

    se_rhs = np.array(
        [lam(np.concatenate(([t], y, param_vals))) for t, y in zip(ts, ys, strict=True)]
    )

    rust_rhs = eval_ode_rhs_batch(compiled.bytecode, ts, ys, param_vals)
    assert rust_rhs.shape == se_rhs.shape == (n_samples, inst.dim)

    # NaN handling: PowF / log / sqrt on out-of-domain inputs produce NaN
    # on both sides (e.g. WindmiReduced's ``v ** (1/2)`` for negative
    # samples).  Accept NaN matching but reject any NaN that appears on
    # only one side — that would indicate a true lowering bug.
    se_nan = ~np.isfinite(se_rhs)
    rust_nan = ~np.isfinite(rust_rhs)
    assert np.array_equal(se_nan, rust_nan), (
        f"{class_name}: NaN/inf pattern differs between SymEngine and IR "
        f"(SE nan @ {np.argwhere(se_nan).tolist()}, "
        f"IR nan @ {np.argwhere(rust_nan).tolist()})"
    )
    if se_nan.all():
        # Both sides entirely NaN — trivially consistent but the test
        # has no real assertion value.  Surface this so the maintainer
        # can pick a smaller IC range for that system.
        pytest.skip(f"{class_name}: every sampled (t, y) gives NaN RHS")

    finite_mask = ~se_nan
    diff = np.abs(rust_rhs[finite_mask] - se_rhs[finite_mask])
    max_err = float(diff.max()) if diff.size else 0.0
    assert np.allclose(rust_rhs[finite_mask], se_rhs[finite_mask], rtol=RHS_RTOL, atol=atol), (
        f"{class_name}: IR vs SymEngine RHS mismatch (max abs={max_err:.3e}, "
        f"rtol={RHS_RTOL}, atol={atol})"
    )

    # Single-sample path: same evaluator under the hood but different
    # PyO3 trampoline — guard against per-sample regressions.  Pick the
    # first finite sample so the comparison is meaningful.
    first_finite = int(np.argmax(finite_mask.all(axis=1)))
    one = eval_ode_rhs(compiled.bytecode, float(ts[first_finite]), ys[first_finite], param_vals)
    assert one.shape == (inst.dim,)
    if finite_mask[first_finite].all():
        assert np.allclose(one, rust_rhs[first_finite], rtol=0.0, atol=1e-15), (
            f"{class_name}: single-sample evaluator disagrees with batched path"
        )


# ---------------------------------------------------------------------------
# Golden-file presence + sanity, then trajectory replay vs Rust DP8.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("module_path", "class_name"), ALL_ODE_SYSTEMS, ids=_IDS)
def test_golden_ode_loadable(module_path: str, class_name: str) -> None:
    if class_name in GOLDEN_SKIP:
        pytest.skip(f"{class_name}: golden skipped by design (see GOLDEN_SKIP)")

    path = GOLDEN_DIR / f"{class_name}.npz"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")

    g = np.load(path, allow_pickle=True)
    t = g["t"]
    y = g["y"]
    assert t.ndim == 1
    assert y.ndim == 2
    assert y.shape[0] == t.shape[0]
    assert np.all(np.isfinite(y))
    assert np.std(y) > 0, f"{class_name}: golden trajectory is constant"

    inst = _instantiate(module_path, class_name)
    assert y.shape[1] == inst.dim

    # Required metadata for the N2.b stepper comparison.
    assert "ic" in g.files
    assert "params" in g.files
    assert "param_names" in g.files
    assert "structural_params" in g.files
    assert "method" in g.files
    assert "rtol" in g.files
    assert "atol" in g.files

    # Parameter names line up with the lowered IR's order.
    expected_names = [k for k in inst.params if k not in type(inst)._structural_params]
    saved_names = [str(n) for n in g["param_names"]]
    # Goldens store *all* params (structural + non-structural); the
    # lowered IR uses only the non-structural ones, so the saved order
    # must be a superset that preserves the original insertion order.
    saved_nonstructural = [n for n in saved_names if n in expected_names]
    assert saved_nonstructural == expected_names, (
        f"{class_name}: param order drift (saved={saved_nonstructural}, expected={expected_names})"
    )


@pytest.mark.parametrize(("module_path", "class_name"), ALL_ODE_SYSTEMS, ids=_IDS)
def test_golden_trajectory_matches_rust_dp8(module_path: str, class_name: str) -> None:
    """Replay each golden with Rust DP8 and match stored trajectory."""
    if class_name in GOLDEN_SKIP:
        pytest.skip(f"{class_name}: no golden — random IC unstable (see GOLDEN_SKIP)")

    path = GOLDEN_DIR / f"{class_name}.npz"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")

    g = np.load(path, allow_pickle=True)
    inst = _instantiate(module_path, class_name)
    for name, val in zip(g["param_names"], g["params"], strict=True):
        inst.params[str(name)] = float(val)

    ic = np.asarray(g["ic"], dtype=float)
    traj = inst.integrate(
        final_time=float(g["final_time"]),
        dt=float(g["dt"]),
        ic=ic,
        method="DP8",
        rtol=float(g["rtol"]),
        atol=float(g["atol"]),
    )
    np.testing.assert_allclose(traj.t, g["t"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(traj.y, g["y"], rtol=GOLDEN_TRAJ_RTOL, atol=GOLDEN_TRAJ_ATOL)
