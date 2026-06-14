"""Python end of the ``tsdyn-ir`` (stream F1) instruction-tape contract.

The Rust crate freezes the IR — opcodes, tape layout, validation — and validates
it against committed golden fixtures (``crates/tsdyn-ir/tests/``).  This module
guards the *Python* side of that contract, in the standard test environment (no
Rust accelerator needed, exactly like ``test_rustcore_translation.py``):

1. the tape emitter's opcode numbers still match the frozen wire values the Rust
   ``Op`` enum decodes;
2. each committed golden fixture is still current — its tape arrays match a fresh
   emission (exact integers) and its checkpoint values match today's symbolic
   right-hand side and Jacobian (to ~1e-12, tolerant of cross-platform ULP).

If (2) fails after an intentional emitter change, regenerate the fixtures::

    python crates/tsdyn-ir/tests/gen_fixtures.py
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GEN_PATH = _REPO_ROOT / "crates" / "tsdyn-ir" / "tests" / "gen_fixtures.py"


def _load_gen():
    spec = importlib.util.spec_from_file_location("tsdyn_ir_gen_fixtures", _GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_gen()


# ---------------------------------------------------------------------------
# 1. Opcode wire values — the FFI contract, pinned on the Python side.
# ---------------------------------------------------------------------------

# These integers are the frozen contract: the Rust `Op` enum decodes exactly
# them (see crates/tsdyn-ir/src/op.rs). Never renumber.
_FROZEN_STRUCTURAL = {
    "OP_CONST": 0,
    "OP_STATE": 1,
    "OP_PARAM": 2,
    "OP_TIME": 3,
    "OP_ADD": 10,
    "OP_SUB": 11,
    "OP_MUL": 12,
    "OP_DIV": 13,
    "OP_POW": 14,
    "OP_POWI": 15,
    "OP_NEG": 20,
    "OP_RECIP": 21,
}
_FROZEN_FUNCS = {
    "sin": 30,
    "cos": 31,
    "tan": 32,
    "exp": 33,
    "log": 34,
    "sqrt": 35,
    "Abs": 36,
    "sign": 37,
    "sinh": 38,
    "cosh": 39,
    "tanh": 40,
    "asin": 41,
    "acos": 42,
    "atan": 43,
    "asinh": 44,
    "acosh": 45,
    "atanh": 46,
}


def test_emitter_opcodes_match_the_frozen_wire_values():
    from tsdynamics.engine import rustcore

    for const_name, wire in _FROZEN_STRUCTURAL.items():
        assert getattr(rustcore, const_name) == wire, const_name
    assert rustcore._FUNC_OPS == _FROZEN_FUNCS
    assert rustcore._OP_SQRT == 35


# ---------------------------------------------------------------------------
# 2. Golden fixtures are still current (structure exact, values to tolerance).
# ---------------------------------------------------------------------------

_SYSTEMS = list(gen.FIXTURE_SYSTEMS)


@pytest.mark.parametrize("name", _SYSTEMS)
def test_committed_fixture_tape_structure_is_current(name):
    """A fresh emission must reproduce the committed tape arrays exactly."""
    import tsdynamics as ts
    from tsdynamics.engine.rustcore import compile_tape

    fx = gen.parse_fixture(gen.fixture_path(name).read_text())
    system = getattr(ts, name)()
    tape = compile_tape(system, with_jacobian=True)

    assert fx["system"] == name
    assert fx["n_state"] == tape.n_state
    assert fx["n_param"] == tape.n_param
    assert fx["dim"] == system.dim
    assert fx["ops"] == tape.ops.tolist()
    assert fx["a"] == tape.a.tolist()
    assert fx["b"] == tape.b.tolist()
    assert fx["outputs"] == tape.outputs.tolist()
    assert fx["jac_outputs"] == tape.jac_outputs.tolist()
    # imm are exact small constants (e.g. -1.0); compare exactly.
    assert fx["imm"] == tape.imm.tolist()


@pytest.mark.parametrize("name", _SYSTEMS)
def test_committed_fixture_values_match_symbolic_rhs(name):
    """Committed checkpoint deriv/jac must match today's symbolic engine."""
    import tsdynamics as ts
    from tsdynamics.engine.rustcore import compile_tape

    fx = gen.parse_fixture(gen.fixture_path(name).read_text())
    system = getattr(ts, name)()
    control_names = compile_tape(system).control_names
    expected_params = [float(system.params[k]) for k in control_names]
    np.testing.assert_allclose(fx["params"], expected_params, rtol=1e-12, atol=1e-12)

    rhs = system._rhs_numeric()
    assert fx["checkpoints"], f"{name}: fixture has no checkpoints"
    for cp in fx["checkpoints"]:
        u = np.asarray(cp["u"], dtype=float)
        t = cp["t"]
        np.testing.assert_allclose(rhs(u, t), cp["deriv"], rtol=1e-12, atol=1e-12)
        jac = np.asarray(system.jacobian(u, t), dtype=float).ravel()
        np.testing.assert_allclose(jac, cp["jac"], rtol=1e-12, atol=1e-12)
