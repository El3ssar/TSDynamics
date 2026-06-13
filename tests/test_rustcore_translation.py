"""
Rust-core *tape compiler* coverage — runs without the ``tsdynamics-core`` extra.

``compile_tape`` only needs SymEngine + the jitcode symbols, so this guards the
symbolic→tape lowering for the whole ODE catalogue on every CI run, regardless
of whether the Rust accelerator is installed.  The numeric cross-validation
against the symbolic RHS (and the RK4/ensemble kernels) lives in
``test_rustcore.py``, which needs the compiled crate and is skipped when absent.
"""

from __future__ import annotations

import numpy as np

from tsdynamics import registry
from tsdynamics.backends.rustcore import TapeCompileError, compile_tape


def test_every_ode_compiles_to_tape(ode_entry) -> None:
    """Every built-in ODE must lower to a well-formed instruction tape."""
    try:
        tape = compile_tape(ode_entry.cls())
    except TapeCompileError as exc:
        raise AssertionError(f"{ode_entry.name}: no longer compiles to a tape — {exc}") from exc

    n = tape.ops.size
    # Parallel instruction arrays agree in length.
    assert tape.a.size == n and tape.b.size == n and tape.imm.size == n
    # One output register per state component.
    assert tape.outputs.size == ode_entry.cls().dim == tape.n_state
    # Every register reference points to an already-emitted instruction.
    assert np.all(tape.outputs < n) and np.all(tape.outputs >= 0)
    # control inputs = the non-structural params (structural ones are folded
    # to constants in the tape, mirroring the DiffSL backend).
    cls = ode_entry.cls
    structural = getattr(cls, "_structural_params", frozenset())
    expected = [k for k in cls().params if k not in structural]
    assert list(tape.control_names) == expected


def test_tape_coverage_is_total() -> None:
    """The whole ODE catalogue lowers to a tape — the Rust-core coverage fact."""
    failed = []
    for e in registry.all_systems(family="ode"):
        try:
            compile_tape(e.cls())
        except Exception as exc:  # noqa: BLE001 — record, don't abort
            failed.append((e.name, str(exc).splitlines()[0][:60]))
    assert not failed, f"{len(failed)} ODE systems no longer compile to a tape: {failed}"


def test_every_ode_jacobian_compiles_to_tape(ode_entry) -> None:
    """The analytic Jacobian must lower too (stiff solver) — incl. abs/sign systems."""
    cls = ode_entry.cls
    try:
        tape = compile_tape(cls(), with_jacobian=True)
    except TapeCompileError as exc:
        raise AssertionError(
            f"{ode_entry.name}: Jacobian no longer lowers to a tape — {exc}"
        ) from exc
    dim = cls().dim
    # One register per Jacobian entry, row-major dim×dim.
    assert tape.jac_outputs.size == dim * dim
    n = tape.ops.size
    assert tape.jac_outputs.size == 0 or (
        tape.jac_outputs.max() < n and tape.jac_outputs.min() >= 0
    )


def test_jacobian_coverage_is_total() -> None:
    """Every ODE's analytic Jacobian lowers — abs/sign derivatives resolved a.e."""
    failed = []
    for e in registry.all_systems(family="ode"):
        try:
            compile_tape(e.cls(), with_jacobian=True)
        except Exception as exc:  # noqa: BLE001 — record, don't abort
            failed.append((e.name, str(exc).splitlines()[0][:60]))
    assert not failed, f"{len(failed)} ODE Jacobians no longer compile to a tape: {failed}"
