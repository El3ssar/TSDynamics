"""Regression: Rust solver capabilities have a single source of truth.

Finding R12:R12-1.  Each solver kernel declares its :class:`Caps` *twice* — once
in the ``register_solver!`` literal (stored in ``SolverRegistration::caps``) and
once in the ``Solver::caps`` trait method — and the two copies are read by
*different* engine paths: the Jacobian / implicit-kind guards read the
*registered* copy (``crates/tsdyn-core/src/bridge/{marshal,dde,sde}.rs``) while
the dense-output / event path reads the *instance* method
(``crates/tsdyn-engine/src/event.rs``).  Before the fix nothing asserted the two
agree, so a kernel author who edited one and not the other (e.g. added
``.with_dense()`` to the registration but not to ``caps()``) produced a silent
correctness divergence — the engine builds a Jacobian-free tape off one
capability set yet drives event refinement off a *different* one.

The fix adds a consistency guard ``#[test]`` in
``crates/tsdyn-solvers/src/registry.rs`` that asserts, for every kernel linked
into the test binary, ``(reg.make)().caps() == reg.caps`` and
``(reg.make)().name() == reg.name`` — so any drift fails ``cargo test`` loudly.

This is a Rust work package: the substantive guard is the ``cargo test``
``#[test]`` above (verified to fail on an injected drift and pass otherwise).
These pytest assertions are a source-level mirror that fails on the pre-fix tree
(the guard and its contract doc did not exist) and passes after, with no compiled
engine required — keeping the regression in the fast Python tier too.
"""

from __future__ import annotations

import re
from pathlib import Path

_CRATE = Path(__file__).resolve().parents[1] / "crates" / "tsdyn-solvers" / "src"
_REGISTRY = _CRATE / "registry.rs"
_SOLVER = _CRATE / "solver.rs"

_GUARD_FN = "registered_caps_match_instance_caps"


def _registry_src() -> str:
    return _REGISTRY.read_text(encoding="utf-8")


def _solver_src() -> str:
    return _SOLVER.read_text(encoding="utf-8")


def test_registry_file_exists() -> None:
    """The guard lives in the solver registry module (no dangling path)."""
    assert _REGISTRY.is_file(), f"missing {_REGISTRY}"
    assert _SOLVER.is_file(), f"missing {_SOLVER}"


def test_consistency_guard_test_present() -> None:
    """A ``#[test]`` named for the caps single-source-of-truth guard exists.

    Pre-fix there was no test asserting the registered caps equal the instance
    ``caps()``; this names the exact guard the finding calls for.
    """
    src = _registry_src()
    assert f"fn {_GUARD_FN}(" in src, "consistency-guard #[test] is absent from registry.rs"
    # It must be a real test, not a stray helper.
    pattern = re.compile(r"#\[test\]\s*\n\s*fn\s+" + re.escape(_GUARD_FN) + r"\b")
    assert pattern.search(src), f"{_GUARD_FN} is not annotated as a #[test]"


def test_guard_compares_both_caps_and_name() -> None:
    """The guard checks both axes the finding names: ``caps()`` and ``name()``.

    A guard that compared only one would let the other drift silently.
    """
    src = _registry_src()
    # The body iterates registrations, building a fresh instance and comparing.
    assert "(reg.make)()" in src, "guard does not build a fresh instance via reg.make"
    assert ".caps()" in src, "guard does not read the instance caps()"
    assert ".name()" in src, "guard does not read the instance name()"
    # And it compares each against the registered copy.
    assert "reg.caps" in src, "guard does not compare against the registered caps"
    assert "reg.name" in src, "guard does not compare against the registered name"


def test_guard_is_non_vacuous() -> None:
    """A second test guards against a dead-stripped (empty) registry.

    Without it the caps guard would pass trivially if the linker dropped every
    kernel, hiding a real regression.
    """
    src = _registry_src()
    assert "registry_is_non_empty" in src, "missing non-empty-registry guard"


def test_solver_caps_contract_documents_single_source() -> None:
    """``Solver::caps`` doc states it must equal the registration caps.

    The contract is the human-facing half of the single-source-of-truth fix:
    a kernel author reading ``caps()`` is told the two copies must agree and that
    a test enforces it.
    """
    src = _solver_src()
    # Find the caps() doc + signature block.
    idx = src.index("fn caps(&self) -> Caps;")
    block = src[max(0, idx - 1200) : idx]
    assert "register_solver!" in block, "caps() doc does not reference register_solver!"
    assert _GUARD_FN in block, "caps() doc does not cite the enforcing test"
