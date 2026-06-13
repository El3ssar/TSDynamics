"""
Diffsol *translator* coverage — runs without the pydiffsol extra.

``to_diffsl`` only needs SymEngine + the jitcode symbols, so this guards the
SymEngine→DiffSL translation for the whole ODE catalogue on every CI run,
independently of whether the Rust solver itself is installed.  The numeric
cross-validation against JiTCODE lives in ``test_diffsol_backend.py`` (which
needs pydiffsol and is therefore skipped when the extra is absent).
"""

from __future__ import annotations

from tsdynamics import registry
from tsdynamics.engine.diffsol import DiffSLTranslationError, to_diffsl


def test_every_ode_translates_to_diffsl(ode_entry) -> None:
    """Every built-in ODE must translate to a well-formed DiffSL module."""
    try:
        code, control_names = to_diffsl(ode_entry.cls())
    except DiffSLTranslationError as exc:
        # A genuine "DiffSL can't express this" is allowed but must be visible:
        # fail loudly so the coverage number can't silently regress unnoticed.
        raise AssertionError(f"{ode_entry.name}: no longer translates to DiffSL — {exc}") from exc
    assert "in_i {" in code and "u_i {" in code and "F_i {" in code
    # control inputs = the non-structural params (structural ones, e.g.
    # Lorenz96's N, are baked into the generated module, not solve-time inputs).
    cls = ode_entry.cls
    structural = getattr(cls, "_structural_params", frozenset())
    expected = [k for k in cls().params if k not in structural]
    assert list(control_names) == expected


def test_translation_coverage_is_total() -> None:
    """The whole ODE catalogue translates — the headline Phase-2 coverage fact."""
    failed = []
    for e in registry.all_systems(family="ode"):
        try:
            to_diffsl(e.cls())
        except Exception as exc:  # noqa: BLE001 — record, don't abort
            failed.append((e.name, str(exc).splitlines()[0][:60]))
    assert not failed, f"{len(failed)} ODE systems no longer translate: {failed}"
