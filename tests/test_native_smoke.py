"""Round-trip test for the Rust ``_smoke`` extension module.

If this import or assertion fails, the maturin build is broken — every other
Rust-backed milestone (R2+, N1+) depends on the same plumbing.
"""

from __future__ import annotations


def test_smoke_module_importable() -> None:
    from tsdynamics._native import _smoke  # noqa: F401


def test_add_one_returns_successor() -> None:
    from tsdynamics._native._smoke import add_one

    assert add_one(41) == 42
    assert add_one(-1) == 0
    assert add_one(0) == 1


def test_facade_reexports_add_one() -> None:
    from tsdynamics._native import add_one

    assert add_one(100) == 101
