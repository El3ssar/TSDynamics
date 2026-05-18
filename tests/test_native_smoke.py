"""Round-trip test for the Rust ``_core`` extension module.

If this import or assertion fails, the maturin build is broken — every
Rust-backed milestone (R1+, N1+) depends on the same plumbing.
"""

from __future__ import annotations


def test_core_module_importable() -> None:
    from tsdynamics._native import _core  # noqa: F401


def test_facade_reexports_map_kernels() -> None:
    from tsdynamics._native import iterate_map, lyapunov_spectrum_map

    assert callable(iterate_map)
    assert callable(lyapunov_spectrum_map)
