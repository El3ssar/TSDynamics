"""Internal package for Rust-backed extension modules.

User code never imports from here directly. The public Python API in
:mod:`tsdynamics` re-exports anything that is meant to be user-facing.
This package exists so the single Rust extension (``_core``) compiled by
maturin has a stable installation target and the rest of the library can
route hot loops through one facade.

Contents grow incrementally as Rust kernels land (see
``.planning/ROADMAP.md`` Track C / Track E). N1 ships:

- ``add_one`` — R1 smoke-test holdover.
- ``iterate_map`` / ``lyapunov_spectrum_map`` — discrete-map kernels.
"""

from __future__ import annotations

from ._core import add_one, iterate_map, lyapunov_spectrum_map

__all__ = ["add_one", "iterate_map", "lyapunov_spectrum_map"]
