"""Internal package for Rust-backed extension modules.

User code never imports from here directly. The public Python API in
:mod:`tsdynamics` re-exports anything that is meant to be user-facing.
This package exists so each Rust crate compiled by maturin has a stable
installation target (e.g. ``tsdynamics._native._smoke``) and so the rest of
the library can route hot loops through a single facade.

The contents are populated incrementally as Rust kernels land (see
``.planning/ROADMAP.md`` Track C / Track E). Milestone R1 only ships the
``_smoke`` module, which exists to prove the PyO3 round trip works.
"""

from __future__ import annotations

from ._smoke import add_one

__all__ = ["add_one"]
