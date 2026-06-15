"""Packaging-shape invariants for the ``tsdynamics._rust`` engine wheel (I-WHEEL).

These guard the separable-accelerator layout documented in
``docs/theory/packaging.md``: the engine wheel (``tsdynamics-rust-engine``) must
ship ONLY ``tsdynamics/_rust.*`` into the ``tsdynamics`` import namespace, so it
coexists with the pure-Python ``tsdynamics`` wheel without a file collision. A
regression here — a stray ``__init__.py`` in the namespace mount, a dropped
``python-source``, or a renamed module — silently splits ``tsdynamics.__path__``
or makes the built wheel unimportable. The checks read the build config only (no
compilation), so they run in the fast tier.
"""

import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CORE = _REPO / "crates" / "tsdyn-core"

# An installed sdist of `tsdynamics` does not carry the engine crate; only the
# source checkout does. Skip cleanly elsewhere rather than fail.
pytestmark = pytest.mark.skipif(
    not _CORE.exists(),
    reason="engine crate (crates/tsdyn-core) not present in this checkout",
)


def _pyproject() -> dict:
    return tomllib.loads((_CORE / "pyproject.toml").read_text())


def _cargo() -> dict:
    return tomllib.loads((_CORE / "Cargo.toml").read_text())


def test_build_backend_is_maturin():
    assert _pyproject()["build-system"]["build-backend"] == "maturin"


def test_module_name_is_tsdynamics_rust():
    # The extension must be the `_rust` submodule of `tsdynamics`, not a
    # standalone top-level module.
    assert _pyproject()["tool"]["maturin"]["module-name"] == "tsdynamics._rust"


def test_mixed_layout_namespace_mount():
    # `python-source` makes maturin nest the extension under `tsdynamics/`
    # (mixed layout) instead of emitting a broken top-level `_rust/` package.
    mat = _pyproject()["tool"]["maturin"]
    assert mat["python-source"] == "python"
    assert (_CORE / "python" / "tsdynamics").is_dir(), "namespace mount dir missing"


def test_namespace_mount_ships_nothing_but_the_extension():
    # An `__init__.py` here would be shipped as `tsdynamics/__init__.py` and
    # collide with the pure-Python `tsdynamics` wheel -> split __path__. The mount
    # must hold only the `.gitkeep`, and `.gitkeep` must be excluded from the wheel.
    mount = _CORE / "python" / "tsdynamics"
    assert not (mount / "__init__.py").exists()
    extra = sorted(p.name for p in mount.iterdir() if p.name != ".gitkeep")
    assert extra == [], f"unexpected files in the namespace mount: {extra}"
    assert "**/.gitkeep" in _pyproject()["tool"]["maturin"].get("exclude", [])


def test_extension_module_feature_enabled():
    # The shipped .so must not link libpython; maturin turns on this feature.
    assert "extension-module" in _pyproject()["tool"]["maturin"]["features"]


def test_cargo_builds_an_abi3_cdylib():
    cargo = _cargo()
    assert "cdylib" in cargo["lib"]["crate-type"]
    pyo3 = cargo["dependencies"]["pyo3"]
    # abi3 -> one wheel per platform covers every CPython >= 3.12.
    assert any(f.startswith("abi3") for f in pyo3["features"]), pyo3["features"]
