"""Packaging-shape invariants for the single maturin wheel (post-M3, I-XVAL).

At M3 the two pre-migration distributions (a pure-Python ``tsdynamics`` wheel +
a separable ``tsdynamics-rust-engine`` engine wheel) converge to **one maturin
wheel** built from the root ``pyproject.toml``: the Rust engine is now the sole
integration backend, so the compiled ``tsdynamics._rust`` extension is mandatory
and ships in the same wheel as the Python package. These checks guard that
mixed-layout build config (``python-source = "src"`` + ``module-name =
"tsdynamics._rust"`` + the engine crate as the Cargo manifest) so a regression —
a dropped ``python-source``, a renamed module, a non-abi3 build — is caught
before it produces an unimportable or platform-fragmented wheel. They read the
build config only (no compilation), so they run in the fast tier.
"""

import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CORE = _REPO / "crates" / "tsdyn-core"


def _pyproject() -> dict:
    return tomllib.loads((_REPO / "pyproject.toml").read_text())


def _cargo() -> dict:
    return tomllib.loads((_CORE / "Cargo.toml").read_text())


def test_build_backend_is_maturin():
    # One wheel, built by maturin from the root project (hatchling is gone).
    assert _pyproject()["build-system"]["build-backend"] == "maturin"


def test_module_name_is_tsdynamics_rust():
    # The extension must be the `_rust` submodule of `tsdynamics`, not a
    # standalone top-level module.
    assert _pyproject()["tool"]["maturin"]["module-name"] == "tsdynamics._rust"


def test_mixed_layout_from_src():
    # `python-source = "src"` is the mixed layout: maturin ships the pure-Python
    # `tsdynamics` package from `src/` and drops the compiled extension into it as
    # `tsdynamics/_rust.*` — one importable namespace, no split `__path__`.
    mat = _pyproject()["tool"]["maturin"]
    assert mat["python-source"] == "src"
    assert (_REPO / "src" / "tsdynamics" / "__init__.py").is_file()


def test_manifest_points_at_the_engine_crate():
    # The wheel's native code is the engine crate; maturin builds it via its
    # Cargo manifest. (`crates/tsdyn-core` is the PyO3 binding crate.)
    mat = _pyproject()["tool"]["maturin"]
    assert mat["manifest-path"] == "crates/tsdyn-core/Cargo.toml"
    assert (_CORE / "Cargo.toml").is_file()


def test_no_separable_engine_pyproject():
    # The pre-M3 second distribution (`crates/tsdyn-core/pyproject.toml`, the
    # `tsdynamics-rust-engine` wheel) is gone — folded into the root wheel.
    assert not (_CORE / "pyproject.toml").exists()


def test_extension_module_feature_enabled():
    # The shipped .so must not link libpython; maturin turns on this feature.
    assert "extension-module" in _pyproject()["tool"]["maturin"]["features"]


def test_cargo_builds_an_abi3_cdylib():
    cargo = _cargo()
    assert "cdylib" in cargo["lib"]["crate-type"]
    pyo3 = cargo["dependencies"]["pyo3"]
    # abi3 -> one wheel per platform covers every CPython >= 3.12.
    assert any(f.startswith("abi3") for f in pyo3["features"]), pyo3["features"]
