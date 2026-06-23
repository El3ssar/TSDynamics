"""Packaging-shape invariants for the single maturin wheel (post-M3, I-XVAL).

At M3 the two pre-migration distributions (a pure-Python ``tsdynamics`` wheel +
a separable ``tsdynamics-rust-engine`` engine wheel) converge to **one maturin
wheel** built from the root ``pyproject.toml``: the Rust engine is now the sole
integration backend, so the compiled ``tsdynamics._rust`` extension is mandatory
and ships in the same wheel as the Python package. These checks guard that
mixed-layout build config (``python-source = "src"`` + ``module-name =
"tsdynamics._rust"`` + the engine crate as the Cargo manifest) so a regression —
a dropped ``python-source``, a renamed module, a non-abi3 build — is caught
before it produces an unimportable or platform-fragmented wheel. These read the
build config only (no compilation), so they run in the fast tier.

A second group reads a *built* wheel's contents (the zip layout + the WHEEL tag)
— the only place a wrong-shape build truly shows up: the wheel must ship
``tsdynamics/_rust.*.so`` *inside* the package, never a top-level ``_rust`` module,
and carry the ``cp312-abi3`` tag. Building one needs maturin + the Rust toolchain,
so those checks are opt-in: they skip when no compiled wheel is present (the fast
inner loop) and ``.github/workflows/wheels.yml`` points them at each freshly built
artifact via the ``TSD_WHEEL`` environment variable.
"""

import os
import re
import tomllib
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CORE = _REPO / "crates" / "tsdyn-core"

# A compiled abi3 wheel's filename tag, e.g. `tsdynamics-3.1.0-cp312-abi3-...whl`.
# Pure-Python / editable wheels carry `py3-none-any` and are NOT a release wheel,
# so the discovery below deliberately ignores them.
_ABI3_WHEEL_RE = re.compile(r"-cp3\d+-abi3-", re.IGNORECASE)


def _pyproject() -> dict:
    return tomllib.loads((_REPO / "pyproject.toml").read_text())


def _cargo() -> dict:
    return tomllib.loads((_CORE / "Cargo.toml").read_text())


def _find_built_wheel() -> Path | None:
    """Locate a freshly built **compiled** ``tsdynamics`` abi3 wheel, or ``None``.

    These checks inspect a real wheel's *contents* (the zip layout + the WHEEL
    tag) — the only place a dropped ``python-source``, a renamed extension, or a
    non-abi3 build actually shows up. Building one needs maturin + the Rust
    toolchain, so this is opt-in rather than part of the fast inner loop:

    * ``TSD_WHEEL`` (a path to a single ``.whl``) — set by ``wheels.yml`` so the
      shape is verified per built artifact; or
    * a compiled ``tsdynamics-*-cp3*-abi3-*.whl`` under ``dist/``.

    Pure-Python / editable wheels (``py3-none-any``) are skipped: they carry no
    extension and are not the release artifact this guards.
    """
    env = os.environ.get("TSD_WHEEL")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    dist = _REPO / "dist"
    if not dist.is_dir():
        return None
    cands = [w for w in sorted(dist.glob("tsdynamics-*.whl")) if _ABI3_WHEEL_RE.search(w.name)]
    return cands[-1] if cands else None


def _wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


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


# ── Built-wheel content invariants (opt-in: needs a compiled wheel) ───────────
#
# The config checks above guard the *intent*; these read the actual wheel zip so a
# build that silently produces the wrong shape is caught too. They skip cleanly
# when no compiled wheel is present (the fast inner loop), and `wheels.yml` runs
# them per built artifact via `TSD_WHEEL`.

_NEED_WHEEL = "no compiled abi3 wheel found (set TSD_WHEEL or build into dist/)"


def test_wheel_ships_the_rust_extension_inside_the_package():
    # The compiled extension must land as `tsdynamics/_rust.*.so` (or `.pyd`) —
    # inside the package, the mixed-layout drop point. A missing extension means a
    # broken `module-name`/`python-source`; an unimportable wheel.
    wheel = _find_built_wheel()
    if wheel is None:
        pytest.skip(_NEED_WHEEL)
    names = _wheel_names(wheel)
    ext = [n for n in names if re.fullmatch(r"tsdynamics/_rust\.(abi3\.so|so|pyd)", n)]
    assert ext, f"no tsdynamics/_rust extension in {wheel.name}: {names}"
    # The pure-Python package ships alongside it (the `python-source = src` half).
    assert "tsdynamics/__init__.py" in names, names


def test_wheel_has_no_top_level_rust_module():
    # Regression guard for the original broken maturin scaffold (I-WHEEL): the
    # extension must NOT appear as a top-level `_rust` module/package — that wheel
    # was unimportable as `tsdynamics._rust`.
    wheel = _find_built_wheel()
    if wheel is None:
        pytest.skip(_NEED_WHEEL)
    for n in _wheel_names(wheel):
        top = n.split("/", 1)[0]
        assert top != "_rust", f"top-level _rust entry in {wheel.name}: {n}"
        assert not re.fullmatch(r"_rust\.(abi3\.so|so|pyd)", n), (
            f"top-level _rust extension in {wheel.name}: {n}"
        )


def test_wheel_is_cp312_abi3_tagged():
    # abi3 (cp312) -> one wheel per (platform, arch) covers every CPython >= 3.12.
    # Assert both the filename tag and the WHEEL metadata `Tag:` agree.
    wheel = _find_built_wheel()
    if wheel is None:
        pytest.skip(_NEED_WHEEL)
    assert "cp312-abi3" in wheel.name, wheel.name
    with zipfile.ZipFile(wheel) as zf:
        meta = next(
            (n for n in zf.namelist() if re.fullmatch(r".*\.dist-info/WHEEL", n)),
            None,
        )
        assert meta is not None, f"no WHEEL metadata in {wheel.name}"
        wheel_meta = zf.read(meta).decode()
    tags = [
        line.split(":", 1)[1].strip() for line in wheel_meta.splitlines() if line.startswith("Tag:")
    ]
    assert tags, wheel_meta
    assert all("cp312-abi3" in t for t in tags), tags


def test_viz_extras_declare_pinned_backends():
    """The optional viz / interactive extras pin the plot backends (VIZ-PACKAGING).

    ``pip install tsdynamics[viz]`` pulls in matplotlib (the reference renderer);
    ``[interactive]`` pulls in plotly.  ``json`` / ``threejs`` export need no extra
    (stdlib JSON over ``PlotSpec.to_dict``).  Core stays plot-free regardless.
    """
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "viz" in extras, "the [viz] extra (matplotlib) must exist"
    assert "interactive" in extras, "the [interactive] extra (plotly) must exist"
    assert any(d.startswith("matplotlib") and "<5" in d for d in extras["viz"]), extras["viz"]
    assert any(d.startswith("plotly") for d in extras["interactive"]), extras["interactive"]
