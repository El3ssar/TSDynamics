"""Tests for the C-SOLV selection layer — specs, resolution, auto-stiffness.

Covers the C-SOLV acceptance (ROADMAP §6 / §13b):

* the in-tree :class:`~tsdynamics.solvers.SolverSpec`s are registered and mirror
  the Rust ``register_solver!`` / ``register_sde_kernel!`` kernels (a parity test
  reads the crate source so the mirror cannot silently drift);
* ``method=`` **resolves by name/caps** (aliases, case, family filtering) and an
  **unknown method raises** a clear, listing error;
* **auto-stiffness** picks an implicit kernel on a stiff RHS and explicit
  otherwise; and
* an implicit method **auto-sets ``with_jacobian``** (the C-FAM seam hook), while
  out-of-tree **plugin solvers are selectable** through the same resolver.

The F2 *mechanism* (register/get/discover) is covered by
:mod:`tests.test_solver_registry`; this file covers the C-SOLV layer on top.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import tsdynamics as ts
from tsdynamics import solvers
from tsdynamics.solvers import SolverCaps, SolverSpec

# The kernels C-SOLV mirrors, with the caps facts the resolver keys on.
# (name, kind, adaptive, needs_jacobian, family)
_EXPECTED_BUILTINS = {
    # explicit — fixed step
    "euler": ("explicit", False, False, "ode"),
    "midpoint": ("explicit", False, False, "ode"),
    "heun": ("explicit", False, False, "ode"),
    "ralston": ("explicit", False, False, "ode"),
    "rk4": ("explicit", False, False, "ode"),
    "rk4_38": ("explicit", False, False, "ode"),
    "ssprk3": ("explicit", False, False, "ode"),
    # explicit — linear-multistep (Adams)
    "ab3": ("explicit", False, False, "ode"),
    "ab4": ("explicit", False, False, "ode"),
    "abm4": ("explicit", False, False, "ode"),
    # explicit — adaptive embedded pairs
    "heun_euler": ("explicit", True, False, "ode"),
    "bs3": ("explicit", True, False, "ode"),
    "rk45": ("explicit", True, False, "ode"),
    "rkf45": ("explicit", True, False, "ode"),
    "cashkarp": ("explicit", True, False, "ode"),
    "tsit5": ("explicit", True, False, "ode"),
    "dop853": ("explicit", True, False, "ode"),
    # implicit / stiff
    "backward_euler": ("implicit", True, True, "ode"),
    "implicit_midpoint": ("implicit", True, True, "ode"),
    "trapezoid": ("implicit", True, True, "ode"),
    "sdirk2": ("implicit", True, True, "ode"),
    "rosenbrock": ("implicit", True, True, "ode"),
    "trbdf2": ("implicit", True, True, "ode"),
    "bdf": ("implicit", True, True, "ode"),
    # SDE
    "euler_maruyama": ("explicit", False, False, "sde"),
    "milstein": ("explicit", False, True, "sde"),
}


@pytest.fixture
def clean_registry():
    """Snapshot the global solver registry and restore it after the test."""
    before = solvers.all_specs()
    yield
    for name in list(solvers.all_specs()):
        if name not in before:
            solvers.unregister(name)
    for spec in before.values():
        solvers.register(spec, override=True)


# A correctly-written synthetic stiff ODE: eigenvalues -1 and -1e6, so the
# stiffness ratio is 1e6 — comfortably above the default 1e3 threshold.  The
# `_equations` contract is `(y, t, *, **params)` with `y(i)` component access.
class _StiffLinear(ts.ContinuousSystem):
    dim = 2
    params: dict = {}
    ic = [1.0, 1.0]

    @staticmethod
    def _equations(y, t):
        return (-1.0 * y(0), -1.0e6 * y(1))


# ── in-tree specs ───────────────────────────────────────────────────────────────


def test_all_builtins_registered():
    names = set(solvers.available())
    assert set(_EXPECTED_BUILTINS) <= names


@pytest.mark.parametrize("name", sorted(_EXPECTED_BUILTINS))
def test_builtin_caps_match_expected(name):
    kind, adaptive, needs_jac, family = _EXPECTED_BUILTINS[name]
    spec = solvers.get(name)
    assert spec.origin == "builtin"
    assert spec.caps.kind == kind
    assert spec.caps.adaptive is adaptive
    assert spec.caps.needs_jacobian is needs_jac
    assert spec.caps.supports_family(family)


def test_available_for_filters_by_family():
    # Derive the expected sets from the single source of truth (_EXPECTED_BUILTINS)
    # so adding a solver needs no edit here — only the table above.
    ode = {n for n, (_k, _a, _j, fam) in _EXPECTED_BUILTINS.items() if fam == "ode"}
    sde = {n for n, (_k, _a, _j, fam) in _EXPECTED_BUILTINS.items() if fam == "sde"}
    # DDE reuses the explicit ODE stage integrators (method-of-steps), so the
    # implicit kernels are excluded.
    dde = {
        n for n, (k, _a, _j, fam) in _EXPECTED_BUILTINS.items() if fam == "ode" and k == "explicit"
    }

    assert set(solvers.available_for("ode")) == ode
    assert set(solvers.available_for("sde")) == sde
    assert set(solvers.available_for("dde")) == dde
    # Maps iterate on the engine's own loop — no solver kernel.
    assert solvers.available_for("map") == []


# ── registry parity with the Rust source ────────────────────────────────────────


def _strip_line_comments(text: str) -> str:
    """Drop ``//`` … line comments (incl. ``///`` / ``//!`` doc comments).

    Register-macro *examples* live in doc comments; stripping them keeps the
    parser from treating a documented call as a real invocation.  Kernel
    registrations contain no ``//`` inside their arguments, so this is safe.
    """
    out = []
    for line in text.splitlines():
        idx = line.find("//")
        out.append(line if idx < 0 else line[:idx])
    return "\n".join(out)


def _parse_rust_kernels() -> dict[str, tuple[str, bool]]:
    """Extract ``name -> (kind, needs_jacobian)`` from the Rust register macros.

    Reads ``crates/tsdyn-solvers/src/**`` directly so the Python mirror cannot
    drift from the link-time registry without this test failing.
    """
    root = Path(__file__).resolve().parent.parent
    src = root / "crates" / "tsdyn-solvers" / "src"
    if not src.is_dir():
        pytest.skip("Rust crate source not present (sdist build); skipping parity check")

    kernels: dict[str, tuple[str, bool, bool]] = {}
    # Require an actual invocation `register_*!(` and strip line comments first,
    # so doc-comment examples / intra-doc links are not mistaken for real calls.
    macro = re.compile(r"register_(?:solver|sde_kernel)!\s*\(")
    for rs in src.rglob("*.rs"):
        text = _strip_line_comments(rs.read_text())
        for m in macro.finditer(text):
            # A register call contains no ';' until its end — take the whole call.
            call = text[m.end() :].split(";", 1)[0]
            name_match = re.search(r'"([^"]+)"', call)
            if not name_match:
                continue
            name = name_match.group(1)
            implicit = "Caps::implicit" in call
            # implicit defaults to needs_jacobian=true; an explicit kernel may
            # opt in via a `needs_jacobian: true` struct-update (e.g. Milstein).
            needs_jac = (implicit and "needs_jacobian: false" not in call) or (
                "needs_jacobian: true" in call
            )
            kind = "implicit" if implicit else "explicit"
            # `.adaptive()` in the caps builder marks an adaptive (own error
            # control) kernel; its absence is a fixed-step / multistep kernel.
            adaptive = ".adaptive()" in call
            kernels[name] = (kind, needs_jac, adaptive)
    return kernels


def test_python_specs_mirror_rust_register_macros():
    rust = _parse_rust_kernels()
    # Every Rust kernel must have a Python builtin spec with matching kind,
    # Jacobian need, and adaptivity; and the Python builtins must not invent kernels.
    builtins = {
        name: spec for name, spec in solvers.all_specs().items() if spec.origin == "builtin"
    }
    assert set(builtins) == set(rust), (
        f"Python builtin specs {sorted(builtins)} != Rust kernels {sorted(rust)}"
    )
    for name, (kind, needs_jac, adaptive) in rust.items():
        spec = builtins[name]
        assert spec.caps.kind == kind, f"{name}: kind {spec.caps.kind!r} != Rust {kind!r}"
        assert spec.caps.adaptive is adaptive, (
            f"{name}: adaptive {spec.caps.adaptive} != Rust {adaptive}"
        )
        assert spec.caps.needs_jacobian is needs_jac, (
            f"{name}: needs_jacobian {spec.caps.needs_jacobian} != Rust {needs_jac}"
        )


# ── name resolution + aliases ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("RK45", "rk45"),
        ("rk45", "rk45"),
        ("dopri5", "rk45"),
        ("DOPRI5", "rk45"),
        ("DP45", "rk45"),
        ("DOP853", "dop853"),
        ("dopri8", "dop853"),
        ("Tsit5", "tsit5"),
        ("Rosenbrock", "rosenbrock"),
        ("ros", "rosenbrock"),
        ("TR-BDF2", "trbdf2"),
        ("tr_bdf2", "trbdf2"),
        ("EM", "euler_maruyama"),
        ("euler-maruyama", "euler_maruyama"),
    ],
)
def test_resolve_aliases(alias, canonical):
    assert solvers.resolve(alias).name == canonical


def test_normalize():
    assert solvers.normalize("  TR-BDF2 ") == "tr_bdf2"
    assert solvers.normalize("Dormand-Prince") == "dormand_prince"


def test_resolve_returns_kernel_and_spec():
    res = solvers.resolve("RK45", family="ode")
    assert res.name == "rk45"
    assert res.kernel == "rk45"
    assert res.adaptive is True
    assert res.is_implicit is False
    assert res.spec is solvers.get("rk45")


def test_unknown_method_raises_listing_available():
    with pytest.raises(ValueError, match=r"unknown solver method 'banana'"):
        solvers.resolve("banana")
    # the message lists what is available
    try:
        solvers.resolve("banana", family="ode")
    except ValueError as exc:
        assert "rk45" in str(exc) and "rosenbrock" in str(exc)


@pytest.mark.parametrize("scipy_stiff", ["LSODA", "Radau", "vode"])
def test_unknown_scipy_stiff_name_hints_engine_kernel(scipy_stiff):
    with pytest.raises(ValueError, match="rosenbrock"):
        solvers.resolve(scipy_stiff)


@pytest.mark.parametrize("name", ["bdf", "BDF", "gear"])
def test_bdf_resolves_to_the_engine_kernel(name):
    # "bdf" used to be a SciPy stiff name with no engine kernel; stream E-BDF added
    # the variable-order BDF kernel, so it now resolves like any built-in.
    res = solvers.resolve(name, family="ode")
    assert res.name == "bdf"
    assert res.is_implicit is True
    assert res.build_kwargs == {"with_jacobian": True}


def test_resolve_rejects_wrong_family():
    # an SDE kernel is not an ODE method...
    with pytest.raises(ValueError, match="does not support the 'ode' family"):
        solvers.resolve("euler_maruyama", family="ode")
    # ...and an ODE kernel is not an SDE method.
    with pytest.raises(ValueError, match="does not support the 'sde' family"):
        solvers.resolve("rk45", family="sde")


def test_resolve_dde_accepts_explicit_ode_but_not_implicit():
    assert solvers.resolve("rk45", family="dde").name == "rk45"
    with pytest.raises(ValueError, match="does not support the 'dde' family"):
        solvers.resolve("rosenbrock", family="dde")


# ── with_jacobian auto-set (the C-FAM seam hook) ─────────────────────────────────


def test_build_kwargs_sets_with_jacobian_for_implicit():
    assert solvers.build_kwargs("rosenbrock") == {"with_jacobian": True}
    assert solvers.build_kwargs("trbdf2") == {"with_jacobian": True}
    assert solvers.build_kwargs("milstein", family="sde") == {"with_jacobian": True}


def test_build_kwargs_empty_for_explicit_no_jac():
    assert solvers.build_kwargs("rk45") == {}
    assert solvers.build_kwargs("rk4") == {}
    assert solvers.build_kwargs("euler_maruyama", family="sde") == {}


def test_predicates():
    assert solvers.needs_jacobian("rosenbrock") is True
    assert solvers.needs_jacobian("rk45") is False
    assert solvers.is_implicit("trbdf2") is True
    assert solvers.is_implicit("dop853") is False


# ── selection policy ─────────────────────────────────────────────────────────────


def test_default_method():
    assert solvers.default_method("ode") == "rk45"
    assert solvers.default_method("sde") == "euler_maruyama"
    assert solvers.default_method("dde") == "rk45"


def test_default_method_for_maps_raises():
    with pytest.raises(ValueError, match="no default solver for family 'map'"):
        solvers.default_method("map")


def test_select_policy():
    assert solvers.select("ode", stiff=False) == "rk45"
    # The variable-order BDF is the default stiff ODE kernel (stream E-BDF).
    assert solvers.select("ode", stiff=True) == "bdf"
    # SDE has no implicit alternative — stiff verdict is ignored.
    assert solvers.select("sde", stiff=True) == "euler_maruyama"


# ── auto-stiffness detection ─────────────────────────────────────────────────────


def test_is_stiff_synthetic_linear():
    sl = _StiffLinear()
    assert solvers.is_stiff(sl) is True
    # ratio is 1e6; raising the threshold past it flips the verdict
    assert solvers.is_stiff(sl, ratio_threshold=1e7) is False


def test_is_stiff_lorenz_is_false():
    # Lorenz's Jacobian eigenvalues are O(10) — nowhere near a stiff ratio.
    assert solvers.is_stiff(ts.Lorenz(), ic=[1.0, 1.0, 1.0]) is False


def test_is_stiff_oregonator_is_true():
    # The Oregonator (Field–Noyes BZ model) is a classic stiff system; at a
    # physical concentration point its spectrum spans ~-1e6…-2 (ratio ~5e5).
    # A fixed ic keeps the test off the random default ic.
    assert solvers.is_stiff(ts.Oregonator(), ic=[1.0, 1.0, 1.0]) is True


def test_is_stiff_is_conservative_on_bad_probe():
    # A non-ODE system has no ∂f/∂u to spectrum-analyse → never reported stiff
    # (selection must not raise on a probe it cannot form).
    assert solvers.is_stiff(ts.Henon()) is False


def test_recommend_picks_implicit_on_stiff():
    res = solvers.recommend(_StiffLinear())
    assert res.name == "bdf"
    assert res.is_implicit is True
    assert res.build_kwargs == {"with_jacobian": True}


def test_recommend_picks_explicit_on_nonstiff():
    res = solvers.recommend(ts.Lorenz(), ic=[1.0, 1.0, 1.0])
    assert res.name == "rk45"
    assert res.build_kwargs == {}


# ── plugin solvers are selectable through the same resolver ──────────────────────


def test_plugin_solver_is_resolvable(clean_registry):
    spec = SolverSpec(
        name="my_method",
        caps=SolverCaps(kind="implicit", needs_jacobian=True, supports={"ode"}),
        description="out-of-tree",
        origin="plugin",
    )
    solvers.register(spec)

    res = solvers.resolve("My-Method", family="ode")  # alias-normalised + resolved
    assert res.name == "my_method"
    assert res.spec is spec
    assert res.needs_jacobian is True
    assert solvers.build_kwargs("my_method") == {"with_jacobian": True}
    assert "my_method" in solvers.available_for("ode")
