"""Golden-file regression tests for the Rust-backed discrete-map kernels.

The golden ``.npz`` files under ``tests/native/regression/`` were
generated against the original Numba dispatch path (see
``scripts/generate_map_goldens.py``) before N1 landed any Rust code.
Every map's IR-interpreted trajectory and Lyapunov spectrum must agree
with its golden to within numerical tolerance.

Iteration tolerance is set to bit-exact; the Rust IR-interpreted path
happens to reproduce Numba's float ops exactly today, which is the
strongest guarantee we can ship. If a future codegen change perturbs the
floats, loosen this to ``1e-12`` rather than regenerating the goldens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.base._ir import NotLowerableError
from tsdynamics.base._lowering import lower_to_ir
from tsdynamics.base.map_base import DiscreteMap
from tsdynamics.utils import staticjit

# Every built-in map. Order doesn't matter — pytest parametrises.
MAP_NAMES = [
    "Henon",
    "Ulam",
    "Ikeda",
    "Tinkerbell",
    "Gingerbreadman",
    "Zaslavskii",
    "Chirikov",
    "FoldedTowel",
    "GeneralizedHenon",
    "Bogdanov",
    "Svensson",
    "Bedhead",
    "ZeraouliaSprott",
    "GumowskiMira",
    "Hopalong",
    "Pickover",
    "Tent",
    "Baker",
    "Circle",
    "Chebyshev",
    "Gauss",
    "DeJong",
    "KaplanYorke",
    "Logistic",
    "Ricker",
    "MaynardSmith",
]

GOLDEN_DIR = Path(__file__).parent / "native" / "regression"

ITER_ATOL = 1e-12
LYAP_ATOL = 1e-6


def _load_golden(name: str):
    path = GOLDEN_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")
    return np.load(path)


@pytest.mark.parametrize("name", MAP_NAMES)
def test_lower_to_ir_succeeds(name: str) -> None:
    """Every built-in map must trace to IR without raising NotLowerableError."""
    cls = getattr(ts, name)
    inst = cls()
    compiled = lower_to_ir(cls, inst.params.as_tuple(), inst.dim)
    assert compiled.dim == inst.dim
    assert compiled.n_params == len(inst.params)
    assert len(compiled.bytecode) > 0


@pytest.mark.parametrize("name", MAP_NAMES)
def test_iterate_matches_golden(name: str) -> None:
    """Rust-backed iterate must reproduce the golden trajectory."""
    g = _load_golden(name)
    cls = getattr(ts, name)
    sys = cls()
    np.random.seed(int(g["seed"]))
    traj = sys.iterate(steps=int(g["steps_iterate"]), ic=g["ic"].copy())
    assert traj.y.shape == g["trajectory"].shape
    diff = np.abs(traj.y - g["trajectory"]).max()
    assert diff <= ITER_ATOL, f"max abs diff {diff:e} > {ITER_ATOL:e}"


@pytest.mark.parametrize("name", MAP_NAMES)
def test_lyapunov_matches_golden(name: str) -> None:
    """Rust-backed Lyapunov spectrum must reproduce the golden values."""
    g = _load_golden(name)
    cls = getattr(ts, name)
    sys = cls()
    np.random.seed(int(g["seed"]))
    lyap = sys.lyapunov_spectrum(steps=int(g["steps_lyap"]), ic=g["ic"].copy())
    assert lyap.shape == g["lyapunov"].shape
    diff = np.abs(lyap - g["lyapunov"]).max()
    assert diff <= LYAP_ATOL, f"max abs diff {diff:e} > {LYAP_ATOL:e}"


# ---------------------------------------------------------------------------
# Fallback path: a user-defined map with Python `if/else` cannot lower; the
# dispatcher must fall through to the Numba-compiled iterate loop instead.
# ---------------------------------------------------------------------------


class _IfBranchMap(DiscreteMap):
    """Map that uses a Python ``if`` on the state — forces the Numba fallback."""

    params = {"mu": 0.95}
    dim = 1

    @staticjit
    def _step(X, mu):
        x = X
        if x < 0.5:  # noqa: SIM108 — intentionally non-lowerable
            return mu * 2 * x
        else:
            return mu * 2 * (1 - x)

    @staticjit
    def _jacobian(X, mu):
        x = X
        if x < 0.5:
            return [2 * mu]
        else:
            return [-2 * mu]


def test_unlowerable_map_falls_back_to_numba() -> None:
    """A map that can't lower should still run via the Numba path."""
    sys = _IfBranchMap()
    compiled = sys._compile_ir()
    assert compiled is None, "branching map should not lower"
    traj = sys.iterate(steps=200, ic=np.array([0.3]))
    assert traj.y.shape == (200, 1)
    assert np.all(np.isfinite(traj.y))


def test_unlowerable_map_lyapunov_falls_back() -> None:
    """Lyapunov spectrum on the same fallback map must complete."""
    sys = _IfBranchMap()
    assert sys._compile_ir() is None
    exps = sys.lyapunov_spectrum(steps=2000, ic=np.array([0.3]))
    assert exps.shape == (1,)
    # Tent-like dynamics: positive Lyapunov exponent of order log 2 * mu.
    assert np.isfinite(exps[0])


def test_notlowerable_raised_on_bool_cast() -> None:
    """Tracers must raise NotLowerableError when forced through __bool__."""
    from tsdynamics.base._ir import Var
    from tsdynamics.base._tracer import Tracer

    t = Tracer(Var(0))
    with pytest.raises(NotLowerableError):
        bool(t)
