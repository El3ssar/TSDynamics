"""Regression: Lyapunov spectra vs committed ``*.lyap.npz`` goldens (**N3**)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_ode_systems import ALL_ODE_SYSTEMS  # noqa: E402

from tsdynamics.base._ir import NotLowerableError  # noqa: E402
from tsdynamics.base._ode_lowering import lower_ode_to_ir  # noqa: E402
from tsdynamics.base.ode_base import ContinuousSystem  # noqa: E402

_REG = Path(__file__).resolve().parent / "native" / "regression" / "ode"

_ID_BY_CLASS = {name: (mod, name) for mod, name in ALL_ODE_SYSTEMS}

# Catalogue systems that do not ship a ``*.lyap.npz``: the variational Rust
# driver hits non-finite augmented states with the usual IC / tolerances
# (stiff chemistry, or polynomial blow-up in the tangent equations).  Keep in
# sync with ``_SKIP_LYAP_GOLDEN`` in ``scripts/generate_ode_lyap_goldens.py``.
_LYAP_GOLDEN_EXCLUDED: frozenset[str] = frozenset(
    {
        "Oregonator",
        "RabinovichFabrikant",
        "SprottJerk",
    }
)


def _goldens() -> list[Path]:
    return sorted(_REG.glob("*.lyap.npz"))


@pytest.mark.slow
@pytest.mark.parametrize("golden_path", _goldens(), ids=lambda p: p.stem.replace(".lyap", ""))
def test_lyapunov_matches_golden(golden_path: Path) -> None:
    stem = golden_path.name.removesuffix(".lyap.npz")
    pair = _ID_BY_CLASS.get(stem)
    if pair is None:
        pytest.fail(f"no ALL_ODE_SYSTEMS entry for golden {golden_path.name}")

    g = np.load(golden_path, allow_pickle=True)
    ref = np.asarray(g["lyapunov"], dtype=float)
    ic = np.asarray(g["ic"], dtype=float)
    burn_in = float(g["burn_in"])
    final_time = float(g["final_time"])
    dt = float(g["dt"])
    method = str(np.asarray(g["method"]).item())
    rtol = float(g["rtol"])
    atol = float(g["atol"])
    n_exp = int(g["n_exp"])

    mod = importlib.import_module(pair[0])
    cls = getattr(mod, pair[1])
    sys = cls()
    exps = sys.lyapunov_spectrum(
        final_time=final_time,
        dt=dt,
        ic=ic.copy(),
        n_exp=n_exp,
        burn_in=burn_in,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    assert exps.shape == ref.shape, (exps.shape, ref.shape)
    assert np.allclose(exps, ref, rtol=1e-3, atol=1e-3), (
        f"{stem} max abs diff {np.max(np.abs(exps - ref))}: got {exps}, ref {ref}"
    )


def test_lowerable_jacobian_catalogue_has_lyap_golden_or_exclusion() -> None:
    """Every finite-dim built-in ODE that lowers with a Jacobian has a golden or a waiver."""
    for module_path, class_name in ALL_ODE_SYSTEMS:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        if not (isinstance(cls, type) and issubclass(cls, ContinuousSystem)):
            continue
        try:
            inst = cls()
        except TypeError:
            continue
        if inst.dim > 20:
            continue
        try:
            co = lower_ode_to_ir(
                cls,
                dim=inst.dim,
                params=dict(inst.params),
                structural_params=cls._structural_params,
            )
        except NotLowerableError:
            continue
        if not co.has_jacobian:
            continue
        path = _REG / f"{class_name}.lyap.npz"
        if class_name in _LYAP_GOLDEN_EXCLUDED:
            assert not path.is_file(), (
                f"{class_name}: remove from _LYAP_GOLDEN_EXCLUDED if adding {path.name}"
            )
            continue
        assert path.is_file(), (
            f"{class_name}: missing {path.name} (generate via "
            f"``uv run python scripts/generate_ode_lyap_goldens.py --only-missing`` "
            "or add a documented waiver to _LYAP_GOLDEN_EXCLUDED)"
        )
