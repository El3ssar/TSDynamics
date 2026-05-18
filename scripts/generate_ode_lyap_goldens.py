#!/usr/bin/env python3
"""Generate ``tests/native/regression/ode/<Class>.lyap.npz`` for IR+Jacobian ODE systems.

Run from the repo root after editing Lyapunov numerics if goldens need refreshing:

    uv run python scripts/generate_ode_lyap_goldens.py

Only systems without a ``*.lyap.npz`` yet (saves time; avoids churn):

    uv run python scripts/generate_ode_lyap_goldens.py --only-missing

Each file stores ``lyapunov`` (1d), ``ic``, scalar ``burn_in``, ``final_time``, ``dt``,
``method``, ``rtol``, ``atol``, ``n_exp`` so tests can pin the conditioning.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.test_ode_systems import ALL_ODE_SYSTEMS  # noqa: E402
from tsdynamics.base._ir import NotLowerableError  # noqa: E402
from tsdynamics.base._ode_lowering import lower_ode_to_ir  # noqa: E402
from tsdynamics.base.ode_base import ContinuousSystem  # noqa: E402

_OUT = _REPO / "tests/native/regression/ode"

# Systems where the Rust variational Lyapunov driver hits non-finite augmented
# states (stiffness / attractor geometry) under trajectory-golden ICs — no
# reference file; must match ``_LYAP_GOLDEN_EXCLUDED`` in
# ``tests/test_ode_lyapunov_goldens.py``.
_SKIP_LYAP_GOLDEN: frozenset[str] = frozenset(
    {
        "Oregonator",
        "RabinovichFabrikant",
        "SprottJerk",
    }
)

# Per-class solver conditioning (defaults otherwise match ``_CFG``).
# Henon–Heiles: QR burn-in drives the augmented system unstable near t≈16 for
# the catalogue IC; use ``burn_in=0`` with a modest ``final_time``.
# Hyper-Rössler: dim-4 variational integration is very costly at ``final_time=80``;
# a shorter window still pins the leading exponents.
_LYAP_CFG_OVERRIDES: dict[str, dict] = {
    "HenonHeiles": {"burn_in": 0.0, "final_time": 14.0},
    "HyperRossler": {"burn_in": 10.0, "final_time": 20.0},
}

# Basin-anchored ICs when no trajectory golden exists (align with
# ``scripts/generate_ode_goldens.py``).
_HAND_PICKED_IC: dict[str, np.ndarray] = {
    "Duffing": np.array([0.1, 0.1, 0.0], dtype=float),
    "SprottD": np.array([0.01, 0.01, 0.01], dtype=float),
    "SprottI": np.array([0.01, 0.01, 0.01], dtype=float),
}

# Short horizon for file size / CI time; increase if exponents need thinner noise.
_CFG = dict(
    burn_in=20.0,
    final_time=80.0,
    dt=0.1,
    method="DP8",
    rtol=1e-7,
    atol=1e-10,
)


def _lyap_cfg(class_name: str) -> dict:
    """Return full keyword dict for :meth:`~tsdynamics.ContinuousSystem.lyapunov_spectrum`."""
    cfg = dict(_CFG)
    cfg.update(_LYAP_CFG_OVERRIDES.get(class_name, {}))
    return cfg


def _resolve_ic(class_name: str, inst: ContinuousSystem) -> np.ndarray:
    """Match trajectory goldens when present so Lyapunov uses a stable attractor IC."""
    traj_npz = _OUT / f"{class_name}.npz"
    if traj_npz.is_file():
        data = np.load(traj_npz, allow_pickle=True)
        return np.asarray(data["ic"], dtype=float).copy()
    fixed = _HAND_PICKED_IC.get(class_name)
    if fixed is not None:
        return fixed.copy()
    inst.resolve_ic(None)
    return np.asarray(inst.ic, dtype=float).copy()


def main(*, only_missing: bool) -> None:
    """Write one ``*.lyap.npz`` per eligible catalogue system under ``_OUT``."""
    _OUT.mkdir(parents=True, exist_ok=True)
    for module_path, class_name in ALL_ODE_SYSTEMS:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        if not (isinstance(cls, type) and issubclass(cls, ContinuousSystem)):
            continue
        target = _OUT / f"{class_name}.lyap.npz"
        if only_missing and target.is_file():
            continue
        try:
            inst = cls()
        except TypeError:
            continue
        if inst.dim > 20:
            print(f"skip {class_name}: dim {inst.dim} > 20 (avoid huge augmented systems)")
            continue
        try:
            co = lower_ode_to_ir(
                cls,
                dim=inst.dim,
                params=dict(inst.params),
                structural_params=cls._structural_params,
            )
        except NotLowerableError:
            print(f"skip {class_name}: not lowerable")
            continue
        if not co.has_jacobian:
            print(f"skip {class_name}: no Jacobian in IR")
            continue
        if class_name in _SKIP_LYAP_GOLDEN:
            print(f"skip {class_name}: variational Lyapunov not generated (see _SKIP_LYAP_GOLDEN)")
            continue
        ic = _resolve_ic(class_name, inst)
        n_exp = inst.dim
        lyap_kw = _lyap_cfg(class_name)
        try:
            exps = inst.lyapunov_spectrum(
                n_exp=n_exp,
                ic=ic.copy(),
                **lyap_kw,
            )
        except (ValueError, RuntimeError) as e:
            print(f"skip {class_name}: Lyapunov failed ({e})")
            continue
        np.savez(
            target,
            lyapunov=exps,
            ic=ic,
            burn_in=lyap_kw["burn_in"],
            final_time=lyap_kw["final_time"],
            dt=lyap_kw["dt"],
            method=np.array(lyap_kw["method"]),
            rtol=lyap_kw["rtol"],
            atol=lyap_kw["atol"],
            n_exp=n_exp,
        )
        print(f"wrote {target}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip systems that already have a ``*.lyap.npz`` file.",
    )
    args = ap.parse_args()
    main(only_missing=args.only_missing)
