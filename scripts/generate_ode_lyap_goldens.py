#!/usr/bin/env python3
"""Generate ``tests/native/regression/ode/<Class>.lyap.npz`` for IR+Jacobian ODE systems.

Run from the repo root after editing Lyapunov numerics if goldens need refreshing:

    uv run python scripts/generate_ode_lyap_goldens.py

Each file stores ``lyapunov`` (1d), ``ic``, scalar ``burn_in``, ``final_time``, ``dt``,
``method``, ``rtol``, ``atol``, ``n_exp`` so tests can pin the conditioning.
"""

from __future__ import annotations

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

# Short horizon for file size / CI time; increase if exponents need thinner noise.
_CFG = dict(
    burn_in=20.0,
    final_time=80.0,
    dt=0.1,
    method="DP8",
    rtol=1e-7,
    atol=1e-10,
)


def main() -> None:
    """Write one ``*.lyap.npz`` per eligible catalogue system under ``_OUT``."""
    _OUT.mkdir(parents=True, exist_ok=True)
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
        inst.resolve_ic(None)
        ic = np.asarray(inst.ic, dtype=float).copy()
        n_exp = inst.dim
        try:
            exps = inst.lyapunov_spectrum(
                n_exp=n_exp,
                ic=ic.copy(),
                **_CFG,
            )
        except (ValueError, RuntimeError) as e:
            print(f"skip {class_name}: Lyapunov failed ({e})")
            continue
        target = _OUT / f"{class_name}.lyap.npz"
        np.savez(
            target,
            lyapunov=exps,
            ic=ic,
            burn_in=_CFG["burn_in"],
            final_time=_CFG["final_time"],
            dt=_CFG["dt"],
            method=np.array(_CFG["method"]),
            rtol=_CFG["rtol"],
            atol=_CFG["atol"],
            n_exp=n_exp,
        )
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
