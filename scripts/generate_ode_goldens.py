"""Generate golden ODE trajectories for every built-in continuous system.

Runs ``ContinuousSystem.integrate(..., method="dop853")`` (which resolves to the
Rust **DP8** driver when lowering succeeds; JiTCODE ``dop853`` otherwise) and
writes one ``.npz`` per system under ``tests/native/regression/ode/``.  These
files are the trajectory regression reference for that integration path.

Run once, commit the npz files, do NOT regenerate after N2 lands unless
making a deliberate semantic change.  If you regenerate, note it in
``CHANGELOG.md``.

Per-system goldens carry:

- ``t``                  : 1-D float64 time axis (uniform)
- ``y``                  : (T, dim) state trajectory
- ``ic``                 : the resolved initial condition
- ``params``             : flat float64 array of parameter values in
                            ``sys.params.items()`` insertion order
- ``param_names``        : str array of parameter names matching ``params``
- ``structural_params``  : str array (possibly empty) of names baked at
                            compile time
- ``structural_values``  : float64 array of the structural parameter
                            values (only populated for variable-dim
                            systems like Lorenz96 / KS / MultiChua)
- scalars: ``final_time``, ``dt``, ``rtol``, ``atol``, ``method``,
  ``seed``

Strategy
--------
Every system integrates with ``dop853``, which maps to **Rust DP8** in builds
that ship the native integrator (matching SciPy-style coefficients).  Older
JiTCODE-only snapshots drift numerically from Rust dense sampling on chaotic
systems — regenerate after intentional solver changes.

A per-system wallclock timeout (``signal.SIGALRM``) skips any system
whose integration stalls.  Skipped systems print a warning; the missing
.npz is then a known gap that N2.b can address with a hand-picked
method / tolerance.
"""

from __future__ import annotations

import importlib
import signal
import sys as _sys
import time
import warnings
from pathlib import Path

import numpy as np

# Reuse the canonical system list from the test suite so the golden set
# and the integration-test set never drift.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT / "tests"))
from test_ode_systems import ALL_ODE_SYSTEMS  # noqa: E402

# Variable-dim systems must be instantiated with a chosen (N, ...). Pin
# the configurations we use for goldens so the npz contents are stable
# across regenerations.
_VARIABLE_DIM_FACTORIES = {
    "Lorenz96": lambda: importlib.import_module("tsdynamics").Lorenz96(N=8),
    "KuramotoSivashinsky": lambda: importlib.import_module("tsdynamics").KuramotoSivashinsky(
        N=8, L=8.0
    ),
    "MultiChua": lambda: importlib.import_module("tsdynamics").MultiChua(n_circuits=2),
}

# Basin-anchored ICs for systems where ``resolve_ic(None)`` with the golden
# RNG seed blows up or stalls under the default tolerances (N2.b).
_HAND_PICKED_IC: dict[str, np.ndarray] = {
    "Duffing": np.array([0.1, 0.1, 0.0], dtype=float),
    "SprottD": np.array([0.01, 0.01, 0.01], dtype=float),
    "SprottI": np.array([0.01, 0.01, 0.01], dtype=float),
}

# Default integration parameters for the goldens. Short enough to keep
# regeneration cheap, long enough that chaotic systems start populating
# their attractor.
FINAL_TIME = 5.0
DT = 0.05
RTOL = 1e-8
ATOL = 1e-10
METHOD = "dop853"
SEED = 0
PER_SYSTEM_TIMEOUT_S = 45


class _TimeoutError(RuntimeError):
    """Raised by :func:`_alarm_handler` when a per-system timeout fires."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _TimeoutError("integration timed out")


def _instantiate(class_name: str, module_path: str):
    factory = _VARIABLE_DIM_FACTORIES.get(class_name)
    if factory is not None:
        return factory()
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls()


def _resolve_ic(sys: object, class_name: str) -> np.ndarray:
    fixed = _HAND_PICKED_IC.get(class_name)
    if fixed is not None:
        return fixed.copy()
    np.random.seed(SEED)
    return sys.resolve_ic(ic=None).copy()


def _structural_arrays(sys: object) -> tuple[np.ndarray, np.ndarray]:
    structural_names = sorted(type(sys)._structural_params)
    structural_vals = np.array([float(sys.params[k]) for k in structural_names], dtype=float)
    return np.array(structural_names, dtype=object), structural_vals


def _integrate_with_timeout(sys, ic):
    """Integrate with a hard wallclock timeout.

    ``signal.SIGALRM`` is POSIX-only; if the script ever needs to run on
    Windows, swap for a thread-based timeout. We don't worry about that
    here — goldens are generated on the maintainer's machine, which is
    Linux today.
    """
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(PER_SYSTEM_TIMEOUT_S)
    try:
        with warnings.catch_warnings():
            # Treat "step size becomes too small" as a hard failure so
            # we don't churn forever on a degenerate stiff config.
            warnings.simplefilter("error", UserWarning)
            return sys.integrate(
                final_time=FINAL_TIME,
                dt=DT,
                ic=ic,
                method=METHOD,
                rtol=RTOL,
                atol=ATOL,
            )
    finally:
        signal.alarm(0)


def main() -> None:
    """Iterate the catalogue, integrate each system, write golden ``.npz``."""
    out_dir = _REPO_ROOT / "tests" / "native" / "regression" / "ode"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[tuple[str, str]] = []
    t_start = time.perf_counter()

    for module_path, class_name in ALL_ODE_SYSTEMS:
        sys = _instantiate(class_name, module_path)
        ic = _resolve_ic(sys, class_name)

        t0 = time.perf_counter()
        try:
            traj = _integrate_with_timeout(sys, ic)
        except _TimeoutError:
            skipped.append((class_name, f"timeout after {PER_SYSTEM_TIMEOUT_S}s"))
            print(
                f"  SKIP  {class_name:<28}  timeout after {PER_SYSTEM_TIMEOUT_S}s",
                flush=True,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            skipped.append((class_name, f"{type(exc).__name__}: {exc}"))
            print(
                f"  SKIP  {class_name:<28}  {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        dt_elapsed = time.perf_counter() - t0

        if not np.all(np.isfinite(traj.y)):
            skipped.append((class_name, "non-finite trajectory"))
            print(
                f"  SKIP  {class_name:<28}  non-finite trajectory",
                flush=True,
            )
            continue

        param_names = np.array(list(sys.params.keys()), dtype=object)
        params = np.array(list(sys.params.values()), dtype=float)
        struct_names, struct_vals = _structural_arrays(sys)

        out_path = out_dir / f"{class_name}.npz"
        np.savez(
            out_path,
            t=traj.t,
            y=traj.y,
            ic=ic,
            params=params,
            param_names=param_names,
            structural_params=struct_names,
            structural_values=struct_vals,
            final_time=np.float64(FINAL_TIME),
            dt=np.float64(DT),
            rtol=np.float64(RTOL),
            atol=np.float64(ATOL),
            method=np.array(METHOD, dtype=object),
            seed=np.int64(SEED),
        )
        written += 1
        print(
            f"  OK    {class_name:<28}  dim={sys.dim:<3} shape={traj.y.shape}  {dt_elapsed:5.2f}s",
            flush=True,
        )

    total = time.perf_counter() - t_start
    print(
        f"\nWrote {written} golden files to {out_dir}  "
        f"(total {total:.1f}s, skipped {len(skipped)})",
        flush=True,
    )
    if skipped:
        print("\nSkipped systems (regenerate with explicit ic= or different method):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
