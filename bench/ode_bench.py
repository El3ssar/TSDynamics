"""Wall-clock benchmarks: native Rust ODE path vs JiTCODE fallback.

Run::

    uv run python bench/ode_bench.py

For each built-in model we time :meth:`ContinuousSystem.integrate` on identical
`(final_time, dt, rtol, atol, ic)` grids. ``Rust DP5`` uses the IR bytecode + Pure-Rust stepper when
SymEngine lowering succeeds. ``JiTCODE`` forces the legacy path by temporarily making
:data:`tsdynamics.base.ode_base.lower_ode_to_ir` raise :exc:`NotLowerableError`,
then integrate with SciPy-compatible ``method="dopri5"``.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tsdynamics as ts  # noqa: E402
import tsdynamics.base.ode_base as ode_base_mod  # noqa: E402

INT_KW = {
    "final_time": 80.0,
    "dt": 0.02,
    "rtol": 1e-6,
    "atol": 1e-9,
}


def _systems() -> list[tuple[str, Callable[[], object], np.ndarray]]:
    return [
        ("Lorenz", lambda: ts.Lorenz(), np.array([1.0, 1.0, 1.0], dtype=float)),
        ("Rossler", lambda: ts.Rossler(), np.ones(3, dtype=float)),
    ]


@contextlib.contextmanager
def _forcing_jitcode() -> Iterator[None]:
    real = ode_base_mod.lower_ode_to_ir

    def _raise(*_: object, **__: object) -> None:
        raise ode_base_mod.NotLowerableError()

    ode_base_mod.lower_ode_to_ir = _raise  # type: ignore[assignment]
    try:
        yield
    finally:
        ode_base_mod.lower_ode_to_ir = real


def _clear_ode_ir_cache(system: object) -> None:
    type(system)._compiled_ode_ir.clear()


def _elapsed(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _bench_one(factory: Callable[[], object], ic: np.ndarray) -> tuple[float, float]:
    """Warm JiTCODE and Rust caches, then measure one timed run each."""
    sys_rust = factory()
    sys_jit = factory()
    ic_r = ic.copy()
    ic_j = ic.copy()

    _clear_ode_ir_cache(sys_rust)
    _clear_ode_ir_cache(sys_jit)

    # Warm Rust + compile IR to cache — not timed.
    _ = sys_rust.integrate(**INT_KW, ic=ic.copy(), method="DP5")

    # WarmJiTCODE: poison lowering + jitcode compile cache.
    with _forcing_jitcode():
        _clear_ode_ir_cache(sys_jit)
        _ = sys_jit.integrate(**INT_KW, ic=ic.copy(), method="dopri5")

    def run_rust() -> None:
        s = factory()
        _clear_ode_ir_cache(s)
        _ = s.integrate(**INT_KW, ic=ic_r.copy(), method="DP5")

    def run_jit() -> None:
        s = factory()
        with _forcing_jitcode():
            _clear_ode_ir_cache(s)
            _ = s.integrate(**INT_KW, ic=ic_j.copy(), method="dopri5")

    t_rust = _elapsed(run_rust)
    t_jit = _elapsed(run_jit)
    return t_rust, t_jit


def main() -> None:
    """Print timing rows and append the ODE section to ``bench/RESULTS.md``."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    kw_str = ", ".join(f"{k}={INT_KW[k]!r}" for k in ("final_time", "dt", "rtol", "atol"))

    rows: list[tuple[str, float, float]] = []

    print(f"Grid: {kw_str}")
    print(f"{'System':<10} {'Rust DP5':>11} {'JiTCODE dp5':>13} {'JIT/Rust':>10}")
    print("-" * 50)
    for label, mk, ic in _systems():
        tr, tj = _bench_one(mk, ic)
        ratio = tj / tr if tr > 0 else float("nan")
        rows.append((label, tr, tj))
        print(f"{label:<10} {tr:11.4f}s {tj:13.4f}s {ratio:10.3f}x")

    out = Path(__file__).resolve().parent / "RESULTS.md"
    with out.open("a") as f:
        f.write(f"\n## ODE benchmarks — run {stamp}\n\n")
        f.write(
            "Same `(final_time, dt, rtol, atol)` grid; **Rust DP5** evaluates the bytecode RHS in-process;\n"
            "**JiTCODE** runs ``dopri5`` with IR lowering toggled off via `NotLowerableError`. "
            "Whole-call wall time includes output-grid sampling.\n\n"
        )
        f.write("| System | Rust DP5 (s) | JiTCODE dopri5 (s) | JiTCODE / Rust |\n")
        f.write("|--------|---------------:|-------------------:|---------------:|\n")
        for label, tr, tj in rows:
            ratio = tj / tr if tr > 0 else float("nan")
            f.write(f"| {label} | {tr:.4f} | {tj:.4f} | {ratio:.3f}× |\n")
    print(f"\nAppended ODE rows to {out}")


if __name__ == "__main__":
    main()
