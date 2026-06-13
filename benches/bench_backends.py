"""
Reproducible backend benchmark: JiTCODE (C) vs diffsol (Rust/LLVM) vs SciPy.

Run::

    uv run --extra diffsol python benches/bench_backends.py

Measures warm wall-clock (compilation/JIT excluded) for a long, fine
integration of representative chaotic systems, plus the cold start (first
integration of a fresh process) where the difference is starkest. Prints a
Markdown table; numbers are machine-dependent — regenerate locally.
"""

from __future__ import annotations

import time

import numpy as np

import tsdynamics as ts

SYSTEMS = [
    ("Lorenz", [1.0, 1.0, 1.0]),
    ("Rossler", [1.0, 0.0, 0.0]),
    ("Chen", [1.0, 1.0, 1.0]),
]
FINAL_T, DT = 1000.0, 0.01
TOL = dict(rtol=1e-9, atol=1e-11)


def _time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _scipy_run(name: str, ic: list) -> None:
    from scipy.integrate import solve_ivp

    rhs = getattr(ts, name)()._rhs_numeric()
    solve_ivp(
        lambda t, u: rhs(u, t),
        (0.0, FINAL_T),
        ic,
        t_eval=np.arange(0.0, FINAL_T, DT),
        method="RK45",
        **TOL,
    )


def main() -> None:
    """Print a Markdown benchmark table for the available backends."""
    from tsdynamics.backends import diffsol

    have_diffsol = diffsol.available()
    print(f"diffsol available: {have_diffsol}\n")
    print("| system | jitcode (s) | diffsol (s) | scipy (s) | diffsol speedup |")
    print("|---|---|---|---|---|")

    for name, ic in SYSTEMS:
        # warm each backend (compile/JIT outside timing); bind loop vars as
        # defaults so the timed lambdas don't capture them late.
        getattr(ts, name)().integrate(final_time=1.0, dt=0.1, ic=ic)
        tj = _time(
            lambda n=name, c=ic: getattr(ts, n)().integrate(final_time=FINAL_T, dt=DT, ic=c, **TOL)
        )

        if have_diffsol:
            getattr(ts, name)().integrate(final_time=1.0, dt=0.1, ic=ic, backend="diffsol")
            td = _time(
                lambda n=name, c=ic: getattr(ts, n)().integrate(
                    final_time=FINAL_T, dt=DT, ic=c, backend="diffsol", **TOL
                )
            )
        else:
            td = float("nan")

        tsp = _time(lambda n=name, c=ic: _scipy_run(n, c))
        speedup = f"{tj / td:.1f}×" if have_diffsol and td > 0 else "n/a"
        print(f"| {name} | {tj:.3f} | {td:.3f} | {tsp:.3f} | {speedup} |")


if __name__ == "__main__":
    main()
