"""
Reproducible backend benchmark: Rust engine (interp/jit) vs JiTCODE/diffsol/SciPy.

Run::

    uv run --extra diffsol python benches/bench_backends.py

Measures warm wall-clock (compilation/JIT excluded) for a long, fine
integration of representative chaotic systems. ``interp`` is the zero-warmup
SSA-tape interpreter and ``jit`` is the Cranelift native-code evaluator, both
reached through ``integrate(backend=...)`` once the engine extension
(:mod:`tsdynamics._rust`) is built; they are skipped with a note if it is not.
Prints a Markdown table; numbers are machine-dependent — regenerate locally.

This is the quick warm-only table. For the full decision-grade harness — cold
time-to-first-result, warm throughput *and* ensemble throughput across every
family (ODE / stiff / DDE / map / SDE) — use :mod:`benches.bench_engine` and see
``benches/REPORT.md``.
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


def _engine_available() -> bool:
    """Whether the compiled Rust engine (tsdynamics._rust) is importable."""
    try:
        import tsdynamics._rust  # noqa: F401
    except ImportError:
        return False
    return True


def _warm_engine(name: str, ic: list, backend: str) -> float:
    """Warm then time one engine-backend integration (interp / jit)."""
    getattr(ts, name)().integrate(final_time=1.0, dt=0.1, ic=ic, backend=backend)
    return _time(
        lambda n=name, c=ic, b=backend: getattr(ts, n)().integrate(
            final_time=FINAL_T, dt=DT, ic=c, backend=b, **TOL
        )
    )


def main() -> None:
    """Print a Markdown benchmark table for the available backends."""
    from tsdynamics.engine import diffsol

    have_diffsol = diffsol.available()
    have_engine = _engine_available()
    print(f"diffsol available: {have_diffsol} · rust engine available: {have_engine}\n")
    print(
        "| system | interp (s) | jit (s) | jitcode (s) | diffsol (s) | scipy (s) "
        "| best-engine speedup vs jitcode |"
    )
    print("|---|---|---|---|---|---|---|")

    for name, ic in SYSTEMS:
        # warm each backend (compile/JIT outside timing); bind loop vars as
        # defaults so the timed lambdas don't capture them late.
        getattr(ts, name)().integrate(final_time=1.0, dt=0.1, ic=ic)
        tj = _time(
            lambda n=name, c=ic: getattr(ts, n)().integrate(final_time=FINAL_T, dt=DT, ic=c, **TOL)
        )

        if have_engine:
            ti = _warm_engine(name, ic, "interp")
            tg = _warm_engine(name, ic, "jit")
        else:
            ti = tg = float("nan")

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
        best_engine = min(t for t in (ti, tg) if t == t) if have_engine else float("nan")
        speedup = f"{tj / best_engine:.1f}×" if have_engine and best_engine > 0 else "n/a"
        print(f"| {name} | {ti:.3f} | {tg:.3f} | {tj:.3f} | {td:.3f} | {tsp:.3f} | {speedup} |")


if __name__ == "__main__":
    main()
