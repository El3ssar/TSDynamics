"""Benchmark Rust-backed maps against the legacy Numba dispatch path.

Run with::

    uv run python bench/maps_bench.py

Writes a Markdown row per benchmark into ``bench/RESULTS.md`` (creating
it if it doesn't exist). The N1 acceptance gate is ≥ 2× speedup on
Henon at 1e7 steps.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Re-import after sys.path adjustment so the in-tree package wins.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tsdynamics as ts  # noqa: E402

BENCHES: list[tuple[str, int]] = [
    ("Henon", 10_000_000),
    ("Logistic", 10_000_000),
    ("Ikeda", 1_000_000),       # transcendental ops; smaller N
    ("FoldedTowel", 1_000_000), # 3-D
]


def _bench_once(sys_obj, steps: int) -> float:
    """Time one ``iterate`` call. Returns wall-clock seconds."""
    ic = np.full(sys_obj.dim, 0.1)
    t0 = time.perf_counter()
    _ = sys_obj.iterate(steps=steps, ic=ic.copy())
    return time.perf_counter() - t0


def _bench_rust(name: str, steps: int) -> float:
    cls = getattr(ts, name)
    inst = cls()
    # Warm up the IR cache + verify Rust path is in use.
    assert inst._compile_ir() is not None
    # First run includes any lazy work; do a tiny warmup then time.
    _ = inst.iterate(steps=128, ic=np.full(inst.dim, 0.1))
    return _bench_once(inst, steps)


def _bench_numba(name: str, steps: int) -> float:
    cls = getattr(ts, name)
    inst = cls()
    # Force the fallback path by poisoning the IR cache for this (class, hash).
    key = (type(inst).__name__, inst.params.param_hash())
    type(inst)._ir_cache[key] = None
    # Warm Numba's JIT.
    _ = inst.iterate(steps=128, ic=np.full(inst.dim, 0.1))
    return _bench_once(inst, steps)


def main() -> None:
    """Run all map benchmarks and append results to ``bench/RESULTS.md``."""
    print(f"{'Map':<14} {'Steps':>12} {'Rust (s)':>10} {'Numba (s)':>11} {'Speedup':>9}")
    print("-" * 60)
    rows = []
    for name, steps in BENCHES:
        t_rust = _bench_rust(name, steps)
        t_numba = _bench_numba(name, steps)
        speedup = t_numba / t_rust if t_rust > 0 else float("inf")
        rows.append((name, steps, t_rust, t_numba, speedup))
        print(
            f"{name:<14} {steps:>12} {t_rust:>10.3f} {t_numba:>11.3f} {speedup:>8.2f}x"
        )

    # Append to bench/RESULTS.md
    out = Path(__file__).resolve().parent / "RESULTS.md"
    if not out.exists():
        out.write_text(
            "# Map benchmarks\n\n"
            "Wall-clock time for ``iterate(steps)`` on built-in maps.\n"
            "Rust column uses the N1 IR-interpreted kernel; Numba column\n"
            "forces the fallback path via the `_ir_cache = None` poison.\n\n"
        )
    with out.open("a") as f:
        f.write(f"## Run {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Map | Steps | Rust (s) | Numba (s) | Speedup |\n")
        f.write("|-----|------:|---------:|----------:|--------:|\n")
        for name, steps, t_rust, t_numba, speedup in rows:
            f.write(
                f"| {name} | {steps:,} | {t_rust:.3f} | {t_numba:.3f} | {speedup:.2f}× |\n"
            )
        f.write("\n")
    print(f"\nAppended results to {out}")


if __name__ == "__main__":
    main()
