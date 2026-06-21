"""Performance-regression harness for the TSDynamics analysis layer.

This benchmark times a representative spread of the analysis toolkit on
**fixed, deterministic inputs** so that a wall-clock number is comparable from
run to run. It is the analysis-layer counterpart of the (now historical) engine
benchmark whose decision record lives in ``benches/REPORT.md``: where that one
answered "is the Rust engine fast enough to be the default", this one answers
"did an analysis routine just get slower".

Design notes
------------
- **Fixed inputs.** Every case runs on the same seeded array / system, built
  once in :func:`build_inputs`. A Lorenz ``x`` series (fixed ``final_time`` /
  ``dt``, transient dropped) feeds the data-driven quantifiers; the Hénon and
  Logistic maps feed the map analyses. No call draws fresh randomness without a
  pinned ``seed=``, so the *work done* is identical across runs and only the
  *time taken* varies.
- **Robust timing.** Each case is timed ``repeat`` times and we keep the
  **minimum** (the cleanest sample — the machine can only ever be slower than
  the true cost, never faster), following the standard ``timeit`` guidance.
- **Self-contained.** Only the public ``tsdynamics`` API is used, so the harness
  exercises exactly what a user calls. It runs on the compiled engine when
  present and on the pure-Python reference backend otherwise (the data-driven
  analyses need no engine at all).

Usage
-----
    uv run python benches/analysis_bench.py                 # full, pretty table
    uv run python benches/analysis_bench.py --quick         # CI-sized, fewer reps
    uv run python benches/analysis_bench.py --out perf.json # machine-readable
    uv run python benches/analysis_bench.py --case rqa      # one case (substring)

The JSON schema is::

    {
      "meta": {"quick": bool, "repeat": int, "python": str, "engine": str|null,
               "platform": str, "tsdynamics": str},
      "cases": {"<name>": {"seconds": float, "repeat": int, "n": int}, ...}
    }

``perf-analysis.yml`` consumes this JSON: it records the ``cases[*].seconds`` as
a baseline and flags any case that regresses beyond a tolerance. The gate is
**advisory, not blocking** — wall-clock numbers on shared CI runners are noisy.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import tsdynamics as ts

# --------------------------------------------------------------------------- #
# Fixed inputs (built once, deterministic)
# --------------------------------------------------------------------------- #


@dataclass
class Inputs:
    """The fixed inputs every benchmark case shares."""

    lorenz: Any
    henon: Any
    logistic: Any
    x: np.ndarray  # Lorenz x-component, transient dropped
    x_full: np.ndarray  # full Lorenz x-component


def build_inputs(*, quick: bool) -> Inputs:
    """Build the fixed, deterministic inputs for the benchmark.

    Parameters
    ----------
    quick : bool
        If ``True``, use shorter integrations / series so the whole suite runs
        in a CI-friendly wall-clock budget.

    Returns
    -------
    Inputs
        The shared systems and the seeded Lorenz ``x`` series.
    """
    final_time = 40.0 if quick else 80.0
    lorenz = ts.systems.Lorenz()
    # Pin the initial condition so the series is byte-identical run to run —
    # otherwise the default random IC makes the benchmark measure different
    # *work* each time and a "regression" would be meaningless.
    traj = lorenz.integrate(ic=[1.0, 1.0, 1.0], final_time=final_time, dt=0.01)
    x_full = np.ascontiguousarray(np.asarray(traj["x"]), dtype=float)
    # Drop the first ~20 time units of transient so the series is on-attractor.
    x = x_full[2000:]
    return Inputs(
        lorenz=lorenz,
        henon=ts.systems.Henon(),
        logistic=ts.systems.Logistic(),
        x=x,
        x_full=x_full,
    )


# --------------------------------------------------------------------------- #
# Case registry
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    """A single benchmark case: a named, fixed-input analysis call."""

    name: str
    #: builds the zero-argument callable to time, given the shared inputs and
    #: the ``quick`` flag (so a case can shrink its own work in quick mode).
    make: Callable[[Inputs, bool], Callable[[], object]]
    #: the input size this case reports (for context in the JSON / table).
    size: Callable[[Inputs, bool], int] = field(default=lambda _i, _q: 0)


def _series_len(quick: bool, full: int, q: int) -> int:
    return q if quick else full


def all_cases() -> list[Case]:
    """Build the representative analysis cases, one per major analysis family.

    Returns
    -------
    list of Case
        Data-driven quantifiers (dimensions / entropy / recurrence / surrogate
        / data-Lyapunov) plus system-driven analyses (map Lyapunov, 0-1 test,
        fixed points, orbit diagram). Each is a headline routine of its
        subpackage, so a slowdown anywhere in the layer is likely to surface in
        at least one case.
    """

    def n_dim(quick: bool) -> int:
        return _series_len(quick, 3000, 1500)

    def n_recur(quick: bool) -> int:
        return _series_len(quick, 1500, 900)

    return [
        # --- dimensions (A-DIM) ---
        Case(
            "correlation_dimension",
            lambda i, q: lambda: ts.correlation_dimension(i.x[: n_dim(q)]),
            lambda i, q: n_dim(q),
        ),
        # --- entropy (A-ENT) ---
        Case(
            "permutation_entropy",
            lambda i, q: lambda: ts.permutation_entropy(i.x[: n_dim(q)]),
            lambda i, q: n_dim(q),
        ),
        Case(
            "sample_entropy",
            lambda i, q: lambda: ts.sample_entropy(i.x[: (1500 if q else 2000)]),
            lambda i, q: 1500 if q else 2000,
        ),
        # --- recurrence / RQA (A-RQA) ---
        Case(
            "recurrence_matrix",
            lambda i, q: lambda: ts.recurrence_matrix(i.x[: n_recur(q)], recurrence_rate=0.05),
            lambda i, q: n_recur(q),
        ),
        Case(
            "rqa",
            lambda i, q: lambda: ts.rqa(i.x[: n_recur(q)], recurrence_rate=0.05),
            lambda i, q: n_recur(q),
        ),
        # --- surrogate / nonlinearity (A-SURR) ---
        Case(
            "surrogate_test",
            lambda i, q: lambda: ts.surrogate_test(i.x[: n_recur(q)], n=(9 if q else 19), seed=0),
            lambda i, q: n_recur(q),
        ),
        # --- data-driven Lyapunov (A-LYAP) ---
        Case(
            "lyapunov_from_data",
            lambda i, q: lambda: ts.lyapunov_from_data(i.x[: (2000 if q else 3000)], dt=0.01),
            lambda i, q: 2000 if q else 3000,
        ),
        # --- system-driven Lyapunov (A-LYAP / C-DERIV) ---
        Case(
            "lyapunov_spectrum_map",
            lambda i, q: lambda: ts.lyapunov_spectrum(i.henon, n=(2000 if q else 4000)),
            lambda i, q: 2000 if q else 4000,
        ),
        # --- chaos indicators (A-CHAOS) ---
        Case(
            "zero_one_test",
            lambda i, q: lambda: ts.zero_one_test(i.logistic, n=(1500 if q else 2000), seed=0),
            lambda i, q: 1500 if q else 2000,
        ),
        # --- fixed points (A-FP) ---
        Case(
            "fixed_points_map",
            lambda i, q: lambda: ts.fixed_points(i.henon, seed=0),
            lambda i, q: 0,
        ),
        # --- orbit diagram (A-ORBIT) ---
        Case(
            "orbit_diagram_map",
            lambda i, q: (
                lambda: ts.orbit_diagram(
                    i.henon,
                    "a",
                    np.linspace(1.0, 1.4, 30 if q else 60),
                    n=100,
                    transient=200,
                )
            ),
            lambda i, q: 30 if q else 60,
        ),
    ]


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def time_case(fn: Callable[[], object], *, repeat: int) -> float:
    """Time ``fn`` ``repeat`` times and return the minimum elapsed seconds.

    The minimum is the most reproducible estimator of the intrinsic cost: the
    machine can only add noise (other processes, GC, frequency scaling), never
    remove it, so the fastest sample is the closest to the true work.

    Parameters
    ----------
    fn : callable
        The zero-argument call to time.
    repeat : int
        Number of repetitions (>= 1).

    Returns
    -------
    float
        The minimum wall-clock seconds over the repetitions.
    """
    best = float("inf")
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _engine_version() -> str | None:
    try:
        import tsdynamics._rust as _rust  # type: ignore[import-not-found]

        return str(_rust._version())
    except Exception:
        return None


def run(*, quick: bool, repeat: int, only: str | None = None) -> dict[str, Any]:
    """Run the benchmark suite and return the results dict.

    Parameters
    ----------
    quick : bool
        CI-sized inputs and fewer repetitions.
    repeat : int
        Repetitions per case (the minimum is kept).
    only : str or None
        If given, run only cases whose name contains this substring.

    Returns
    -------
    dict
        The results, matching the JSON schema documented in the module
        docstring.
    """
    inputs = build_inputs(quick=quick)
    cases = all_cases()
    if only is not None:
        cases = [c for c in cases if only in c.name]
        if not cases:
            raise SystemExit(f"no benchmark case matches {only!r}")

    results: dict[str, Any] = {}
    for case in cases:
        fn = case.make(inputs, quick)
        # One warm-up call (outside timing) so lazy lowering / caches are paid
        # once and the timed runs measure steady-state work.
        fn()
        seconds = time_case(fn, repeat=repeat)
        results[case.name] = {
            "seconds": seconds,
            "repeat": repeat,
            "n": int(case.size(inputs, quick)),
        }

    return {
        "meta": {
            "quick": quick,
            "repeat": repeat,
            "python": sys.version.split()[0],
            "engine": _engine_version(),
            "platform": platform.platform(),
            "tsdynamics": getattr(ts, "__version__", "unknown"),
        },
        "cases": results,
    }


def compare(
    baseline: dict[str, Any], current: dict[str, Any], *, tolerance: float
) -> dict[str, dict[str, Any]]:
    """Compare a current run against a baseline and flag regressions.

    A case is flagged ``regressed`` when its current time exceeds the baseline
    time by more than ``tolerance`` (a fractional slowdown — ``0.5`` means "more
    than 50% slower than baseline"). Cases present in only one of the two runs
    are reported as ``"added"`` / ``"removed"`` and never count as a regression.

    Parameters
    ----------
    baseline : dict
        A results dict (from :func:`run`) treated as the reference.
    current : dict
        The freshly measured results dict.
    tolerance : float
        Fractional slowdown above which a case is flagged (>= 0).

    Returns
    -------
    dict
        Maps each case name to ``{"baseline", "current", "ratio", "status"}``
        where ``status`` is ``"ok"`` / ``"regressed"`` / ``"added"`` /
        ``"removed"`` and ``ratio`` is ``current / baseline`` (or ``None`` when
        a side is missing).
    """
    base_cases = baseline.get("cases", {})
    cur_cases = current.get("cases", {})
    report: dict[str, dict[str, Any]] = {}
    for name in sorted(set(base_cases) | set(cur_cases)):
        b = base_cases.get(name, {}).get("seconds")
        c = cur_cases.get(name, {}).get("seconds")
        if b is None:
            report[name] = {"baseline": None, "current": c, "ratio": None, "status": "added"}
            continue
        if c is None:
            report[name] = {"baseline": b, "current": None, "ratio": None, "status": "removed"}
            continue
        ratio = c / b if b > 0 else float("inf")
        status = "regressed" if ratio > 1.0 + tolerance else "ok"
        report[name] = {"baseline": b, "current": c, "ratio": ratio, "status": status}
    return report


def format_table(data: dict[str, Any]) -> str:
    """Render the results dict as a Markdown table.

    Parameters
    ----------
    data : dict
        The results from :func:`run`.

    Returns
    -------
    str
        A Markdown table (header + one row per case).
    """
    lines = [
        "| analysis | n | best of {} (ms) |".format(data["meta"]["repeat"]),
        "|---|---:|---:|",
    ]
    for name, row in data["cases"].items():
        n = row["n"] or "—"
        lines.append(f"| {name} | {n} | {row['seconds'] * 1e3:.1f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str or None
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="CI-sized inputs and fewer repetitions"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="repetitions per case (default: 5, or 3 with --quick)",
    )
    parser.add_argument("--out", type=str, default=None, help="write JSON results here")
    parser.add_argument(
        "--case", type=str, default=None, help="run only cases whose name contains this substring"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="compare against this baseline JSON and print a regression table",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="fractional slowdown above which a case is flagged (default: 0.5)",
    )
    args = parser.parse_args(argv)

    repeat = args.repeat if args.repeat is not None else (3 if args.quick else 5)
    data = run(quick=args.quick, repeat=repeat, only=args.case)

    print(format_table(data))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nwrote {args.out}")

    if args.baseline:
        with open(args.baseline) as fh:
            baseline = json.load(fh)
        report = compare(baseline, data, tolerance=args.tolerance)
        print(f"\n### Regression check (tolerance {args.tolerance:+.0%}, advisory)\n")
        print("| analysis | baseline (ms) | current (ms) | ratio | status |")
        print("|---|---:|---:|---:|---|")
        for name, row in report.items():
            b = "—" if row["baseline"] is None else f"{row['baseline'] * 1e3:.1f}"
            c = "—" if row["current"] is None else f"{row['current'] * 1e3:.1f}"
            r = "—" if row["ratio"] is None else f"{row['ratio']:.2f}x"
            flag = "⚠️ " if row["status"] == "regressed" else ""
            print(f"| {name} | {b} | {c} | {r} | {flag}{row['status']} |")
        regressed = [n for n, r in report.items() if r["status"] == "regressed"]
        if regressed:
            print(f"\n{len(regressed)} case(s) regressed: {', '.join(regressed)} (advisory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
