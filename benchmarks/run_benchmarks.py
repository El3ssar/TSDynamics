"""Cross-library dynamical-systems benchmark — orchestrator.

Runs every benchmarked library on a shared, frozen set of canonical tasks
(integration, the Lyapunov family, correlation dimension, bifurcation diagram,
basins of attraction, fixed points, Poincaré section) and emits comparison
tables (Markdown + CSV + JSON). Each library runs in its own isolated process and
writes a JSON record; this orchestrator merges them, scores precision against
literature references, and renders the tables. Libraries that are not installed —
or tasks a library does not implement — render as blank cells, as the brief asks.

Pipeline:

1. Compute the integration-accuracy reference (SciPy DOP853 at 1e-13) and dump
   :mod:`config` to ``results/config.json`` (Python workers and the Julia script
   read identical parameters from it).
2. Spawn one worker per Python adapter (``runworker.py``) under the *current*
   interpreter, plus the external DynamicalSystems.jl Julia script, each with a
   wall-clock timeout.
3. Merge the per-library JSON records (adding documented unavailable columns for
   PyDSTool / TISEAN) and write ``RESULTS.md``, ``results/*.csv`` and the merged
   ``results/all.json``.

Usage::

    python run_benchmarks.py                  # full run
    python run_benchmarks.py --quick          # CI-sized
    python run_benchmarks.py --only lyapunov_spectrum,correlation_dimension
    python run_benchmarks.py --libs tsdynamics-interp,scipy --skip-julia
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import harness  # noqa: E402
from adapters import REGISTRY  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Libraries we considered but that do not install/run in this environment. They
# get documented all-blank columns so the comparison records that they were
# evaluated (per the brief's "leave it blank" instruction).
UNAVAILABLE_COLUMNS: list[dict[str, Any]] = [
    {
        "id": "pydstool",
        "name": "PyDSTool",
        "language": "python",
        "version": "—",
        "available": False,
        "reason": "import fails on modern NumPy (removed numpy.distutils)",
        "tasks": {},
    },
    {
        "id": "tisean",
        "name": "TISEAN",
        "language": "C",
        "version": "—",
        "available": False,
        "reason": "no binaries present / legacy C+Fortran does not build with current toolchain",
        "tasks": {},
    },
]


def compute_accuracy_reference() -> list[float]:
    """Integrate Lorenz to ``T_ACC`` at ultra-tight tolerance (the ∞-norm anchor)."""
    import numpy as np
    from scipy.integrate import solve_ivp

    p = config.LORENZ_PARAMS

    def rhs(_t: float, u: np.ndarray) -> list[float]:
        x, y, z = u
        return [p["sigma"] * (y - x), x * (p["rho"] - z) - y, x * y - p["beta"] * z]

    sol = solve_ivp(
        rhs,
        (0.0, config.T_ACC),
        config.LORENZ_IC,
        method="DOP853",
        rtol=config.ACC_REF_RTOL,
        atol=config.ACC_REF_ATOL,
    )
    return [float(v) for v in sol.y[:, -1]]


def dump_series() -> dict[str, str]:
    """Dump the shared from-data series so the Julia script reads identical input.

    ``series`` is deterministic (fixed SciPy DOP853 / NumPy loop), so the dumped
    files equal what the Python adapters regenerate in-process — every library,
    Python or Julia, analyses byte-identical numbers.
    """
    import numpy as np
    import series

    RESULTS.mkdir(parents=True, exist_ok=True)
    x = series.lorenz_series()
    corr = x[: config.CORR_N]
    lyap = x[:: config.LYAP_STRIDE][: config.LYAP_N]
    henon = series.henon_series(8000)
    files = {
        "lorenz_corr": str(RESULTS / "series_lorenz_corr.txt"),
        "lorenz_lyap": str(RESULTS / "series_lorenz_lyap.txt"),
        "henon": str(RESULTS / "series_henon.txt"),
    }
    np.savetxt(files["lorenz_corr"], corr)
    np.savetxt(files["lorenz_lyap"], lyap)
    np.savetxt(files["henon"], henon)
    return files


def write_config() -> Path:
    """Dump the shared config (with the injected accuracy reference) to JSON."""
    cfg = config.as_dict()
    cfg["references"]["lorenz_acc_final"] = compute_accuracy_reference()
    cfg["series_files"] = dump_series()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def run_python_worker(
    adapter_id: str, cfg_path: Path, *, quick: bool, only: str | None, timeout: float
) -> dict[str, Any]:
    """Run one Python adapter in a subprocess; return its JSON record."""
    out = RESULTS / f"{adapter_id}.json"
    cmd = [
        sys.executable,
        str(HERE / "runworker.py"),
        "--adapter",
        adapter_id,
        "--config",
        str(cfg_path),
        "--out",
        str(out),
    ]
    if quick:
        cmd.append("--quick")
    if only:
        cmd += ["--only", only]
    print(f"\n=== {adapter_id} ===", flush=True)
    try:
        subprocess.run(cmd, cwd=str(HERE), timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"[{adapter_id}] TIMEOUT after {timeout}s", file=sys.stderr)
    if out.exists():
        return json.loads(out.read_text())
    cls, _ = REGISTRY[adapter_id]
    return {
        "id": adapter_id,
        "name": adapter_id,
        "language": "python",
        "version": "—",
        "available": False,
        "reason": "worker produced no output (timeout/crash)",
        "tasks": {},
    }


def run_julia(cfg_path: Path, *, quick: bool, only: str | None, timeout: float) -> dict[str, Any]:
    """Run the DynamicalSystems.jl Julia script; return its JSON record (or a stub)."""
    julia = shutil.which("julia")
    out = RESULTS / "dynamicalsystems_jl.json"
    proj = HERE / "julia"
    script = proj / "bench.jl"
    stub = {
        "id": "dynamicalsystems-jl",
        "name": "DynamicalSystems.jl",
        "language": "julia",
        "version": "—",
        "available": False,
        "tasks": {},
    }
    if julia is None or not script.exists():
        stub["reason"] = "julia not found" if julia is None else "bench.jl missing"
        return stub
    cmd = [julia, f"--project={proj}", str(script), "--config", str(cfg_path), "--out", str(out)]
    if quick:
        cmd.append("--quick")
    if only:
        cmd += ["--only", only]
    print("\n=== DynamicalSystems.jl (julia) ===", flush=True)
    try:
        subprocess.run(cmd, cwd=str(HERE), timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"[julia] TIMEOUT after {timeout}s", file=sys.stderr)
    if out.exists():
        return json.loads(out.read_text())
    stub["reason"] = "julia worker produced no output (timeout/crash)"
    return stub


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="CI-sized inputs, fewer reps")
    ap.add_argument("--only", default=None, help="comma-separated task keys to run")
    ap.add_argument("--libs", default=None, help="comma-separated adapter ids (default: all)")
    ap.add_argument("--skip-julia", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-library timeout (s)")
    args = ap.parse_args(argv)

    cfg_path = write_config()
    references = json.loads(cfg_path.read_text())["references"]

    lib_ids = args.libs.split(",") if args.libs else list(REGISTRY)
    libraries: list[dict[str, Any]] = []
    for aid in lib_ids:
        if aid not in REGISTRY:
            print(f"unknown adapter id {aid!r}; known: {list(REGISTRY)}", file=sys.stderr)
            continue
        libraries.append(
            run_python_worker(aid, cfg_path, quick=args.quick, only=args.only, timeout=args.timeout)
        )

    if not args.skip_julia and (args.libs is None or "dynamicalsystems-jl" in (args.libs or "")):
        libraries.append(
            run_julia(cfg_path, quick=args.quick, only=args.only, timeout=args.timeout)
        )

    libraries.extend(UNAVAILABLE_COLUMNS)

    merged = harness.Merged(
        libraries=libraries,
        references=references,
        meta={
            "quick": args.quick,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu": platform.processor() or "—",
        },
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "all.json").write_text(
        json.dumps(
            {"meta": merged.meta, "references": references, "libraries": libraries}, indent=2
        )
    )
    _write_csv(merged)
    report = _render_report(merged)
    (HERE / "RESULTS.md").write_text(report)
    print("\n" + report)
    print(f"\nwrote {HERE / 'RESULTS.md'}, {RESULTS / 'all.json'}, {RESULTS}/*.csv")
    return 0


def _write_csv(merged: harness.Merged) -> None:
    """Write speed + precision CSVs (rows=tasks, cols=libraries)."""
    import csv

    names = [lib["name"] for lib in merged.libraries]
    with open(RESULTS / "speed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "unit=seconds", *names])
        for task in harness.TASKS:
            row = [task.title, ""]
            for lib in merged.libraries:
                cell = lib.get("tasks", {}).get(task.key)
                row.append("" if not cell or cell.get("status") != "ok" else cell["seconds"])
            w.writerow(row)
    with open(RESULTS / "precision.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "reference", *names])
        for task in harness.TASKS:
            if not task.precision:
                continue
            ref = merged.references.get(task.reference_key) if task.reference_key else ""
            row = [task.title, ref]
            for lib in merged.libraries:
                cell = lib.get("tasks", {}).get(task.key)
                row.append("" if not cell or cell.get("status") != "ok" else cell["estimate"])
            w.writerow(row)


def _render_report(merged: harness.Merged) -> str:
    m = merged.meta
    parts = [
        "# Cross-library dynamical-systems benchmark — results\n",
        "_Auto-generated by `benchmarks/run_benchmarks.py`. Each cell is the "
        "best-of-N wall time (minimum); blanks mean the library does not provide "
        "that capability or is not installed here._\n",
        f"- **Platform:** {m.get('platform', '—')}",
        f"- **Python:** {m.get('python', '—')}  ·  **mode:** "
        f"{'quick' if m.get('quick') else 'full'}\n",
        "## Library availability\n",
        harness.availability_table(merged),
        "\n## Speed (best-of-N wall time; lower is better)\n",
        harness.speed_table(merged),
        "\n## Precision (estimate, and Δ from the literature reference)\n",
        harness.precision_table(merged),
        "\n_See `benchmarks/README.md` for the methodology and per-task protocol._",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
