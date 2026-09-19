#!/usr/bin/env python3
"""Order-of-magnitude regression gate for the Rust engine's criterion benches.

The engine's performance story — FSAL stage reuse, frozen-Jacobian/LU reuse,
interpreter dead-register elimination, the compiled-evaluator cache — is
answer-preserving, so the correctness suite cannot see any of it. The criterion
benches under ``crates/*/benches/**`` measure it; this script is what turns those
measurements into a CI signal.

What it does
------------
Reads criterion's per-case ``estimates.json`` (written under
``crates/target/criterion/<group>/<case>/new/``), takes each case's **median**
point estimate, and compares it with the ceiling committed in
``benchmarks/engine_bench_baseline.json``.

What it catches, and what it does not
-------------------------------------
The ceilings are 10x the reference measurement, so this is an
**order-of-magnitude** gate and nothing finer. That is a deliberate trade: wall
clock on a shared CI runner varies by 2-3x between runs, so a tight threshold
would flake, and a flaky blocking gate gets disabled and then ignored. A 10x
ceiling still catches the regressions that matter — a cache that stopped caching
(a JIT hit reverting to a full compile is ~1000x), an allocation added to a hot
loop, a lost fast path.

It does **not** catch a lost FSAL reuse (~8-18% of an rk45 step) or a lost
liveness mask (~25% of an RHS evaluation on a Jacobian-bearing tape). Those are
pinned by deterministic *counting* tests in the Rust crates, which cannot flake;
see the notes in the baseline JSON.

A **missing** case is also a failure: it means a bench was renamed or deleted
without updating the baseline, which would silently remove the coverage.

Usage
-----
    cd crates && cargo bench --workspace
    python benchmarks/check_engine_bench.py crates/target/criterion
    python benchmarks/check_engine_bench.py --update crates/target/criterion
    python benchmarks/check_engine_bench.py --summary "$GITHUB_STEP_SUMMARY" ...

Standard library only: it runs in the Rust CI job, which has no Python
environment beyond the runner's interpreter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The committed ceilings, next to this script.
BASELINE = Path(__file__).with_name("engine_bench_baseline.json")


def _fmt(ns: float) -> str:
    """Human-readable duration from nanoseconds."""
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3f} {unit}"
    return f"{ns:.1f} ns"


def collect(criterion_dir: Path) -> dict[str, float]:
    """Return ``{full_id: median_ns}`` for every case criterion just measured.

    Only ``new/`` results are read (``base/`` is the previous run criterion keeps
    for its own comparison). The **median** is used rather than the mean: it is
    the estimate least disturbed by the occasional descheduled sample that a
    shared runner produces.
    """
    out: dict[str, float] = {}
    for est in sorted(criterion_dir.glob("*/*/new/estimates.json")):
        meta = est.with_name("benchmark.json")
        if not meta.is_file():
            continue
        full_id = json.loads(meta.read_text())["full_id"]
        out[full_id] = json.loads(est.read_text())["median"]["point_estimate"]
    return out


def update(criterion_dir: Path, measured: dict[str, float]) -> None:
    """Rewrite the baseline from a fresh run, keeping the multiplier and notes."""
    doc = json.loads(BASELINE.read_text())
    mult = float(doc["multiplier"])
    doc["cases"] = {
        name: {
            "reference_ns": round(ns, 1),
            "ceiling_ns": round(ns * mult, 1),
        }
        for name, ns in sorted(measured.items())
    }
    BASELINE.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {len(doc['cases'])} cases to {BASELINE}")


def main() -> int:
    """Compare a criterion run with the committed ceilings; 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "criterion_dir",
        type=Path,
        nargs="?",
        default=Path("crates/target/criterion"),
        help="criterion's output directory (default: crates/target/criterion)",
    )
    ap.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from this run instead of checking against it",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="also append a markdown table here (e.g. $GITHUB_STEP_SUMMARY)",
    )
    args = ap.parse_args()

    if not args.criterion_dir.is_dir():
        print(f"error: no criterion output at {args.criterion_dir}", file=sys.stderr)
        print("run `cd crates && cargo bench --workspace` first", file=sys.stderr)
        return 2

    measured = collect(args.criterion_dir)
    if not measured:
        print(f"error: {args.criterion_dir} holds no bench results", file=sys.stderr)
        return 2

    if args.update:
        update(args.criterion_dir, measured)
        return 0

    doc = json.loads(BASELINE.read_text())
    cases: dict[str, dict[str, float]] = doc["cases"]

    rows: list[tuple[str, str, str, str, str]] = []
    failures: list[str] = []

    for name, spec in sorted(cases.items()):
        ceiling = float(spec["ceiling_ns"])
        if name not in measured:
            failures.append(f"{name}: MISSING from this run (bench renamed or deleted?)")
            rows.append((name, "-", _fmt(ceiling), "-", "MISSING"))
            continue
        got = measured[name]
        ratio = got / ceiling
        ok = got <= ceiling
        if not ok:
            failures.append(
                f"{name}: {_fmt(got)} exceeds the {_fmt(ceiling)} ceiling ({ratio:.1f}x over)"
            )
        rows.append(
            (
                name,
                _fmt(got),
                _fmt(ceiling),
                f"{100 * ratio:.0f}%",
                "ok" if ok else "OVER",
            )
        )

    extra = sorted(set(measured) - set(cases))

    width = max(len(r[0]) for r in rows)
    print(f"{'case'.ljust(width)}  {'measured':>12}  {'ceiling':>12}  {'used':>6}  status")
    for name, got, ceiling, used, status in rows:
        print(f"{name.ljust(width)}  {got:>12}  {ceiling:>12}  {used:>6}  {status}")
    for name in extra:
        print(
            f"{name.ljust(width)}  {_fmt(measured[name]):>12}  {'-':>12}  {'-':>6}  new (not gated)"
        )

    if args.summary is not None:
        lines = [
            "### Engine bench vs committed ceilings",
            "",
            "Order-of-magnitude gate (ceiling = 10x reference). See",
            "`benchmarks/engine_bench_baseline.json` for what it does and does not catch.",
            "",
            "| case | measured | ceiling | used | status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        lines += [f"| `{n}` | {g} | {c} | {u} | {s} |" for n, g, c, u, s in rows]
        lines += [f"| `{n}` | {_fmt(measured[n])} | - | - | new (not gated) |" for n in extra]
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nA case is only flagged at 10x, so this is very unlikely to be runner "
            "noise. Profile the named case, or — if the slowdown is intended — raise "
            "its ceiling in benchmarks/engine_bench_baseline.json and say why in the PR.",
            file=sys.stderr,
        )
        return 1

    print("\nall cases within their ceilings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
