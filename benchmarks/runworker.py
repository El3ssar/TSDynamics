"""Per-library benchmark worker.

Runs ONE adapter (all its supported tasks) in an isolated process and writes a
JSON record. Process isolation keeps a crash or a slow library from taking the
whole suite down, lets the orchestrator apply a wall-clock timeout per library,
and gives each library a clean interpreter (numba threading, import side effects).

Usage::

    python runworker.py --adapter nolds --config results/config.json \
        --out results/nolds.json [--quick] [--only correlation_dimension,...]

The JSON schema is::

    {"id", "name", "language", "version", "available", "reason",
     "tasks": {"<key>": {"status", "seconds", "estimate", "note"}}}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
from adapters import make_adapter  # noqa: E402


def run(adapter_id: str, config: dict, *, quick: bool, only: list[str] | None) -> dict:
    """Run all supported tasks for one adapter and return its result record."""
    adapter = make_adapter(adapter_id, config)
    record: dict = {
        "id": adapter_id,
        "name": adapter.name,
        "language": adapter.language,
        "version": adapter.version,
        "available": adapter.available,
        "reason": adapter.reason,
        "tasks": {},
    }
    if not adapter.available:
        return record

    tasks = [t for t in harness.TASKS if only is None or t.key in only]
    for task in tasks:
        cell: dict = {"status": "unsupported", "seconds": None, "estimate": None, "note": None}
        try:
            fn = adapter.build(task.key, quick=quick)
            if fn is not None:
                estimate = fn()  # warm-up (pays lowering/JIT) + captures the estimate
                repeat = task.repeat_quick if quick else task.repeat
                seconds = harness.best_of(fn, repeat=repeat)
                cell = {
                    "status": "ok",
                    "seconds": seconds,
                    "estimate": (None if estimate is None else float(estimate)),
                    "note": None,
                }
        except Exception as exc:  # a single task failing must not lose the rest
            cell = {
                "status": "error",
                "seconds": None,
                "estimate": None,
                "note": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{adapter_id}/{task.key}] ERROR: {exc}", file=sys.stderr)
            traceback.print_exc()
        record["tasks"][task.key] = cell
        ms = "" if cell["seconds"] is None else f"{cell['seconds'] * 1e3:.1f} ms"
        print(f"[{adapter_id}] {task.key}: {cell['status']} {ms}", file=sys.stderr)
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--only", default=None, help="comma-separated task keys")
    args = p.parse_args(argv)

    with open(args.config) as fh:
        config = json.load(fh)
    only = args.only.split(",") if args.only else None

    record = run(args.adapter, config, quick=args.quick, only=only)
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
