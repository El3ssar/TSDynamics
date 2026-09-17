"""A/B harness for the matplotlib animation writer (frames per second).

Answers one question: **how long does it take to write a movie**, with and
without the frame compositor in
:mod:`tsdynamics.viz.render.mpl._anim`, on the same machine, in the same
process, interleaved.

Where ``analysis_bench.py`` times the analysis layer and ``check_engine_bench.py``
gates the Rust engine, this one covers the third hot path a user waits on: the
render loop behind ``plot.save("attractor.mp4")``.

Design notes
------------
- **Warm up, then min-of-N, interleaved.** This is the whole methodology, and
  getting it wrong is how a benchmark lies in your favour.  A first measurement in
  a fresh process pays costs that belong to *no* arm — matplotlib's font-cache
  scan, the Agg backend's first render, the first ``ffmpeg`` spawn.  Measured here,
  the same unoptimised 60-frame ``.mp4`` took ``11.39 s`` on its first run and
  ``6.74 / 7.01 / 7.12 s`` afterwards: a **1.64x** inflation, and it landed
  entirely on ``before``, because ``before`` ran first.  So each arm gets a
  discarded warm-up, then ``--repeat`` timed runs, alternating ``before`` /
  ``after`` so a machine that gets busy slows both — and the **minimum** is
  reported, since noise only ever adds time.  ``CLAUDE.md`` already taught this
  for backend timing ("pin ``ic``, interleave, min-of-25"); it applies here too.
- **``before`` is the real pre-compositor path**: the compositor disabled through
  ``TSDYNAMICS_NO_BLIT`` *and* matplotlib's own ``_post_draw`` (the discarded full
  draw per frame) restored.
- **Pinned inputs.** Every trajectory names its ``ic``; a system with no declared
  initial condition draws a fresh random one per call, which silently compares
  two different amounts of work (the same trap documented for backend timing in
  ``CLAUDE.md``).
- **Real writers.** ``.mp4`` goes through ffmpeg and ``.gif`` through pillow, so
  the encode cost is in the number a user actually pays.  The frame *count* is
  the knob that matters: the compositor's calibration is a fixed cost, so short
  movies show a smaller factor than the 360-frame default.
- **Correctness is not here.** ``tests/test_viz_anim_fast.py`` owns the
  byte-identity proof; this file only measures.

Usage
-----
    uv run python benchmarks/animation_bench.py                  # default table
    uv run python benchmarks/animation_bench.py --frames 360     # the real default
    uv run python benchmarks/animation_bench.py --repeat 5       # steadier numbers
    uv run python benchmarks/animation_bench.py --format .gif
    uv run python benchmarks/animation_bench.py --kind 2d --kind 3d
    uv run python benchmarks/animation_bench.py --out anim.json

The JSON schema is::

    {
      "meta": {"frames": int, "format": str, "repeat": int, "python": str,
               "platform": str, "tsdynamics": str, "matplotlib": str},
      "cases": {"<kind>": {"before_s": float, "after_s": float,
                           "before_spread": float, "after_spread": float,
                           "before_fps": float, "after_fps": float,
                           "speedup": float, "blitting": bool|null}, ...}
    }

``*_s`` is the minimum over the timed runs and ``*_spread`` the max/min ratio
within an arm — a spread far above 1 means the machine was too busy to trust the
row.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import tsdynamics as ts

#: Every animation kind the renderer has, in the order the table prints them.
#:
#: The list is deliberately exhaustive: the three that gain least — ``spin`` and
#: ``frames`` (both un-blittable by construction) and ``fade`` — are exactly the
#: ones a table would flatter itself by omitting.
KINDS = ("2d", "3d", "series", "composite", "field", "frames", "spin", "fade", "clock")


def _lorenz() -> Any:
    """Return a pinned Lorenz orbit (a random IC would compare two workloads)."""
    return ts.systems.Lorenz().run(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0])


def build_spec(kind: str, n_frames: int) -> Any:
    """Build one animated :class:`~tsdynamics.viz.spec.Plot` for ``kind``."""
    animate = {"n_frames": n_frames}
    if kind == "2d":
        return ts.plot(_lorenz(), components=("x", "z"), animate=animate)
    if kind == "3d":
        return ts.plot(_lorenz(), animate=animate)
    if kind == "series":
        return ts.plot(_lorenz(), components="x", animate=animate)
    if kind == "composite":
        traj = _lorenz()
        return ts.viz.plot(
            ts.plot(traj, components=("x", "z")),
            ts.plot(traj, components="x"),
            layout="row",
            animate=animate,
        )
    if kind == "field":
        traj = ts.systems.SwiftHohenberg(N=48).run(final_time=5.0, dt=0.05)
        return ts.plot(traj, "spatial_field", animate=animate)
    if kind == "spin":
        return ts.plot(_lorenz(), animate=animate).camera(spin=90.0)
    if kind == "fade":
        return ts.plot(_lorenz(), animate=animate).trail(length=("steps", 200), fade=True)
    if kind == "clock":
        return ts.plot(_lorenz(), components=("x", "z"), animate=animate).clock()
    if kind == "frames":
        logistic = ts.systems.Logistic()
        panels = [ts.plot(logistic.with_params(r=3.2 + 0.1 * k), "cobweb") for k in range(8)]
        composite = ts.viz.plot(*panels, layout="row", animate=animate)
        return dataclasses.replace(
            composite, layout=dataclasses.replace(composite.layout, mode="frames")
        )
    raise SystemExit(f"unknown kind {kind!r}; choose from {', '.join(KINDS)}")


def time_one(
    kind: str, n_frames: int, suffix: str, *, optimized: bool
) -> tuple[float, bool | None]:
    """Write one movie and return ``(seconds, blitting)``."""
    from matplotlib.animation import FuncAnimation

    from tsdynamics.viz.render.mpl import _anim

    if optimized:
        os.environ.pop("TSDYNAMICS_NO_BLIT", None)
    else:
        os.environ["TSDYNAMICS_NO_BLIT"] = "1"
    spec = build_spec(kind, n_frames)
    anim = _anim.render_animation(spec)
    if not optimized:
        anim._post_draw = FuncAnimation._post_draw.__get__(anim)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"bench{suffix}"
        start = time.perf_counter()
        anim.save(str(out))
        elapsed = time.perf_counter() - start
    compositor = getattr(anim, "_tsd_compositor", None)
    blitting = None if compositor is None else bool(compositor.blitting)
    anim._fig.clear()
    del anim, spec
    gc.collect()
    os.environ.pop("TSDYNAMICS_NO_BLIT", None)
    return elapsed, blitting


def run(kinds: list[str], n_frames: int, suffix: str, repeat: int = 3) -> dict[str, dict[str, Any]]:
    """Time every kind: one discarded warm-up per arm, then ``repeat`` interleaved runs.

    The warm-up is not politeness — see the module docstring.  Without it the
    first arm measured absorbs the process's one-time costs and this table
    over-reports the speedup by roughly 2-3x.
    """
    cases: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        time_one(kind, n_frames, suffix, optimized=False)  # warm-up, discarded
        time_one(kind, n_frames, suffix, optimized=True)  # warm-up, discarded
        befores: list[float] = []
        afters: list[float] = []
        blitting: bool | None = None
        for _ in range(repeat):
            befores.append(time_one(kind, n_frames, suffix, optimized=False)[0])
            elapsed, blitting = time_one(kind, n_frames, suffix, optimized=True)
            afters.append(elapsed)
        before, after = min(befores), min(afters)
        cases[kind] = {
            "before_s": before,
            "after_s": after,
            "before_spread": max(befores) / before,
            "after_spread": max(afters) / after,
            "before_fps": n_frames / before,
            "after_fps": n_frames / after,
            "speedup": before / after,
            "blitting": blitting,
        }
    return cases


def print_table(cases: dict[str, dict[str, Any]], n_frames: int, suffix: str, repeat: int) -> None:
    """Print the frames-per-second table (minimum of ``repeat`` timed runs per arm)."""
    print(f"\n{suffix}  {n_frames} frames  ·  min of {repeat} (after 1 discarded warm-up)\n")
    head = f"{'kind':<11s} {'before s':>9s} {'after s':>9s} {'before fps':>11s} "
    print(head + f"{'after fps':>10s} {'speedup':>8s} {'spread':>7s}  blit")
    print("-" * 80)
    for kind, row in cases.items():
        blit = {None: "exempt", True: "yes", False: "no"}[row["blitting"]]
        spread = max(row["before_spread"], row["after_spread"])
        print(
            f"{kind:<11s} {row['before_s']:9.2f} {row['after_s']:9.2f} "
            f"{row['before_fps']:11.1f} {row['after_fps']:10.1f} "
            f"{row['speedup']:7.2f}x {spread:6.2f}x  {blit}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60, help="frames per movie (default 60)")
    parser.add_argument(
        "--format", dest="suffix", default=".mp4", help="movie extension (default .mp4)"
    )
    parser.add_argument(
        "--kind", action="append", choices=KINDS, help="restrict to a kind (repeatable)"
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="timed runs per arm, after a warm-up (default 3)"
    )
    parser.add_argument("--out", type=str, default=None, help="write JSON results here")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")

    kinds = list(args.kind) if args.kind else list(KINDS)
    cases = run(kinds, args.frames, args.suffix, args.repeat)
    print_table(cases, args.frames, args.suffix, args.repeat)

    if args.out:
        data = {
            "meta": {
                "frames": args.frames,
                "format": args.suffix,
                "repeat": args.repeat,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "tsdynamics": ts.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "cases": cases,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
