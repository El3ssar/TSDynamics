"""
Generate the landing-hero attractor data: one compact JSON per 3D system.

The streamline geometry is produced **by TSDynamics itself** — each system is
integrated with the library, then the 3-D phase portrait is lowered through
``ts.viz`` (the ``threejs`` data-export renderer) so the curve vertices come from
the library's own plotting path, not a hand-rolled projection. Those vertices are
then arc-length-resampled and radius-normalized so the hero canvas can rotate the
attractor at a uniform visual speed and auto-fit it without per-system tuning.

The JSONs are committed static assets; re-run after changing the roster:

    .venv/bin/python docs/_tooling/make_hero.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

import tsdynamics as ts

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "hero"
N = 2400  # points stored per system
RETRIES = 25  # random-IC retries if a run diverges / lands on a fixed point

# A curated set of 14 visually distinct 3-D attractors — butterfly, spiral-fold,
# spherical shell, cyclic labyrinth, multi-lobe, spiky, bursting, climate — not
# the whole catalogue. (system, final_time, dt)
ROSTER = [
    ("Lorenz", 70, 0.01),
    ("Rossler", 240, 0.02),
    ("Aizawa", 90, 0.01),
    ("Thomas", 320, 0.04),
    ("Halvorsen", 90, 0.01),
    ("Dadras", 70, 0.01),
    ("Chen", 60, 0.005),
    ("RabinovichFabrikant", 140, 0.01),
    ("NoseHoover", 170, 0.02),
    ("Arneodo", 130, 0.01),
    ("ShimizuMorioka", 170, 0.02),
    ("Lorenz84", 260, 0.02),
    ("HindmarshRose", 900, 0.05),
    ("NewtonLiepnik", 140, 0.01),
]


def _resample(y, n):
    """Resample a polyline to n points evenly spaced in arc length."""
    seg = np.linalg.norm(np.diff(y, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        raise ValueError("degenerate")
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, y[:, j]) for j in range(y.shape[1])], axis=1)


def _library_curve(sys, final_time, dt):
    """Integrate ``sys`` and return the 3-D streamline vertices via ``ts.viz``.

    The trajectory is integrated with the library, then its 3-D phase portrait is
    lowered through the ``threejs`` data-export renderer — so the curve vertices
    are the library's own plotting geometry (``positions``), not a hand-rolled
    slice. Retries with fresh random ICs on divergence.
    """
    last_err: Exception | None = None
    for _ in range(RETRIES):
        try:
            traj = sys.integrate(final_time=float(final_time), dt=float(dt))
        except Exception as e:  # noqa: BLE001 — a diverged run; retry from a new IC
            last_err = e
            continue
        # Library plotting path: 3-D phase portrait → threejs BufferGeometry payload.
        spec = traj.to_plot_spec()  # 3 components auto-dispatch → phase_portrait_3d
        if spec.kind.value != "phase_portrait_3d":
            raise ValueError(f"expected a 3-D portrait, got {spec.kind.value}")
        payload = spec.render("threejs", raw=True)
        lines = [g for g in payload["geometries"] if g["type"] == "line"]
        if not lines:
            last_err = ValueError("no line geometry in payload")
            continue
        y = np.asarray(lines[0]["positions"], dtype=float).reshape(-1, 3)
        y = y[np.all(np.isfinite(y), axis=1)]
        if len(y) >= 400:
            return y
        last_err = ValueError("too short")
    raise last_err or ValueError("no usable run")


def _prep(name, final_time, dt):
    sys = ts.systems.__dict__[name]()
    y = _library_curve(sys, final_time, dt)
    y = y[int(0.25 * len(y)):]  # drop transient
    if len(y) < 200:
        raise ValueError("too short")
    y = _resample(y, N)
    c = y.mean(0)
    d = np.linalg.norm(y - c, axis=1)
    r = np.quantile(d, 0.985) * 1.04  # robust radius (matches JS view-fit)
    y = (y - c) / r  # center + normalize → ~unit
    return np.round(y, 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = []
    for name, T, dt in ROSTER:
        try:
            y = _prep(name, T, dt)
        except Exception as e:  # noqa: BLE001 — skip a stubborn system, keep the rest
            print(f"  skip {name}: {e}")
            continue
        (OUT / f"{name}.json").write_text(
            json.dumps({"name": name, "n": len(y), "xyz": y.ravel().tolist()},
                       separators=(",", ":"))
        )
        ok.append(name)
        print(f"  ok   {name:22} {len(y)} pts  {(OUT/f'{name}.json').stat().st_size//1024} KB")
    (OUT / "manifest.json").write_text(json.dumps({"systems": ok}, separators=(",", ":")))
    print(f"\nwrote {len(ok)} systems to {OUT}")


if __name__ == "__main__":
    main()
