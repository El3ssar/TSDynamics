"""
Generate the landing-hero attractor data: one compact JSON per 3D system.

Each file holds an arc-length-resampled, radius-normalized streamline (flat
xyz array) so the hero canvas can rotate it in 3D at a uniform visual speed
and auto-fit it without per-system tuning. The JSONs are committed static
assets; re-run after changing the roster.

    uv run python docs/_tooling/make_hero.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

import tsdynamics as ts

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "hero"
N = 2400  # points stored per system

# Iconic, visually distinct 3D attractors. (system, final_time, dt)
ROSTER = [
    ("Lorenz", 70, 0.01), ("Rossler", 240, 0.02), ("Thomas", 320, 0.04),
    ("Halvorsen", 90, 0.01), ("Aizawa", 90, 0.01), ("Dadras", 70, 0.01),
    ("Chen", 60, 0.005), ("RabinovichFabrikant", 140, 0.01), ("Rucklidge", 130, 0.01),
    ("NoseHoover", 170, 0.02), ("Arneodo", 130, 0.01), ("Sakarya", 70, 0.005),
    ("BurkeShaw", 60, 0.005), ("Lorenz84", 260, 0.02), ("DequanLi", 45, 0.003),
    ("ChenLee", 70, 0.005), ("ShimizuMorioka", 170, 0.02), ("SprottB", 130, 0.01),
    ("NewtonLiepnik", 140, 0.01), ("HindmarshRose", 900, 0.05), ("MooreSpiegel", 130, 0.01),
    ("WangSun", 150, 0.01), ("Finance", 150, 0.02), ("Hadley", 150, 0.01),
]


def _resample(y, n):
    """Resample a polyline to n points evenly spaced in arc length."""
    seg = np.linalg.norm(np.diff(y, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        raise ValueError("degenerate")
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, y[:, j]) for j in range(y.shape[1])], axis=1)


def _prep(name, final_time, dt):
    sys = ts.systems.__dict__[name]()
    traj = sys.integrate(final_time=float(final_time), dt=float(dt), max_retries=25)
    y = traj.y[:, :3]
    y = y[int(0.25 * len(y)):]                 # drop transient
    y = y[np.all(np.isfinite(y), axis=1)]
    if len(y) < 200:
        raise ValueError("too short")
    y = _resample(y, N)
    c = y.mean(0)
    d = np.linalg.norm(y - c, axis=1)
    r = np.quantile(d, 0.985) * 1.04           # robust radius (matches JS view-fit)
    y = (y - c) / r                            # center + normalize → ~unit
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
