"""
Generate the landing-hero attractor data: one compact JSON per 3-D system.

The hero attractors are the **same curves the system pages animate**.  Each
system's streamline is built by the interactive three.js viewer's own curve
builder (:func:`threejs_viewer._ode_cloud`) — which reads that system's editorial
``viewer`` block (``final_time`` / ``method`` / ``ic`` / ``transient`` /
``projection``) and arc-length-resamples to the sagitta target, exactly the
geometry drawn on the system page.  Here we merely choose *which* systems appear
in the hero and downsample + radius-normalize each curve so the landing canvas
can rotate a dozen of them at a uniform visual speed on a light page.

So there is **one source of truth** for how each system is integrated — its
editorial entry — shared by the static figure, the page viewer, and this hero.
To retune a hero attractor, edit its ``viewer`` block in ``editorial.json`` (a
3-D viewer ignores ``dt``: it integrates fine and resamples in arc length, so
tune ``final_time`` / ``ic`` / ``method`` / ``transient`` instead).

The JSONs are committed static assets; re-run after changing the roster or a
system's editorial ``viewer`` block:

    .venv/bin/python docs/_tooling/make_hero.py
"""

from __future__ import annotations

import json
import pathlib

import catalog as _catalog
import numpy as np
import threejs_viewer as _viewer  # docs/_tooling sibling — the page viewer's curve builder

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "hero"
N = 2400  # points stored per system — a compact downsample of the page curve

# A curated set of 14 visually distinct 3-D attractors — butterfly, spiral-fold,
# spherical shell, cyclic labyrinth, multi-lobe, spiky, bursting, climate.  This is
# NAMES ONLY: *how* each is integrated lives in its editorial ``viewer`` block (the
# single source of truth shared with the system page), never here.
HERO_SYSTEMS = [
    "Lorenz",
    "Rossler",
    "Aizawa",
    "Thomas",
    "Halvorsen",
    "Dadras",
    "Chen",
    "RabinovichFabrikant",
    "NoseHoover",
    "Arneodo",
    "ShimizuMorioka",
    "Lorenz84",
    "HindmarshRose",
    "NewtonLiepnik",
]


def _resample(y, n):
    """Resample a polyline to n points evenly spaced in arc length."""
    seg = np.linalg.norm(np.diff(y, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        raise ValueError("degenerate")
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, y[:, j]) for j in range(y.shape[1])], axis=1)


def _prep(entry):
    """Build one hero curve: the page viewer's own streamline, downsampled + normalized.

    The curve comes straight from :func:`threejs_viewer._ode_cloud` — the exact
    geometry the system page animates (fine integration + arc-length resample,
    honouring the editorial ``viewer`` block and any ``projection``).  We only
    downsample it to :data:`N` points and centre + radius-normalize to ~unit so the
    landing canvas can auto-fit and rotate it without per-system tuning.
    """
    curve = _viewer._ode_cloud(entry, second=False)
    if curve is None:
        raise ValueError("viewer returned no curve (divergent / off-basin)")
    y = np.asarray(curve, dtype=float)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"expected a 3-D curve, got shape {y.shape}")
    y = y[np.all(np.isfinite(y), axis=1)]
    if len(y) < 200:
        raise ValueError("too short")
    y = _resample(y, N)  # compact for the light landing page
    c = y.mean(0)
    d = np.linalg.norm(y - c, axis=1)
    r = np.quantile(d, 0.985) * 1.04  # robust radius (matches JS view-fit)
    y = (y - c) / r  # center + normalize → ~unit
    return np.round(y, 3)


def main():
    """Regenerate every hero JSON (+ manifest) from the roster; prune orphans."""
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = _catalog.load_catalog()
    ok = []
    for name in HERO_SYSTEMS:
        entry = catalog.by_name(name)
        if entry is None:
            print(f"  skip {name}: not in catalogue")
            continue
        try:
            y = _prep(entry)
        except Exception as e:  # noqa: BLE001 — skip a stubborn system, keep the rest
            print(f"  skip {name}: {e}")
            continue
        (OUT / f"{name}.json").write_text(
            json.dumps(
                {"name": name, "n": len(y), "xyz": y.ravel().tolist()},
                separators=(",", ":"),
            )
        )
        ok.append(name)
        print(f"  ok   {name:22} {len(y)} pts  {(OUT / f'{name}.json').stat().st_size // 1024} KB")
    (OUT / "manifest.json").write_text(json.dumps({"systems": ok}, separators=(",", ":")))
    # Prune orphaned hero JSONs (systems dropped from the roster) so the assets dir
    # only ever holds the current hero set.
    for stale in OUT.glob("*.json"):
        if stale.stem != "manifest" and stale.stem not in ok:
            stale.unlink()
            print(f"  rm   {stale.stem} (orphan)")
    print(f"\nwrote {len(ok)} systems to {OUT}")


if __name__ == "__main__":
    main()
