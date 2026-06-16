"""
Generate the TSDynamics logo mark + favicon from the Aizawa attractor.

The mark is a single decimated streamline of the Aizawa attractor (the
indigo→teal house gradient applied spatially), viewed three-quarters so the
strange-attractor depth reads even at small sizes. Re-run after changing the
viewpoint/params; the SVGs are committed static assets.

    uv run python docs/_tooling/make_logo.py
"""

from __future__ import annotations

import pathlib

import numpy as np

import tsdynamics as ts

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets"


def _rot(y, yaw, pitch):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return y @ ry.T @ rx.T


def _streamline(final_time, yaw, pitch, n_keep):
    sys = ts.systems.Aizawa()
    y = sys.integrate(final_time=final_time, dt=0.01).y
    y = y - y.mean(0)
    r = _rot(y, yaw, pitch)[:, :2]
    step = max(1, len(r) // n_keep)
    r = r[::step]
    # normalize into a 64-box with margin
    lo, hi = r.min(0), r.max(0)
    span = (hi - lo).max()
    r = (r - (lo + hi) / 2) / span  # center, unit-ish
    r[:, 1] = -r[:, 1]  # SVG y is down
    return r * 52 + 32  # → ~[6, 58] inside a 64 viewBox


def _path_d(r):
    d = [f"M{r[0, 0]:.2f} {r[0, 1]:.2f}"]
    d += [f"L{x:.2f} {y:.2f}" for x, y in r[1:]]
    return "".join(d)


def _svg(r, stroke_w):
    d = _path_d(r)
    return f"""<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="TSDynamics">
  <defs>
    <linearGradient id="tsg" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0" stop-color="#4f46e5"/>
      <stop offset="0.55" stop-color="#4338ca"/>
      <stop offset="1" stop-color="#0d9488"/>
    </linearGradient>
  </defs>
  <path d="{d}" fill="none" stroke="url(#tsg)" stroke-width="{stroke_w}"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Hero / nav mark — an open orb: a few interleaving Aizawa loops that keep
    # the strange-attractor structure legible while staying clean at small size.
    mark = _streamline(final_time=14.0, yaw=0.6, pitch=-0.32, n_keep=380)
    (OUT / "logo.svg").write_text(_svg(mark, 2.0))
    # Favicon — fewer loops still so it survives 16–32px.
    fav = _streamline(final_time=10.0, yaw=0.6, pitch=-0.32, n_keep=280)
    (OUT / "favicon.svg").write_text(_svg(fav, 2.8))
    # Fidelity preview (exact committed geometry).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    fig.patch.set_facecolor("#0b0f1a")
    for ax, r, lw, name in [(axes[0], mark, 4, "logo.svg"), (axes[1], fav, 6, "favicon.svg")]:
        pts = r.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], 1)
        t = np.linspace(0, 1, len(segs))
        c0 = np.array([0.31, 0.275, 0.898]); c1 = np.array([0.05, 0.58, 0.53])
        cols = (1 - t)[:, None] * c0 + t[:, None] * c1
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, capstyle="round"))
        ax.set_xlim(0, 64); ax.set_ylim(64, 0); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(name, color="#aab2c8")
    plt.tight_layout(); plt.savefig("/tmp/ts_logo_preview.png", dpi=110, facecolor="#0b0f1a")
    print("wrote", OUT / "logo.svg", "+", OUT / "favicon.svg", "(", len((OUT/"logo.svg").read_text()), "bytes )")


if __name__ == "__main__":
    main()
