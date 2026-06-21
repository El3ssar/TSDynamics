"""
Golden-image corpus + perceptual-hash drift checker for the docs figures.

This package freezes a small, representative subset of the auto-generated
per-system documentation figures (see :mod:`docs._tooling.figures`) as a
committed *golden corpus* — one PNG per system plus a :file:`manifest.json`
recording each figure's **perceptual hash** (a 16x16 average hash, "aHash").

Why a perceptual hash rather than a byte compare?  A figure is a Matplotlib
render of a numerically integrated trajectory: a different Matplotlib/FreeType
version, a different platform, or a roundoff-level change in the engine all
perturb the exact PNG bytes while leaving the *picture* — the attractor's shape
— visually identical.  Byte equality is far too brittle for that; the aHash is
robust to those nuisance perturbations yet still catches a genuine structural
drift (a system that started integrating to a different attractor, a renderer
that dropped an axis, a regime change in the dynamics).

This corpus is the **renderer oracle** for the downstream DOCS-ENG streams:

- **DOCS-ENG-ENGINEFIG** re-renders the ODE figures through the shipped engine
  (instead of the SciPy ``_rhs_numeric`` path used today) and must re-baseline
  this corpus, proving the engine render is *structurally* the same picture.
- **DOCS-ENG-PLOTSEAM** introduces a ``to_plot_spec()``-driven figure generator
  whose output is checked against this same golden set.

Public surface
--------------
- :data:`GOLDEN_SUBSET` — the curated system names (every render branch covered).
- :func:`average_hash` — the perceptual hash of a PNG byte string.
- :func:`hamming_distance` — bit distance between two hex hashes.
- :func:`render_png` — render one registry entry to PNG bytes (deterministic).
- :func:`load_manifest` — read the committed :file:`manifest.json`.
- :func:`regenerate` — (re)write the golden PNGs + manifest (the baseline CLI).

Run ``python docs/_tooling/golden/__main__.py`` (with the ``docs`` dependency
group, which provides Matplotlib) to regenerate the baseline after an
intentional change.
"""

from __future__ import annotations

import io
import json
import pathlib

import numpy as np

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent
FIGURES_DIR = GOLDEN_DIR / "figures"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"

#: The curated golden subset — one system per render branch in
#: :mod:`docs._tooling.figures`, chosen for being canonical and stable:
#:
#: - ``Lorenz``      — 3-D ODE phase portrait
#: - ``Rossler``     — 3-D ODE phase portrait (a second, differently shaped one)
#: - ``Lorenz96``    — space-time ``imshow`` branch (``kind="spacetime"``)
#: - ``Henon``       — 2-D map scatter
#: - ``Logistic``    — 1-D map return-map scatter
#: - ``MackeyGlass`` — DDE time series + delay-embedding branch
GOLDEN_SUBSET: tuple[str, ...] = (
    "Lorenz",
    "Rossler",
    "Lorenz96",
    "Henon",
    "Logistic",
    "MackeyGlass",
)

#: aHash side length: the figure is downscaled to ``HASH_SIDE x HASH_SIDE``
#: grayscale before thresholding, giving a ``HASH_SIDE**2``-bit hash.
HASH_SIDE = 16

#: Default acceptance tolerance for the drift check, in fraction of bits that
#: may differ between the freshly rendered figure and its golden hash.  ~6% of
#: 256 bits ≈ 16 bits — comfortably above the platform/Matplotlib-version
#: jitter floor, well below a real structural change (which flips many more).
DEFAULT_TOLERANCE = 0.06


def _to_grayscale_array(png_bytes: bytes) -> np.ndarray:
    """Decode a PNG byte string to a 2-D float grayscale array (no Pillow).

    Matplotlib already ships with the project's ``docs``/``plot`` extra, so we
    decode through ``matplotlib.image`` rather than adding a Pillow dependency.
    Transparent pixels (the figures save with a transparent background) are
    composited onto white so the perceptual hash keys off the drawn ink, not
    the alpha channel.
    """
    import matplotlib.image as mpimg

    rgba = mpimg.imread(io.BytesIO(png_bytes), format="png")
    # mpimg returns float in [0, 1] for PNG. Shape (H, W, 4) (RGBA) or (H, W, 3).
    rgba = np.asarray(rgba, dtype=np.float64)
    if rgba.ndim == 2:  # already grayscale
        return rgba
    rgb = rgba[..., :3]
    if rgba.shape[-1] == 4:  # composite over white using the alpha channel
        alpha = rgba[..., 3:4]
        rgb = rgb * alpha + (1.0 - alpha)
    # Rec. 601 luma
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _block_downscale(gray: np.ndarray, side: int) -> np.ndarray:
    """Average-pool a 2-D array down to ``side x side`` (dependency-free)."""
    h, w = gray.shape
    rows = np.linspace(0, h, side + 1).astype(int)
    cols = np.linspace(0, w, side + 1).astype(int)
    out = np.empty((side, side), dtype=np.float64)
    for i in range(side):
        r0, r1 = rows[i], max(rows[i] + 1, rows[i + 1])
        for j in range(side):
            c0, c1 = cols[j], max(cols[j] + 1, cols[j + 1])
            out[i, j] = float(gray[r0:r1, c0:c1].mean())
    return out


def average_hash(png_bytes: bytes, side: int = HASH_SIDE) -> str:
    """Perceptual average hash (aHash) of a PNG, as a lowercase hex string.

    The image is decoded to grayscale, average-pooled to ``side x side``, and
    each cell is set to 1 where it is brighter than the overall mean.  The
    resulting ``side**2``-bit field is returned big-endian as hex.
    """
    gray = _to_grayscale_array(png_bytes)
    small = _block_downscale(gray, side)
    bits = (small > small.mean()).flatten()
    n = bits.size
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return format(value, "x").zfill((n + 3) // 4)


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """Count the differing bits between two equal-length hex hash strings."""
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")


def _entry(name: str):
    from tsdynamics import registry

    for entry in registry.all_systems():
        if entry.name == name:
            return entry
    raise KeyError(f"system {name!r} not found in the registry")


def render_png(name: str) -> bytes:
    """Render the golden figure for ``name`` to PNG bytes, deterministically.

    Re-uses the production figure renderer (:mod:`docs._tooling.figures`) so
    the golden corpus tracks exactly what the docs build emits, then serialises
    the figure to an in-memory PNG with the same ``dpi``/``bbox_inches`` the
    docs build uses.  The renderer seeds its own RNG, so the output is
    reproducible run to run.
    """
    import sys

    sys.path.insert(0, str(GOLDEN_DIR.parent))  # docs/_tooling on the path
    import figures as _figures  # type: ignore[import-not-found]

    entry = _entry(name)
    opts = _figures.FIG_OVERRIDES.get(entry.name, {})
    plt = _figures._style()
    if entry.family == "ode":
        fig = _figures._render_ode(entry, plt, opts)
    elif entry.family == "dde":
        fig = _figures._render_dde(entry, plt, opts)
    else:
        fig = _figures._render_map(entry, plt, opts)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def load_manifest() -> dict:
    """Load the committed golden manifest (``manifest.json``)."""
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def drift_report(names: tuple[str, ...] = GOLDEN_SUBSET) -> dict[str, dict[str, object]]:
    """Re-render ``names`` and report each one's perceptual drift from golden.

    Returns ``{name: {"distance": int, "n_bits": int, "budget": int,
    "fresh": hex, "golden": hex, "ok": bool}}``.  This is the in-process core of
    the drift check; the test driver runs it in a subprocess so that importing
    Matplotlib for the render never leaks a plot backend into the parent
    interpreter (the no-plot-backend-on-import contract the viz tests guard).
    """
    manifest = load_manifest()
    n_bits = HASH_SIDE**2
    budget = int(DEFAULT_TOLERANCE * n_bits)
    report: dict[str, dict[str, object]] = {}
    for name in names:
        gold = str(manifest["figures"][name]["ahash"])
        fresh = average_hash(render_png(name))
        dist = hamming_distance(fresh, gold)
        report[name] = {
            "distance": dist,
            "n_bits": n_bits,
            "budget": budget,
            "fresh": fresh,
            "golden": gold,
            "ok": dist <= budget,
        }
    return report


def regenerate() -> dict:
    """(Re)render the golden subset, write the PNGs + manifest, and return it.

    This is the baseline command — run it after an *intentional* renderer or
    dynamics change, review the resulting PNG diffs by eye, and commit them.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures: dict[str, dict[str, object]] = {}
    for name in GOLDEN_SUBSET:
        entry = _entry(name)
        png = render_png(name)
        (FIGURES_DIR / f"{name}.png").write_bytes(png)
        figures[name] = {
            "family": entry.family,
            "ahash": average_hash(png),
        }
    manifest = {
        "hash_side": HASH_SIDE,
        "default_tolerance": DEFAULT_TOLERANCE,
        "figures": figures,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
