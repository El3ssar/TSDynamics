"""Drift checker for the frozen golden docs-figure corpus (DOCS-ENG-GOLDEN).

The committed corpus lives in :file:`docs/_tooling/golden/` — one PNG per
system in ``GOLDEN_SUBSET`` plus a :file:`manifest.json` of their perceptual
hashes.  These tests re-render that subset through the production figure
renderer and assert each fresh render is *structurally* the same picture as the
golden baseline, measured by Hamming distance between 16x16 average hashes
(aHash).

A perceptual hash is deliberately tolerant of nuisance perturbations (a
Matplotlib/FreeType version bump, a platform font, a roundoff-level engine
difference) while still catching a genuine structural drift — a system that
started integrating to a different attractor, or a renderer that dropped an axis
or changed a plot kind.  This is the renderer oracle the downstream
DOCS-ENG-ENGINEFIG / DOCS-ENG-PLOTSEAM streams re-baseline against.

The whole module is marked ``slow`` because each figure integrates a trajectory
and rasterises it; it also needs Matplotlib (the ``docs``/``plot`` extra), so it
skips cleanly where Matplotlib is absent.

Note on isolation: any work that *renders* (and therefore imports Matplotlib)
runs in a **subprocess**.  A sibling guard (``test_renderers_registry``) asserts
that importing :mod:`tsdynamics` pulls in no plot backend by inspecting the
shared ``sys.modules``; doing the Matplotlib render in-process here would leak
the backend into that global and trip the guard.  The pure-data tests (manifest
shape, committed PNGs, the bit metric) need no Matplotlib and stay in-process.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

# Decide via ``find_spec`` rather than ``importorskip`` so that *collecting* this
# module does not import Matplotlib into ``sys.modules`` (see the module note).
if importlib.util.find_spec("matplotlib") is None or importlib.util.find_spec("scipy") is None:
    pytest.skip("matplotlib/scipy (the docs/plot extra) not installed", allow_module_level=True)

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "_tooling" / "golden"


def _load_golden():
    """Import the ``docs/_tooling/golden`` package by file path.

    ``docs/`` is not an importable package (it carries no ``__init__``), so we
    load the golden module directly rather than through a dotted import.  This
    import alone touches no Matplotlib (the renderer defers that), so it is safe
    in-process.
    """
    init = _GOLDEN_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "tsd_docs_golden",
        init,
        submodule_search_locations=[str(_GOLDEN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden = _load_golden()


def _drift_report_via_subprocess() -> dict:
    """Run ``golden.drift_report()`` in a child process and return its JSON.

    Rendering imports Matplotlib; doing it in a subprocess keeps the parent
    interpreter's ``sys.modules`` free of a plot backend (the contract the viz
    tests guard) and asserts no backend leaked even in the child.
    """
    code = (
        "import sys, json, importlib.util;"
        f"d = {str(_GOLDEN_DIR)!r};"
        "spec = importlib.util.spec_from_file_location("
        "'tsd_docs_golden', d + '/__init__.py', submodule_search_locations=[d]);"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "rep = m.drift_report();"
        "print('REPORT_JSON', json.dumps(rep))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(_GOLDEN_DIR.parents[2]),
    )
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("REPORT_JSON"))
    return json.loads(line[len("REPORT_JSON ") :])


def test_manifest_present_and_covers_subset():
    """The committed manifest exists and records every golden subset system."""
    manifest = golden.load_manifest()
    assert manifest["hash_side"] == golden.HASH_SIDE
    figures = manifest["figures"]
    assert set(figures) == set(golden.GOLDEN_SUBSET)
    for info in figures.values():
        # Each hash is the right width for a HASH_SIDE**2-bit aHash.
        assert len(info["ahash"]) == (golden.HASH_SIDE**2 + 3) // 4
        assert info["family"] in {"ode", "dde", "map"}


def test_golden_pngs_committed():
    """A golden PNG is committed for every subset system."""
    for name in golden.GOLDEN_SUBSET:
        png = _GOLDEN_DIR / "figures" / f"{name}.png"
        assert png.exists(), f"missing golden figure: {png}"
        assert png.stat().st_size > 0


def test_hamming_distance_metric():
    """The bit metric is reflexive and detects a flip (in-process, no render)."""
    gold = golden.load_manifest()["figures"][golden.GOLDEN_SUBSET[0]]["ahash"]
    assert golden.hamming_distance(gold, gold) == 0
    flipped = format(int(gold, 16) ^ 0b1111, "x").zfill(len(gold))
    assert golden.hamming_distance(gold, flipped) > 0


def test_renders_match_golden_within_tolerance():
    """Every freshly rendered figure is structurally equivalent to its golden.

    The acceptance bar is a Hamming distance below ``DEFAULT_TOLERANCE`` of the
    total bit budget — robust to platform/version jitter, strict enough to catch
    a real structural change (a different attractor, a dropped axis, a regime
    change).  The render runs in a subprocess (see the module note).
    """
    report = _drift_report_via_subprocess()
    assert set(report) == set(golden.GOLDEN_SUBSET)
    drifted = {name: r for name, r in report.items() if not r["ok"]}
    assert not drifted, (
        "golden docs figures drifted: "
        + ", ".join(
            f"{name} {r['distance']}/{r['n_bits']} bits (budget {r['budget']})"
            for name, r in drifted.items()
        )
        + "; regenerate via `python docs/_tooling/golden/__main__.py` if intentional"
    )
