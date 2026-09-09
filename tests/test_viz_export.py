"""Tests for the viz serialization + resampling leaves (stream VIZ-WEB-EXPORT).

Two modules, both dependency-light and both load-bearing for the web-export story:

- :mod:`tsdynamics.viz.export` — the versioned ``PlotSpec`` JSON envelope.  It is
  what makes a spec *portable*: computed once, cached / shipped / replayed without
  re-running the analysis and without a plotting library installed.
- :mod:`tsdynamics.viz._resample` — the arc-length primitives the threejs vertex
  cap is built on.  These are asserted here as *mathematics* (does an arc-length
  resample actually beat a stride on a fast curve?); their use inside the exporter
  is asserted in ``tests/test_viz_threejs.py``.

Engine-free — no ``tsdynamics._rust`` import.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tsdynamics.viz.export import (
    SCHEMA_VERSION,
    from_dict_envelope,
    from_json,
    to_dict_envelope,
    to_json,
)
from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec


def _spec(n: int = 32) -> PlotSpec:
    """A small 3-D spec with array data, labels and meta to round-trip."""
    t = np.linspace(0.0, 4.0, n)
    return PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        ndim=3,
        title="orbit",
        layers=[
            Layer(
                kind=PlotKind.LINE3D,
                data={"x": np.sin(t), "y": np.cos(t), "z": t},
                label="trajectory",
                style={"color": "#2cc5ae", "linewidth": 0.8},
            )
        ],
        x=Axis(label="x"),
        y=Axis(label="y"),
        z=Axis(label="z"),
        meta={"dt": 0.01},
    )


# ---------------------------------------------------------------------------
# the versioned envelope
# ---------------------------------------------------------------------------


def test_envelope_round_trips_a_spec() -> None:
    """``from_json(to_json(spec))`` reproduces the spec, arrays included."""
    spec = _spec()
    back = from_json(to_json(spec))
    assert back.kind is spec.kind
    assert back.title == spec.title
    assert back.layers[0].label == "trajectory"
    assert back.layers[0].style["color"] == "#2cc5ae"
    np.testing.assert_allclose(back.layers[0].data["x"], spec.layers[0].data["x"])


def test_envelope_stamps_the_current_schema_version() -> None:
    """The envelope carries the version so a future layout change can migrate."""
    envelope = to_dict_envelope(_spec())
    assert envelope["schema_version"] == SCHEMA_VERSION
    assert "spec" in envelope


def test_a_bare_legacy_spec_dict_still_loads() -> None:
    """An unversioned ``PlotSpec.to_dict()`` document (pre-envelope) loads."""
    bare = _spec().to_dict()
    assert "schema_version" not in bare
    assert from_dict_envelope(bare).kind is PlotKind.PHASE_PORTRAIT_3D
    assert from_json(json.dumps(bare)).title == "orbit"


def test_an_older_schema_version_still_loads() -> None:
    """A v1 / v2 payload loads: new fields default rather than raising.

    The envelope's whole purpose is that an artifact saved by an older TSDynamics
    keeps opening.  A version bump that broke old payloads would defeat it.
    """
    envelope = to_dict_envelope(_spec())
    for version in (1, 2):
        older = {**envelope, "schema_version": version}
        assert from_dict_envelope(older).kind is PlotKind.PHASE_PORTRAIT_3D


def test_a_non_spec_payload_raises_with_a_useful_message() -> None:
    """Garbage in gets a named error, not a confusing KeyError deeper down."""
    with pytest.raises(ValueError, match="not a PlotSpec JSON payload"):
        from_json("[1, 2, 3]")
    with pytest.raises(ValueError, match="not a PlotSpec JSON payload"):
        from_dict_envelope({"nothing": "useful"})


# ---------------------------------------------------------------------------
# the arc-length resampling primitives
# ---------------------------------------------------------------------------


def _spiral(n: int, *, turns: float = 100.0, power: float = 4.0) -> np.ndarray:
    """A helix traversed at wildly varying speed — a uniform stride's worst case.

    The angle advances as ``s**power``, so consecutive *time* samples near the end
    of the curve are far apart *in space* while those at the start are bunched.  A
    stride therefore over-samples the slow head and chords straight across the fast
    tail; an arc-length resample equalises them.  This is the synthetic analogue of
    what a fast catalogue attractor (HyperQi) does to a stride.
    """
    s = np.linspace(0.0, 1.0, n)
    theta = 2 * np.pi * turns * s**power
    r = 3.0
    return np.stack([r * np.cos(theta), r * np.sin(theta), 5.0 * s**power], axis=1)


def test_resample_hits_the_requested_vertex_count_and_keeps_the_endpoints() -> None:
    """Resampling thins to exactly ``n`` and pins both ends of the curve."""
    from tsdynamics.viz._resample import resample_arclength

    y = _spiral(5000)
    out, _ = resample_arclength(y, 500)
    assert out.shape == (500, 3)
    np.testing.assert_allclose(out[0], y[0], atol=1e-9)
    np.testing.assert_allclose(out[-1], y[-1], atol=1e-9)


def test_resample_never_grows_a_curve() -> None:
    """Asking for more vertices than the input has returns the input untouched.

    The exporter must never invent vertices it was not given: a cap is a ceiling,
    not an interpolator.
    """
    from tsdynamics.viz._resample import resample_arclength

    y = _spiral(100)
    out, _ = resample_arclength(y, 5000)
    assert out is y or out.shape == y.shape


def test_resample_makes_the_segments_equal_length() -> None:
    """After the resample every drawn chord has (near) the same spatial length.

    That is the definition of "uniform in space", and the reason the cap does not
    starve the fast turns: resolution follows *length*, not time.
    """
    from tsdynamics.viz._resample import resample_arclength

    out, _ = resample_arclength(_spiral(20_000), 2000)
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert seg.std() / seg.mean() < 0.05


def test_arclength_beats_a_stride_on_a_speed_varying_curve() -> None:
    """The measurement the whole design rests on, as an executable assertion.

    A uniform-in-time stride spends its budget where the curve is *slow*; the fast,
    tightly-curved tail stays a chorded polygon.  Arc-length sampling gives every
    turn resolution proportional to its length.
    """
    from tsdynamics.viz._resample import max_sagitta_ratio, resample_arclength

    y = _spiral(40_000)
    cap = 2000
    stride = y[:: max(1, -(-len(y) // cap))][:cap]
    arc, _ = resample_arclength(y, cap)
    assert max_sagitta_ratio(arc) < 0.25 * max_sagitta_ratio(stride)


def test_channels_ride_the_same_parameterisation_as_the_positions() -> None:
    """A per-vertex channel is resampled *with* the geometry, never separately."""
    from tsdynamics.viz._resample import resample_arclength

    y = _spiral(10_000)
    c = np.linspace(0.0, 1.0, len(y))
    out, chans = resample_arclength(y, 800, channels={"c": c})
    assert out.shape[0] == chans["c"].shape[0] == 800
    assert chans["c"][0] == pytest.approx(0.0)
    assert chans["c"][-1] == pytest.approx(1.0)
    assert np.all(np.diff(chans["c"]) >= -1e-12)


def test_sagitta_is_scale_free() -> None:
    """The criterion divides by the bounding-box diagonal, so units cannot matter."""
    from tsdynamics.viz._resample import max_sagitta_ratio

    y = _spiral(3000)
    assert max_sagitta_ratio(y) == pytest.approx(max_sagitta_ratio(1000.0 * y), rel=1e-9)


def test_sagitta_handles_2d_curves_and_degenerate_input() -> None:
    """2-D delay embeddings use the same criterion; degenerate input is 0, not NaN."""
    from tsdynamics.viz._resample import max_sagitta_ratio

    t = np.linspace(0, 2 * np.pi, 1000)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    assert 0.0 < max_sagitta_ratio(circle) < 0.01
    assert max_sagitta_ratio(np.zeros((2, 3))) == 0.0
    assert max_sagitta_ratio(np.zeros((50, 3))) == 0.0


def test_smooth_arclength_stops_at_the_floor_for_a_gentle_curve() -> None:
    """A gently curving path meets the target at the floor and stays light."""
    from tsdynamics.viz._resample import max_sagitta_ratio, smooth_arclength

    t = np.linspace(0, 2 * np.pi, 50_000)
    gentle = np.stack([np.cos(t), np.sin(t), 0.1 * t], axis=1)
    out = smooth_arclength(gentle, target=0.008, nmin=500, nmax=40_000)
    assert len(out) == 500
    assert max_sagitta_ratio(out) < 0.008


def test_smooth_arclength_spends_up_to_the_ceiling_for_a_sharp_one() -> None:
    """A fast, tightly-curved path spends more vertices — but never past ``nmax``."""
    from tsdynamics.viz._resample import smooth_arclength

    out = smooth_arclength(_spiral(60_000, turns=400.0), target=1e-6, nmin=200, nmax=4000)
    assert len(out) == 4000


def test_uniform_subsample_is_deterministic_and_order_preserving() -> None:
    """A point cloud thins reproducibly and keeps iteration order.

    Reproducible so an export is byte-stable across processes; ordered so a
    points-comet still sweeps the cloud in the order the map produced it.
    """
    from tsdynamics.viz._resample import uniform_subsample_indices

    a = uniform_subsample_indices(10_000, 500)
    b = uniform_subsample_indices(10_000, 500)
    assert np.array_equal(a, b)
    assert len(a) == 500
    assert np.all(np.diff(a) > 0)
    # No thinning needed ⇒ exactly the identity.
    assert np.array_equal(uniform_subsample_indices(100, 500), np.arange(100))


def test_uniform_subsample_is_not_a_stride() -> None:
    """A stride would alias a periodic cloud; the seeded draw does not.

    Concretely: a stride keeps a perfectly regular index spacing, so a cloud whose
    structure is commensurate with that spacing collapses.  The draw's gaps vary.
    """
    from tsdynamics.viz._resample import uniform_subsample_indices

    idx = uniform_subsample_indices(10_000, 1000)
    assert float(np.std(np.diff(idx))) > 0.5
