"""Unit tests for the canonical :class:`ScalingResult` schema (WS-SCALING).

``ScalingResult`` is the *one* schema the whole scaling-curve family (every
fractal dimension, Lyapunov-from-data, expansion entropy, Cao / FNN) reparents
onto later — ``estimate``/``stderr``/``abscissa``/``ordinate``/``fit_region``/
``intercept`` — so a single generic ``.plot.scaling()`` renders any of them.
These tests pin that schema, ``float() == estimate``, the ``local_slopes`` /
``scaling_window`` diagnostics, the round-trip, and the deferred plot seam.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import numpy as np
import pytest

from tsdynamics.analysis._result import (
    AnalysisResult,
    ScalingResult,
    VisualizationNotInstalled,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeSystem:
    def _provenance(self, **extra):
        return {"system": "Lorenz", "params": {"sigma": 10.0}, "tsdynamics": "4.0", **extra}


def _curve() -> tuple[np.ndarray, np.ndarray]:
    # A clean log--log line of slope 2 with a kink outside the fit window, so
    # local_slopes has a recognisable plateau over the scaling region.
    x = np.linspace(0.0, 9.0, 10)
    y = 2.0 * x - 1.0
    y[0] = y[0] + 5.0  # corrupt the first point (it sits outside the fit region)
    return x, y


def _scaling() -> ScalingResult:
    x, y = _curve()
    return ScalingResult(
        estimate=2.0,
        stderr=0.05,
        abscissa=x,
        ordinate=y,
        fit_region=(1, 8),
        intercept=-1.0,
        meta=AnalysisResult.build_meta(_FakeSystem(), final_time=300.0),
    )


# ---------------------------------------------------------------------------
# Schema: the six canonical fields exist and carry the right values
# ---------------------------------------------------------------------------


def test_exposes_the_canonical_schema():
    r = _scaling()
    assert r.estimate == 2.0
    assert r.stderr == 0.05
    assert r.fit_region == (1, 8)
    assert r.intercept == -1.0
    np.testing.assert_array_equal(r.abscissa, _curve()[0])
    np.testing.assert_array_equal(r.ordinate, _curve()[1])


def test_is_an_analysis_result_subclass():
    assert issubclass(ScalingResult, AnalysisResult)
    assert isinstance(_scaling(), AnalysisResult)


def test_canonical_field_set_is_exactly_the_schema():
    # Guards the schema against accidental field drift (meta is inherited).
    names = {f.name for f in dataclasses.fields(ScalingResult)}
    assert names == {
        "estimate",
        "stderr",
        "abscissa",
        "ordinate",
        "fit_region",
        "intercept",
        "meta",
    }


# ---------------------------------------------------------------------------
# __float__ returns the estimate
# ---------------------------------------------------------------------------


def test_float_returns_estimate():
    assert float(_scaling()) == 2.0


def test_float_drops_into_arithmetic():
    import math

    r = _scaling()
    # __float__ is what lets the result stand in for its number wherever a float
    # is expected (np.isclose, math.sqrt, np.asarray of a scalar, ...).
    assert np.isclose(float(r), 2.0)
    assert math.sqrt(r) == pytest.approx(2.0**0.5)
    assert np.asarray(r, dtype=float) == 2.0


def test_float_coerces_numpy_scalar_estimate():
    r = ScalingResult(estimate=np.float64(1.886))
    assert float(r) == pytest.approx(1.886)
    assert isinstance(float(r), float)


# ---------------------------------------------------------------------------
# local_slopes
# ---------------------------------------------------------------------------


def test_local_slopes_plateau_over_scaling_region():
    r = _scaling()
    slopes = r.local_slopes
    assert slopes.shape == r.abscissa.shape
    # The curve is a clean slope-2 line except for the corrupted point 0, whose
    # effect reaches only index 1 (the centred difference is one point wide).
    # The interior of the fit region is therefore an exact slope-2 plateau.
    assert np.allclose(slopes[2:], 2.0)
    # And it correctly reads the corrupted endpoint as off the plateau.
    assert not np.isclose(slopes[1], 2.0)


def test_local_slopes_handles_too_short_curve():
    r = ScalingResult(abscissa=np.array([1.0]), ordinate=np.array([3.0]))
    slopes = r.local_slopes
    assert slopes.shape == (1,)
    assert np.all(np.isnan(slopes))


def test_local_slopes_empty_curve_is_empty():
    r = ScalingResult()  # default empty arrays
    assert r.local_slopes.shape == (0,)


# ---------------------------------------------------------------------------
# scaling_window
# ---------------------------------------------------------------------------


def test_scaling_window_is_abscissa_endpoints_of_fit_region():
    r = _scaling()
    lo, hi = r.fit_region
    assert r.scaling_window == (r.abscissa[lo], r.abscissa[hi])
    assert r.scaling_window == (1.0, 8.0)


def test_scaling_window_returns_plain_floats():
    lo, hi = _scaling().scaling_window
    assert isinstance(lo, float) and isinstance(hi, float)


# ---------------------------------------------------------------------------
# repr / summary inherit the AnalysisResult machinery
# ---------------------------------------------------------------------------


def test_repr_shows_scalar_summary_fields_not_arrays():
    text = repr(_scaling())
    assert text.startswith("ScalingResult(")
    assert "estimate=2" in text
    assert "stderr=0.05" in text
    assert "abscissa" not in text  # array field, repr=False
    assert "ordinate" not in text


def test_summary_carries_interpretation_line():
    out = _scaling().summary()
    assert out.splitlines()[0] == "ScalingResult  (Lorenz)"
    assert "→" in out
    assert "fit over 8 points" in out


# ---------------------------------------------------------------------------
# Round-trip: to_dict is JSON-serializable and recovers the schema
# ---------------------------------------------------------------------------


def test_to_dict_round_trips_the_schema():
    r = _scaling()
    d = r.to_dict()
    # Arrays became lists; scalars are plain.
    assert d["estimate"] == 2.0
    assert d["stderr"] == 0.05
    assert d["intercept"] == -1.0
    assert d["fit_region"] == [1, 8]  # tuple -> list
    assert d["abscissa"] == r.abscissa.tolist()
    assert d["ordinate"] == r.ordinate.tolist()
    json.dumps(d)  # must not raise


def test_round_trip_reconstructs_an_equal_result():
    r = _scaling()
    d = r.to_dict()
    # Rebuild from the canonical names; arrays come back from the lists.
    rebuilt = ScalingResult(
        estimate=d["estimate"],
        stderr=d["stderr"],
        abscissa=np.asarray(d["abscissa"]),
        ordinate=np.asarray(d["ordinate"]),
        fit_region=tuple(d["fit_region"]),
        intercept=d["intercept"],
    )
    assert float(rebuilt) == float(r)
    assert rebuilt.scaling_window == r.scaling_window
    np.testing.assert_array_equal(rebuilt.abscissa, r.abscissa)
    # Equality compares the scalar fields (arrays/meta are compare=False).
    assert rebuilt == r


def test_to_dict_includes_meta_provenance():
    d = _scaling().to_dict()
    assert d["meta"]["system"] == "Lorenz"
    assert d["meta"]["final_time"] == 300.0


# ---------------------------------------------------------------------------
# Frozen-dataclass gotcha: arrays are compare=False, instances hash/compare
# ---------------------------------------------------------------------------


def test_is_frozen():
    r = _scaling()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.estimate = 9.0  # type: ignore[misc]


def test_equality_ignores_arrays_and_meta():
    x1, y1 = _curve()
    a = ScalingResult(estimate=2.0, abscissa=x1, ordinate=y1, meta={"run": 1})
    b = ScalingResult(
        estimate=2.0,
        abscissa=np.array([9.0, 9.0]),  # different curve
        ordinate=np.array([0.0, 1.0]),
        meta={"run": 2},  # different provenance
    )
    assert a == b  # would raise ValueError if arrays were in __eq__
    assert hash(a) == hash(b)  # would raise TypeError if arrays were hashed
    assert {a, b} == {a}


def test_different_estimate_compares_unequal():
    assert ScalingResult(estimate=2.0) != ScalingResult(estimate=2.1)


# ---------------------------------------------------------------------------
# The .plot.scaling() seam — raises until a backend registers
# ---------------------------------------------------------------------------


def test_plot_scaling_raises_until_backend():
    with pytest.raises(VisualizationNotInstalled):
        _scaling().plot.scaling()


def test_plot_call_raises_until_backend():
    with pytest.raises(VisualizationNotInstalled):
        _scaling().plot()


def test_plot_scaling_renders_when_a_backend_registers(monkeypatch):
    """Forward-compat: once a renderer registers, ``.plot.scaling()`` renders.

    The typed ``.scaling()`` method routes ``kind="scaling_fit"`` into
    ``to_plot_spec`` and the spec's ``render`` does the drawing.
    """
    import tsdynamics.registry as reg

    # registry.renderers does not exist yet; inject a non-empty stand-in.
    monkeypatch.setattr(reg, "renderers", ["matplotlib-stub"], raising=False)

    rendered = {}

    def fake_render(self, backend=None, **backend_kw):
        rendered["kind"] = self.kind
        rendered["backend"] = backend
        return "FIGURE"

    from tsdynamics.viz.spec import PlotSpec

    monkeypatch.setattr(PlotSpec, "render", fake_render, raising=True)

    out = _scaling().plot.scaling(backend="mpl")
    assert out == "FIGURE"
    assert rendered["backend"] == "mpl"
    # The semantic kind that reached the spec is SCALING_FIT.
    assert str(rendered["kind"]) == "scaling_fit"


# ---------------------------------------------------------------------------
# to_plot_spec — the SCALING_FIT description (no plot library pulled)
# ---------------------------------------------------------------------------


def test_to_plot_spec_builds_a_scaling_fit_spec():
    from tsdynamics.viz.spec import PlotKind

    spec = _scaling().to_plot_spec()
    assert spec.kind == PlotKind.SCALING_FIT
    assert spec.ndim == 2
    # Three layers: the full curve, the highlighted fit region, the fit line.
    assert len(spec.layers) == 3
    kinds = [layer.kind for layer in spec.layers]
    assert kinds == [PlotKind.SCATTER, PlotKind.MARKERS, PlotKind.LINE]


def test_to_plot_spec_fit_line_matches_intercept_and_slope():
    r = _scaling()
    spec = r.to_plot_spec()
    line = spec.layers[2]  # the LINE layer
    lo, hi = r.fit_region
    expected_x = np.array([r.abscissa[lo], r.abscissa[hi]])
    np.testing.assert_allclose(line.data["x"], expected_x)
    np.testing.assert_allclose(line.data["y"], r.intercept + r.estimate * expected_x)


def test_to_plot_spec_round_trips_through_dict():
    from tsdynamics.viz.spec import PlotSpec

    spec = _scaling().to_plot_spec()
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)


def test_to_plot_spec_empty_curve_has_only_the_scatter_layer():
    from tsdynamics.viz.spec import PlotKind

    spec = ScalingResult(estimate=1.0).to_plot_spec()  # empty arrays
    assert len(spec.layers) == 1  # no fit-region / fit-line layers to draw
    assert spec.layers[0].kind == PlotKind.SCATTER


def test_to_plot_spec_honours_kind_override():
    from tsdynamics.viz.spec import PlotKind

    spec = _scaling().to_plot_spec(kind="diagnostic_curve")
    assert spec.kind == PlotKind.DIAGNOSTIC_CURVE


# ---------------------------------------------------------------------------
# A domain subclass reparents cleanly (the WS-WRAP shape, smoke-tested here)
# ---------------------------------------------------------------------------


def test_domain_subclass_can_alias_estimate():
    @dataclass(frozen=True)
    class _Dim(ScalingResult):
        @property
        def dimension(self) -> float:
            return self.estimate

    r = _Dim(estimate=1.886, abscissa=np.arange(5.0), ordinate=np.arange(5.0), fit_region=(0, 4))
    assert r.dimension == 1.886
    assert float(r) == 1.886
    assert r.scaling_window == (0.0, 4.0)
    # The inherited repr survives the @dataclass redecoration (the WS-RESULT gotcha).
    assert repr(r).startswith("_Dim(")
    assert "estimate=1.886" in repr(r)
