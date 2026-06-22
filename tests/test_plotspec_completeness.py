"""Completeness tests for the PlotSpec IR color/legend fields (VIZ-SPEC-COMPLETENESS).

The PlotSpec IR gained three spec-level color/legend fields — ``clim`` (the
color range), ``colorbar`` (a :class:`Colorbar` legend), and ``legend`` (a
:class:`Legend`) — plus the ``colorize`` / ``autocolor`` tweaks.  These tests
cover the acceptance pillars:

1. The new fields round-trip through ``to_dict`` / ``from_dict`` (including the
   nested :class:`Colorbar` / :class:`Legend` dataclasses), and the dict stays
   JSON-serializable.
2. Image / colored kinds *express* a colorbar and a color range via
   :meth:`PlotSpec.autocolor` (and :meth:`has_color_channel`).
3. The new fields are additive: a spec that sets none of them round-trips with
   them ``None`` (no regression to the existing pillars, which keep passing in
   ``tests/test_plotspec.py``).
"""

import json

import numpy as np
import pytest

from tsdynamics.viz.spec import (
    Axis,
    Colorbar,
    Layer,
    Legend,
    PlotKind,
    PlotSpec,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _colored_spec() -> PlotSpec:
    """An IMAGE spec exercising every new color/legend field that round-trips."""
    field = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]])
    return PlotSpec(
        kind=PlotKind.BASINS_IMAGE,
        ndim=2,
        aspect="equal",
        x=Axis("x"),
        y=Axis("y"),
        clim=(0.0, 4.0),
        colorbar=Colorbar(
            label="basin",
            location="bottom",
            ticks=[0.0, 2.0, 4.0],
            tickformat="%d",
            show=True,
        ),
        legend=Legend(show=True, location="upper right", title="attractors"),
        layers=[Layer(PlotKind.IMAGE, {"c": field}, label="basins")],
    )


# ---------------------------------------------------------------------------
# 1. New fields round-trip
# ---------------------------------------------------------------------------


def test_new_fields_are_json_serializable():
    spec = _colored_spec()
    d = spec.to_dict()
    # A stray dataclass / ndarray would make json.dumps raise.
    text = json.dumps(d)
    assert isinstance(text, str)
    assert d["clim"] == [0.0, 4.0]
    assert isinstance(d["colorbar"], dict)
    assert isinstance(d["legend"], dict)


def test_round_trip_preserves_clim_colorbar_legend():
    spec = _colored_spec()
    rebuilt = PlotSpec.from_dict(spec.to_dict())

    # clim becomes a (float, float) tuple again.
    assert rebuilt.clim == (0.0, 4.0)
    assert isinstance(rebuilt.clim, tuple)
    assert all(isinstance(v, float) for v in rebuilt.clim)

    # The Colorbar survives field-for-field.
    assert isinstance(rebuilt.colorbar, Colorbar)
    assert rebuilt.colorbar.label == "basin"
    assert rebuilt.colorbar.location == "bottom"
    assert list(rebuilt.colorbar.ticks) == [0.0, 2.0, 4.0]
    assert rebuilt.colorbar.tickformat == "%d"
    assert rebuilt.colorbar.show is True

    # The Legend survives field-for-field.
    assert isinstance(rebuilt.legend, Legend)
    assert rebuilt.legend.show is True
    assert rebuilt.legend.location == "upper right"
    assert rebuilt.legend.title == "attractors"


def test_colorbar_and_legend_dataclasses_round_trip_standalone():
    cbar = Colorbar(label="L", location="left", ticks=[1.0, 2.0], tickformat="%.2f", show=False)
    assert Colorbar.from_dict(cbar.to_dict()) == cbar

    leg = Legend(show=False, location="center", title="T")
    assert Legend.from_dict(leg.to_dict()) == leg


def test_colorbar_ticks_none_round_trips():
    cbar = Colorbar(label="L")  # ticks left at None
    rebuilt = Colorbar.from_dict(cbar.to_dict())
    assert rebuilt.ticks is None
    assert rebuilt.tickformat is None


# ---------------------------------------------------------------------------
# 2. Additive: a spec setting none of the new fields keeps None through a trip
# ---------------------------------------------------------------------------


def test_uncolored_spec_round_trips_with_none_fields():
    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        ndim=1,
        layers=[Layer(PlotKind.LINE, {"x": np.arange(4.0), "y": np.arange(4.0)})],
    )
    assert spec.clim is None
    assert spec.colorbar is None
    assert spec.legend is None

    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.clim is None
    assert rebuilt.colorbar is None
    assert rebuilt.legend is None
    # And the pre-existing structure is untouched.
    np.testing.assert_array_equal(rebuilt.layers[0].data["y"], np.arange(4.0))


def test_clim_is_coerced_to_float_tuple_on_construction():
    spec = PlotSpec(kind=PlotKind.IMAGE, clim=(np.float64(1), np.int64(5)))
    assert spec.clim == (1.0, 5.0)
    assert all(isinstance(v, float) for v in spec.clim)


# ---------------------------------------------------------------------------
# 3. Image / colored kinds express a colorbar + range (autocolor)
# ---------------------------------------------------------------------------


def test_has_color_channel_detects_image_kinds():
    for kind in (
        PlotKind.IMAGE,
        PlotKind.BASINS_IMAGE,
        PlotKind.RECURRENCE_PLOT,
        PlotKind.SPACETIME,
        PlotKind.SURFACE3D,
    ):
        assert PlotSpec(kind=kind).has_color_channel(), kind


def test_has_color_channel_detects_scalar_c_channel():
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_2D,
        layers=[Layer(PlotKind.SCATTER, {"x": [0.0, 1.0], "y": [0.0, 1.0], "c": [0.2, 0.9]})],
    )
    assert spec.has_color_channel()


def test_uncolored_spec_has_no_color_channel():
    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(PlotKind.LINE, {"x": [0.0, 1.0], "y": [0.0, 1.0]})],
    )
    assert not spec.has_color_channel()


def test_autocolor_attaches_colorbar_and_infers_clim():
    field = np.array([[1.0, 5.0], [3.0, np.nan]])  # NaN must be ignored
    spec = PlotSpec(
        kind=PlotKind.BASINS_IMAGE,
        layers=[Layer(PlotKind.IMAGE, {"c": field})],
    ).autocolor()

    assert isinstance(spec.colorbar, Colorbar)  # a colorbar is now expressed
    assert spec.clim == (1.0, 5.0)  # the finite color range
    # round-trips like any other colored spec
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.clim == (1.0, 5.0)
    assert isinstance(rebuilt.colorbar, Colorbar)


def test_autocolor_infers_clim_from_z_for_image_layer():
    spec = PlotSpec(
        kind=PlotKind.SURFACE3D,
        ndim=3,
        layers=[Layer(PlotKind.SURFACE3D, {"z": np.array([[-2.0, 0.0], [2.0, 4.0]])})],
    ).autocolor()
    assert spec.clim == (-2.0, 4.0)


def test_autocolor_is_noop_on_uncolored_spec():
    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(PlotKind.LINE, {"x": [0.0, 1.0], "y": [0.0, 1.0]})],
    ).autocolor()
    assert spec.colorbar is None
    assert spec.clim is None


def test_autocolor_does_not_override_caller_set_clim_or_colorbar():
    spec = PlotSpec(
        kind=PlotKind.IMAGE,
        clim=(-10.0, 10.0),
        colorbar=Colorbar(label="kept"),
        layers=[Layer(PlotKind.IMAGE, {"c": np.array([[0.0, 1.0]])})],
    ).autocolor()
    assert spec.clim == (-10.0, 10.0)  # not recomputed to (0, 1)
    assert spec.colorbar.label == "kept"  # not replaced by a default


def test_autocolor_all_nan_color_leaves_clim_none_but_attaches_colorbar():
    spec = PlotSpec(
        kind=PlotKind.IMAGE,
        layers=[Layer(PlotKind.IMAGE, {"c": np.array([[np.nan, np.nan]])})],
    ).autocolor()
    assert spec.colorbar is not None  # the kind still expresses a colorbar
    assert spec.clim is None  # but no finite range to infer


# ---------------------------------------------------------------------------
# 4. colorize tweak mutates-and-chains like the other tweaks
# ---------------------------------------------------------------------------


def test_colorize_mutates_and_chains():
    spec = PlotSpec(kind=PlotKind.IMAGE)
    returned = spec.colorize(clim=(0.0, 1.0), colorbar=True, legend=True)
    assert returned is spec  # mutation, not a copy
    assert spec.clim == (0.0, 1.0)
    assert isinstance(spec.colorbar, Colorbar)
    assert isinstance(spec.legend, Legend)


def test_colorize_accepts_explicit_objects_and_can_drop_with_false():
    spec = PlotSpec(kind=PlotKind.IMAGE, colorbar=Colorbar(), legend=Legend())
    spec.colorize(colorbar=Colorbar(label="grad"), legend=False)
    assert spec.colorbar.label == "grad"
    assert spec.legend is None  # False drops it


def test_colorize_only_touches_passed_fields():
    spec = PlotSpec(kind=PlotKind.IMAGE, clim=(2.0, 3.0), colorbar=Colorbar(label="L"))
    spec.colorize(legend=True)  # clim / colorbar untouched
    assert spec.clim == (2.0, 3.0)
    assert spec.colorbar.label == "L"
    assert isinstance(spec.legend, Legend)


def test_clim_inline_tweak_routes_through_colorize():
    from tsdynamics.viz.spec import _apply_inline_tweaks

    spec = PlotSpec(kind=PlotKind.IMAGE)
    leftover = _apply_inline_tweaks(spec, {"clim": (0.0, 7.0), "legend": True, "ax": "passthrough"})
    assert spec.clim == (0.0, 7.0)
    assert isinstance(spec.legend, Legend)
    assert leftover == {"ax": "passthrough"}


# ---------------------------------------------------------------------------
# 5. The new names are exported and the no-plot-import invariant still holds
# ---------------------------------------------------------------------------


def test_new_names_are_exported():
    import tsdynamics.viz.spec as spec_mod

    assert "Colorbar" in spec_mod.__all__
    assert "Legend" in spec_mod.__all__


@pytest.mark.parametrize("name", ["Colorbar", "Legend"])
def test_new_dataclasses_have_docstrings(name):
    import tsdynamics.viz.spec as spec_mod

    obj = getattr(spec_mod, name)
    assert obj.__doc__ and obj.__doc__.strip()
