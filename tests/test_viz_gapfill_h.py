"""Transform-layer visualization gate (stream GAPFILL-H).

The transform result types (:class:`~tsdynamics.transforms.Spectrum`,
:class:`~tsdynamics.transforms.Spectrogram`, :class:`~tsdynamics.transforms.FeatureSet`)
each carry a backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` through
``to_plot_spec``.  These tests assert each result maps to its correct semantic
:class:`~tsdynamics.viz.spec.PlotKind`, builds a valid spec (real kinds, real
layer marks, the documented data channels), and round-trips losslessly through
``to_dict`` / ``from_dict`` — engine-free (no ``tsdynamics._rust``), fast tier.

A property/round-trip test fuzzes the new ``spectrogram`` transform over random
window sizes and signal lengths so the time--frequency image shape, axis grids,
and JSON round-trip hold for arbitrary inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tsdynamics.transforms as tx
from tsdynamics import registry
from tsdynamics.transforms import FeatureSet, Spectrogram, Spectrum
from tsdynamics.viz.spec import PlotKind, PlotSpec

FS = 200.0


def _tone(n: int = 2048, freq: float = 0.05) -> np.ndarray:
    """A pure cosine at ``freq`` cycles/sample (fs=1 convention), length ``n``."""
    return np.cos(2.0 * np.pi * freq * np.arange(n, dtype=float))


# ---------------------------------------------------------------------------
# Spectrum -> POWER_SPECTRUM
# ---------------------------------------------------------------------------


def test_power_spectrum_returns_spectrum_result() -> None:
    """``power_spectrum`` wraps the PSD in a self-describing :class:`Spectrum`."""
    sp = tx.power_spectrum(_tone(2048, 0.05), fs=1.0)
    assert isinstance(sp, Spectrum)
    # The bare PSD pair is unchanged (backward-compat) and matches the wrapped one.
    freqs, psd = tx.power_spectral_density(_tone(2048, 0.05), fs=1.0)
    assert np.allclose(sp.frequencies, freqs)
    assert np.allclose(sp.psd, psd)


def test_spectrum_to_plot_spec_kind_and_markers() -> None:
    """The spectrum spec is a ``POWER_SPECTRUM`` with the feature lines as vlines."""
    sp = tx.power_spectrum(_tone(2048, 0.05), fs=1.0)
    spec = sp.to_plot_spec()
    assert isinstance(spec, PlotSpec)
    assert spec.kind == PlotKind.POWER_SPECTRUM
    assert spec.y.scale == "log"  # power spans orders of magnitude
    assert spec.layers and all(layer.kind == PlotKind.LINE for layer in spec.layers)
    # Dominant frequency + spectral centroid are drawn as vline annotations.
    vlines = {a.text for a in spec.annotations if a.kind == "vline"}
    assert {"dominant", "centroid"} <= vlines
    # The dominant-frequency marker sits near the injected 0.05 cycles/sample line.
    dom = next(a.x for a in spec.annotations if a.text == "dominant")
    assert abs(dom - 0.05) < 0.02


def test_spectrum_no_markers_has_no_annotations() -> None:
    """``markers=False`` skips the extra reductions and the vline annotations."""
    sp = tx.power_spectrum(_tone(1024, 0.1), fs=1.0, markers=False)
    assert sp.dominant_frequency is None
    assert sp.spectral_centroid is None
    assert sp.to_plot_spec().annotations == []


def test_spectrum_multichannel_one_line_per_channel() -> None:
    """A multi-channel spectrum draws one ``LINE`` per channel with a legend."""
    sig = np.column_stack([_tone(2048, 0.05), _tone(2048, 0.12)])
    spec = tx.power_spectrum(sig, fs=1.0).to_plot_spec()
    assert len(spec.layers) == 2
    assert spec.legend is not None


def test_spectrum_round_trips() -> None:
    """The spectrum spec round-trips losslessly through ``to_dict`` / ``from_dict``."""
    spec = tx.power_spectrum(_tone(1024, 0.07), fs=1.0).to_plot_spec()
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert len(rebuilt.layers) == len(spec.layers)
    assert len(rebuilt.annotations) == len(spec.annotations)


# ---------------------------------------------------------------------------
# Spectrogram -> SPECTROGRAM
# ---------------------------------------------------------------------------


def test_spectrogram_returns_result_and_shapes() -> None:
    """``spectrogram`` returns a :class:`Spectrogram` with a consistent power grid."""
    sg = tx.spectrogram(_tone(4096, 0.05), fs=FS, nperseg=256)
    assert isinstance(sg, Spectrogram)
    assert sg.power.shape == (sg.frequencies.size, sg.times.size)
    assert sg.times.ndim == 1 and sg.frequencies.ndim == 1


def test_spectrogram_to_plot_spec_image_with_log_colorbar() -> None:
    """The spectrogram spec is a ``SPECTROGRAM`` IMAGE with a log-norm colorbar."""
    sg = tx.spectrogram(_tone(4096, 0.05), fs=FS, nperseg=256)
    spec = sg.to_plot_spec()
    assert spec.kind == PlotKind.SPECTROGRAM
    assert len(spec.layers) == 1
    layer = spec.layers[0]
    assert layer.kind == PlotKind.IMAGE
    assert {"x", "y", "c"} <= set(layer.data)  # power in the "c" channel
    assert spec.colorbar is not None
    assert spec.colorbar.norm == "log"
    assert spec.colorbar.cmap is not None


def test_spectrogram_multichannel_uses_first_channel() -> None:
    """A multi-channel input reduces to the first channel (a single image)."""
    sig = np.column_stack([_tone(2048, 0.05), _tone(2048, 0.2)])
    sg = tx.spectrogram(sig, fs=FS, nperseg=128)
    single = tx.spectrogram(sig[:, 0], fs=FS, nperseg=128)
    assert np.allclose(sg.power, single.power)


# ---------------------------------------------------------------------------
# FeatureSet -> FEATURE_BARS
# ---------------------------------------------------------------------------


def test_feature_set_returns_result_and_matches_extract() -> None:
    """``feature_set`` carries the same values as the bare ``extract_features`` dict."""
    sig = _tone(2048, 0.05)
    fset = tx.feature_set(sig, fs=1.0)
    assert isinstance(fset, FeatureSet)
    bare = tx.extract_features(sig, fs=1.0)
    assert set(fset.features) == set(bare)
    for name in bare:
        assert np.allclose(fset.features[name], bare[name])


def test_feature_set_bar_spec_over_categorical_axis() -> None:
    """The default variant is a ``BAR`` over a categorical feature-name axis."""
    fset = tx.feature_set(_tone(2048, 0.05), fs=1.0)
    spec = fset.to_plot_spec()
    assert spec.kind == PlotKind.FEATURE_BARS
    assert spec.x.scale == "categorical"
    assert list(spec.x.categories) == tx.feature_names()
    layer = spec.layers[0]
    assert layer.kind == PlotKind.BAR
    assert "cat" in layer.data
    # One bar per feature name.
    assert layer.data["y"].size == len(tx.feature_names())


@pytest.mark.parametrize("variant", ["radar", "parallel"])
def test_feature_set_radar_parallel_variant_is_a_line(variant: str) -> None:
    """The radar / parallel variant (via ``meta``) emits a ``LINE`` over the axis."""
    fset = tx.feature_set(_tone(2048, 0.05), fs=1.0, variant=variant)
    assert fset.meta["variant"] == variant
    spec = fset.to_plot_spec()
    assert spec.kind == PlotKind.FEATURE_BARS
    layer = spec.layers[0]
    assert layer.kind == PlotKind.LINE
    n_feat = len(tx.feature_names())
    # A radar closes the loop (n+1 points); parallel does not.
    expected = n_feat + 1 if variant == "radar" else n_feat
    assert layer.data["y"].size == expected


def test_feature_set_multichannel_one_bar_layer_per_channel() -> None:
    """A multi-channel feature set draws one bar layer per channel with a legend."""
    sig = np.column_stack([_tone(2048, 0.05), _tone(2048, 0.2)])
    spec = tx.feature_set(sig, fs=1.0).to_plot_spec()
    assert len(spec.layers) == 2
    assert spec.legend is not None
    for layer in spec.layers:
        assert layer.kind == PlotKind.BAR


def test_feature_set_round_trips() -> None:
    """The feature-bar spec round-trips losslessly through the JSON dict form."""
    spec = tx.feature_set(_tone(1024, 0.05), fs=1.0).to_plot_spec()
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert rebuilt.x.scale == "categorical"
    assert list(rebuilt.x.categories) == list(spec.x.categories)


# ---------------------------------------------------------------------------
# to_dict on the results themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        tx.power_spectrum(_tone(1024, 0.05), fs=1.0),
        tx.spectrogram(_tone(2048, 0.05), fs=FS, nperseg=128),
        tx.feature_set(_tone(1024, 0.05), fs=1.0),
    ],
    ids=["spectrum", "spectrogram", "feature_set"],
)
def test_result_to_dict_is_json_friendly(result: object) -> None:
    """Each result's ``to_dict`` is a stdlib JSON-serializable mapping."""
    import json

    payload = result.to_dict()  # type: ignore[attr-defined]
    assert isinstance(payload, dict)
    json.dumps(payload)  # must not raise


# ---------------------------------------------------------------------------
# Registry wiring + meta-QA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["power_spectrum", "spectrogram", "feature_set"])
def test_new_transforms_registered(name: str) -> None:
    """The three new transforms self-register and re-export consistently."""
    assert name in registry.transforms
    entry = registry.transforms.entry(name)
    assert entry.obj is getattr(tx, name)
    assert entry.metadata["produces"] == "result"
    assert callable(entry.obj)
    doc = entry.obj.__doc__
    assert isinstance(doc, str) and doc.strip()


# ---------------------------------------------------------------------------
# Property / round-trip: the new spectrogram transform
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=512, max_value=4096),
    nperseg=st.sampled_from([32, 64, 128, 256]),
    freq=st.floats(min_value=0.02, max_value=0.45),
)
@settings(max_examples=25, deadline=None)
def test_spectrogram_property_shape_and_round_trip(n: int, nperseg: int, freq: float) -> None:
    """For any signal length / window, the image shape, grids, and round-trip hold."""
    sig = _tone(n, freq)
    sg = tx.spectrogram(sig, fs=FS, nperseg=nperseg)

    # The power image is (n_freqs, n_times), finite and non-negative (a PSD).
    assert sg.power.shape == (sg.frequencies.size, sg.times.size)
    assert sg.power.size > 0
    assert np.all(np.isfinite(sg.power))
    assert np.all(sg.power >= 0.0)
    # Frequencies are a non-negative, increasing grid bounded by Nyquist.
    assert sg.frequencies[0] >= 0.0
    assert np.all(np.diff(sg.frequencies) > 0.0)
    assert sg.frequencies[-1] <= FS / 2.0 + 1e-9
    # Times are strictly increasing.
    assert np.all(np.diff(sg.times) > 0.0)

    # The spec carries the right kind and round-trips losslessly.
    spec = sg.to_plot_spec()
    assert spec.kind == PlotKind.SPECTROGRAM
    rebuilt = PlotSpec.from_dict(spec.to_dict())
    assert rebuilt.kind == spec.kind
    assert np.allclose(rebuilt.layers[0].data["c"], spec.layers[0].data["c"])
