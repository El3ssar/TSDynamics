"""Self-describing result objects for the transform layer.

The transform functions return bare arrays / tuples / dicts (and stay that way,
so ``freqs, psd = power_spectral_density(...)`` keeps unpacking).  The
*result-typed* entry points in :mod:`tsdynamics.transforms.spectral` and
:mod:`tsdynamics.transforms.features` wrap those returns in the small dataclasses
defined here so they carry a backend-agnostic
:meth:`~tsdynamics.viz.spec.PlotSpec` (and the ``.plot`` / notebook-display
sugar) — the transform-layer analogue of the
:class:`~tsdynamics.analysis._result.AnalysisResult` surface, kept deliberately
*off* that hierarchy so the analysis fake-renderer gate (which sweeps
``AnalysisResult.__subclasses__``) is not perturbed.

Three results, three semantic plot kinds:

- :class:`Spectrum` → :data:`~tsdynamics.viz.spec.PlotKind.POWER_SPECTRUM`
  (a one-sided PSD line, with the dominant frequency and spectral centroid drawn
  as ``vline`` annotations).
- :class:`Spectrogram` → :data:`~tsdynamics.viz.spec.PlotKind.SPECTROGRAM`
  (a time--frequency power image with a log-norm colorbar).
- :class:`FeatureSet` → :data:`~tsdynamics.viz.spec.PlotKind.FEATURE_BARS`
  (a bar over a categorical feature-name axis; a radar / parallel variant is
  selected through ``meta``).

Every ``to_plot_spec`` imports :mod:`tsdynamics.viz.spec` *lazily*, so building a
result — or importing :mod:`tsdynamics` — never pulls in a plotting backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from tsdynamics.viz.spec import PlotSpec


__all__ = [
    "FeatureSet",
    "Spectrogram",
    "Spectrum",
]


class _Plottable:
    """Internal mixin: give a transform result ``.plot`` + a notebook hook.

    Delegates to :class:`tsdynamics.viz.spec.Plottable` lazily (imported inside
    each method) so the transform package never imports the viz seam at module
    scope, exactly as :class:`tsdynamics.data.Trajectory` does.  Until a renderer
    registers itself, :meth:`plot` raises
    :class:`~tsdynamics.viz.spec.VisualizationNotInstalled` and the notebook hook
    no-ops, so a result still reprs as text.
    """

    def to_plot_spec(self, kind: str | None = None) -> PlotSpec:
        """Describe this result as a :class:`PlotSpec` (overridden per result)."""
        raise NotImplementedError  # pragma: no cover - every concrete result overrides

    def plot(self, backend: str | None = None, **tweaks: Any) -> Any:
        """Render this result through a visualization backend (lazy import)."""
        from tsdynamics.viz.spec import Plottable

        return Plottable.plot(cast("Plottable", self), backend, **tweaks)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Notebook display hook — no-ops until a rendering backend is installed."""
        from tsdynamics.viz.spec import Plottable

        return Plottable._repr_mimebundle_(cast("Plottable", self), include, exclude)


# ---------------------------------------------------------------------------
# Spectrum — a one-sided power spectral density
# ---------------------------------------------------------------------------


@dataclass
class Spectrum(_Plottable):
    """A one-sided power spectral density, with its spectral-feature markers.

    The result of :func:`tsdynamics.transforms.power_spectrum`: the PSD computed
    by Welch's method (Welch 1967) or a periodogram, plus the dominant frequency
    and power-weighted spectral centroid drawn as reference lines on the plot.

    Attributes
    ----------
    frequencies : numpy.ndarray
        Frequency bins, shape ``(n_freqs,)``.
    psd : numpy.ndarray
        Power at each bin, shape ``(n_freqs,)`` for a single channel or
        ``(n_freqs, channels)`` for a multi-channel signal.
    dominant_frequency : float or numpy.ndarray or None
        Frequency carrying the most power (per channel), or ``None`` if not
        computed.  Drawn as a ``vline`` marker.
    spectral_centroid : float or numpy.ndarray or None
        Power-weighted mean frequency (per channel), or ``None``.  Drawn as a
        ``vline`` marker.
    scaling : str
        ``"density"`` (units²/Hz) or ``"spectrum"`` (units²) — sets the y label.
    meta : dict
        Provenance (method / window / sampling rate).

    References
    ----------
    Welch, P. D. (1967). The use of fast Fourier transform for the estimation of
    power spectra: a method based on time averaging over short, modified
    periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2),
    70-73.
    """

    frequencies: np.ndarray
    psd: np.ndarray
    dominant_frequency: float | np.ndarray | None = None
    spectral_centroid: float | np.ndarray | None = None
    scaling: str = "density"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the curve arrays to contiguous ``float`` arrays."""
        self.frequencies = np.ascontiguousarray(self.frequencies, dtype=float)
        self.psd = np.ascontiguousarray(self.psd, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping (curve arrays become nested lists)."""
        return {
            "frequencies": self.frequencies.tolist(),
            "psd": self.psd.tolist(),
            "dominant_frequency": _to_jsonable(self.dominant_frequency),
            "spectral_centroid": _to_jsonable(self.spectral_centroid),
            "scaling": self.scaling,
            "meta": dict(self.meta),
        }

    def to_plot_spec(self, kind: str | None = None) -> PlotSpec:
        """Describe the spectrum as a ``POWER_SPECTRUM`` :class:`PlotSpec`.

        One ``LINE`` per channel over ``(frequency, power)``, a log y-scale
        (power spans orders of magnitude), and the dominant frequency /
        spectral centroid as ``vline`` annotations.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (a
            :class:`~tsdynamics.viz.spec.PlotKind` value).  ``None`` uses
            ``POWER_SPECTRUM``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.POWER_SPECTRUM
        freqs = self.frequencies
        psd = self.psd
        cols = psd if psd.ndim == 2 else psd[:, None]
        n_chan = cols.shape[1]

        layers = [
            Layer(
                PlotKind.LINE,
                {"x": freqs, "y": cols[:, j]},
                label=(f"channel {j}" if n_chan > 1 else "PSD"),
            )
            for j in range(n_chan)
        ]

        ylabel = "PSD" if self.scaling == "density" else "power"
        annotations = self._marker_annotations()
        spec = PlotSpec(
            kind=spec_kind,
            ndim=2,
            title="Spectrum",
            x=Axis(label="frequency"),
            y=Axis(label=ylabel, scale="log"),
            layers=layers,
            annotations=annotations,
            legend=Legend() if n_chan > 1 else None,
            meta=dict(self.meta),
        )
        return spec

    def _marker_annotations(self) -> list[Any]:
        """Build the dominant-frequency / centroid ``vline`` annotations."""
        from tsdynamics.viz.spec import Annotation

        annotations: list[Any] = []
        for value, text, color in (
            (self.dominant_frequency, "dominant", "tab:red"),
            (self.spectral_centroid, "centroid", "tab:green"),
        ):
            for x in _scalar_markers(value):
                annotations.append(Annotation(kind="vline", text=text, x=x, style={"color": color}))
        return annotations


# ---------------------------------------------------------------------------
# Spectrogram — a time-frequency power image
# ---------------------------------------------------------------------------


@dataclass
class Spectrogram(_Plottable):
    """A short-time-Fourier-transform power image: time x frequency x power.

    The result of :func:`tsdynamics.transforms.spectrogram` — the windowed PSD
    over a sliding window, the standard view of how a signal's spectral content
    evolves in time.

    Attributes
    ----------
    times : numpy.ndarray
        Segment centre times, shape ``(n_times,)``.
    frequencies : numpy.ndarray
        Frequency bins, shape ``(n_freqs,)``.
    power : numpy.ndarray
        Power, shape ``(n_freqs, n_times)`` (frequency down rows, time across
        columns — the natural image orientation).
    meta : dict
        Provenance (window / nperseg / noverlap / sampling rate).
    """

    times: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the grid + image arrays to contiguous ``float`` arrays."""
        self.times = np.ascontiguousarray(self.times, dtype=float)
        self.frequencies = np.ascontiguousarray(self.frequencies, dtype=float)
        self.power = np.ascontiguousarray(self.power, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping (grid + image arrays become nested lists)."""
        return {
            "times": self.times.tolist(),
            "frequencies": self.frequencies.tolist(),
            "power": self.power.tolist(),
            "meta": dict(self.meta),
        }

    def to_plot_spec(self, kind: str | None = None) -> PlotSpec:
        """Describe the spectrogram as a ``SPECTROGRAM`` :class:`PlotSpec`.

        A single ``IMAGE`` layer carrying the ``time`` / ``frequency`` axes and
        the ``power`` field in its ``"c"`` channel, with a log-norm
        :class:`~tsdynamics.viz.spec.Colorbar` (power spans orders of magnitude).

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``SPECTROGRAM``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Colorbar, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.SPECTROGRAM
        layer = Layer(
            PlotKind.IMAGE,
            {"x": self.times, "y": self.frequencies, "c": self.power},
        )
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title="Spectrogram",
            x=Axis(label="time"),
            y=Axis(label="frequency"),
            layers=[layer],
            colorbar=Colorbar(label="power", cmap="magma", norm="log"),
            meta=dict(self.meta),
        )


# ---------------------------------------------------------------------------
# FeatureSet — a named bag of scalar features
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet(_Plottable):
    """A named bag of scalar per-channel features (the feature vector).

    The result of :func:`tsdynamics.transforms.feature_set`: the catalogue of
    scalar descriptors :func:`tsdynamics.transforms.extract_features` computes,
    held with their names so the whole vector plots as labelled bars.

    Attributes
    ----------
    features : dict
        Maps each feature name to its value — a ``float`` for a single channel or
        a ``(channels,)`` array for a multi-channel signal.
    meta : dict
        Provenance, and the plot variant: ``meta["variant"]`` of ``"bar"``
        (default), ``"radar"``, or ``"parallel"`` switches
        :meth:`to_plot_spec` between a bar chart and a radar / parallel-axes
        line over the same categorical feature axis.
    """

    features: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        """The feature names, in catalogue (insertion) order."""
        return list(self.features)

    def matrix(self) -> np.ndarray:
        """Return the feature values as a ``(n_features, channels)`` float array.

        A single-channel signal yields a ``(n_features, 1)`` column; this is the
        per-bar height (column ``j`` is channel ``j``).
        """
        cols = []
        for value in self.features.values():
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            cols.append(arr)
        width = max((c.size for c in cols), default=1)
        return np.array([np.broadcast_to(c, (width,)) for c in cols], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the named features + provenance."""
        return {
            "features": {k: _to_jsonable(v) for k, v in self.features.items()},
            "meta": dict(self.meta),
        }

    def to_plot_spec(self, kind: str | None = None) -> PlotSpec:
        r"""Describe the feature vector as a ``FEATURE_BARS`` :class:`PlotSpec`.

        Builds a categorical x-axis of the feature names and one
        :data:`~tsdynamics.viz.spec.PlotKind.BAR` layer per channel: the ``"cat"``
        channel carries the integer feature index pairing with
        :attr:`~tsdynamics.viz.spec.Axis.categories`, and ``"y"`` the bar height.
        Setting ``meta["variant"]`` to ``"radar"`` or ``"parallel"`` instead emits
        a closed / open ``LINE`` over the same categorical axis (the renderer
        draws it on a polar / parallel layout) — the requested radar / parallel
        variant, selected through ``meta`` so the uniform
        ``to_plot_spec(self, kind=None)`` signature is preserved.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``FEATURE_BARS``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.FEATURE_BARS
        names = self.names
        mat = self.matrix()  # (n_features, channels)
        n_feat, n_chan = mat.shape if mat.ndim == 2 else (mat.shape[0], 1)
        cat = np.arange(n_feat, dtype=float)
        variant = str(self.meta.get("variant", "bar"))

        layers: list[Layer] = []
        if variant in ("radar", "parallel"):
            # A radar / parallel variant: a LINE over the categorical axis (the
            # renderer lays it out on a polar / parallel-coordinate frame).  Close
            # the loop for a radar so the polygon joins back to the first axis.
            close = variant == "radar"
            for j in range(n_chan):
                xs = np.append(cat, cat[0]) if (close and n_feat) else cat
                ys = mat[:, j]
                ys = np.append(ys, ys[0]) if (close and n_feat) else ys
                layers.append(
                    Layer(
                        PlotKind.LINE,
                        {"x": xs, "y": ys, "cat": xs},
                        label=(f"channel {j}" if n_chan > 1 else "features"),
                    )
                )
        else:
            for j in range(n_chan):
                layers.append(
                    Layer(
                        PlotKind.BAR,
                        {"x": cat, "y": mat[:, j], "cat": cat},
                        label=(f"channel {j}" if n_chan > 1 else "features"),
                    )
                )

        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title="FeatureSet",
            x=Axis(label="feature", scale="categorical", categories=names, ticks=list(cat)),
            y=Axis(label="value"),
            layers=layers,
            legend=Legend() if n_chan > 1 else None,
            meta=dict(self.meta),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Coerce a scalar / array feature value to a JSON-friendly form."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar_markers(value: float | np.ndarray | None) -> list[float]:
    """Return the finite marker positions for a scalar / per-channel value.

    A ``None`` yields no marker; a scalar one; a per-channel array one marker per
    channel (non-finite entries dropped).
    """
    if value is None:
        return []
    arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    return [float(v) for v in arr if np.isfinite(v)]
