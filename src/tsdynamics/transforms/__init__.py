"""
Signal and data transforms that feed the analysis layer.

Three groups of transforms turn a :class:`~tsdynamics.data.Trajectory` (or array)
into the inputs the :mod:`~tsdynamics.analysis` quantifiers consume:

- **Spectral** (:mod:`~tsdynamics.transforms.spectral`) —
  :func:`power_spectral_density`, :func:`spectral_entropy`,
  :func:`spectral_centroid`, :func:`dominant_frequency`.
- **Preprocessing** (:mod:`~tsdynamics.transforms.preprocessing`) —
  :func:`detrend`, :func:`normalize`, and zero-phase Butterworth
  :func:`lowpass` / :func:`highpass` / :func:`bandpass` / :func:`bandstop`
  (shape-preserving: ``Trajectory`` in → ``Trajectory`` out).
- **Features** (:mod:`~tsdynamics.transforms.features`) —
  :func:`extract_features` over the named :data:`FEATURE_FUNCTIONS` catalogue,
  plus :func:`hjorth_parameters` and :func:`zero_crossing_rate`.

Every transform takes time along axis 0 and works channel-by-channel, so a single
1-D signal and a multi-channel ``(T, dim)`` trajectory go through the same call.

The in-tree transforms self-register into :data:`tsdynamics.registry.transforms`
(by name, with ``kind`` / ``produces`` metadata).  Out-of-tree transforms register
through the ``tsdynamics.transforms`` entry-point group (see
:mod:`tsdynamics.plugins`); :func:`discover_plugins` loads them into the same
registry, leaving in-tree names untouched.
"""

from collections.abc import Callable
from typing import Any

from .. import registry as _registry
from ..plugins import TRANSFORMS_GROUP, register_entry_points
from .features import (
    FEATURE_FUNCTIONS,
    extract_features,
    feature_names,
    hjorth_parameters,
    zero_crossing_rate,
)
from .preprocessing import (
    bandpass,
    bandstop,
    butter_filter,
    detrend,
    highpass,
    lowpass,
    normalize,
)
from .spectral import (
    dominant_frequency,
    power_spectral_density,
    spectral_centroid,
    spectral_entropy,
)

__all__ = [
    "FEATURE_FUNCTIONS",
    "bandpass",
    "bandstop",
    "butter_filter",
    "detrend",
    "discover_plugins",
    "dominant_frequency",
    "extract_features",
    "feature_names",
    "highpass",
    "hjorth_parameters",
    "lowpass",
    "normalize",
    "power_spectral_density",
    "spectral_centroid",
    "spectral_entropy",
    "zero_crossing_rate",
]


#: In-tree transforms registered at import: ``name -> (callable, metadata)``.
#: ``kind`` groups them; ``produces`` flags what a downstream consumer gets back
#: (``"signal"`` is shape-preserving, ``"spectrum"`` a frequency-domain pair,
#: ``"scalar"`` a per-channel number, ``"features"`` a named bag).
_INTREE_TRANSFORMS: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
    "power_spectral_density": (
        power_spectral_density,
        {"kind": "spectral", "produces": "spectrum"},
    ),
    "spectral_entropy": (spectral_entropy, {"kind": "spectral", "produces": "scalar"}),
    "spectral_centroid": (spectral_centroid, {"kind": "spectral", "produces": "scalar"}),
    "dominant_frequency": (dominant_frequency, {"kind": "spectral", "produces": "scalar"}),
    "detrend": (detrend, {"kind": "preprocessing", "produces": "signal"}),
    "normalize": (normalize, {"kind": "preprocessing", "produces": "signal"}),
    "lowpass": (lowpass, {"kind": "preprocessing", "produces": "signal"}),
    "highpass": (highpass, {"kind": "preprocessing", "produces": "signal"}),
    "bandpass": (bandpass, {"kind": "preprocessing", "produces": "signal"}),
    "bandstop": (bandstop, {"kind": "preprocessing", "produces": "signal"}),
    "extract_features": (extract_features, {"kind": "features", "produces": "features"}),
}


def _register_intree() -> None:
    """Register the built-in transforms into ``tsdynamics.registry.transforms``.

    Idempotent: re-registering the same object under its name is a no-op, so
    re-importing this package (test reloads, notebook cells) is safe.
    """
    for name, (obj, metadata) in _INTREE_TRANSFORMS.items():
        if name not in _registry.transforms:
            _registry.transforms.register(name, obj, **metadata)


def discover_plugins(*, strict: bool = False) -> list[str]:
    """Load out-of-tree transform plugins into :data:`tsdynamics.registry.transforms`.

    Walks the ``tsdynamics.transforms`` entry-point group and registers each
    loaded object under its entry-point name (see
    :func:`tsdynamics.plugins.register_entry_points`).  Called once at import;
    safe to re-invoke after installing a plugin.  Names already taken by an
    in-tree transform are left untouched.

    Parameters
    ----------
    strict : bool, default False
        Re-raise the first plugin load failure instead of warning and skipping.

    Returns
    -------
    list[str]
        The names newly registered by this call.
    """
    return register_entry_points(_registry.transforms, TRANSFORMS_GROUP, strict=strict)


# Register the in-tree transforms first so they own their names, then fold in any
# out-of-tree plugins (which skip names already present). Plugin failures are
# isolated inside `register_entry_points`.
_register_intree()
discover_plugins()


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
