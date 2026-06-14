"""
Signal and data transforms that feed the analysis layer.

Skeleton package (stream T-XFORM fills it): spectral estimators (PSD, spectral
entropy), detrend/filter/normalize, and generic feature extractors that turn a
:class:`~tsdynamics.data.Trajectory` or array into the inputs the
:mod:`~tsdynamics.analysis` quantifiers consume.

Out-of-tree transforms register through the ``tsdynamics.transforms`` entry-point
group (see :mod:`tsdynamics.plugins`); :func:`discover_plugins` loads them into
:data:`tsdynamics.registry.transforms`.
"""

from .. import registry as _registry
from ..plugins import TRANSFORMS_GROUP, register_entry_points

__all__: list[str] = []


def discover_plugins(*, strict: bool = False) -> list[str]:
    """Load out-of-tree transform plugins into :data:`tsdynamics.registry.transforms`.

    Walks the ``tsdynamics.transforms`` entry-point group and registers each
    loaded object under its entry-point name (see
    :func:`tsdynamics.plugins.register_entry_points`).  Called once at import;
    safe to re-invoke after installing a plugin.

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


# Populate the transforms registry from out-of-tree plugins at import. In-tree
# transforms register themselves (stream T-XFORM); plugin failures are isolated
# inside `register_entry_points`.
discover_plugins()
