"""
Analysis primitives for trajectories.

This subpackage hosts the algorithms backing the enrichment methods on
:class:`tsdynamics.Trajectory` (M1) and, in later milestones, event detection,
bifurcations, spectral toolkit, etc.  The Trajectory methods are thin wrappers
so the algorithms here stay independently unit-testable.

Currently exposed:

- :func:`decimate`, :func:`resample`, :func:`project`, :func:`window`
- :func:`derivative`, :func:`norm`
- :func:`local_maxima`, :func:`local_minima`, :func:`return_times`
- :func:`to_dataspec` — placeholder dict-builder until V1 ships ``DataSpec``.
"""

from __future__ import annotations

from .trajectory_ops import (
    decimate,
    derivative,
    local_maxima,
    local_minima,
    norm,
    project,
    resample,
    return_times,
    to_dataspec,
    window,
)

__all__ = [
    "decimate",
    "derivative",
    "local_maxima",
    "local_minima",
    "norm",
    "project",
    "resample",
    "return_times",
    "to_dataspec",
    "window",
]
