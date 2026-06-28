"""Tiny shared helpers for the result layers (analysis *and* transforms).

The analysis result hierarchy (:mod:`tsdynamics.analysis._result`) and the
transform result dataclasses (:mod:`tsdynamics.transforms._result`) both build
backend-agnostic :class:`~tsdynamics.viz.spec.PlotSpec` objects, and both open
every ``to_plot_spec`` with the identical "resolve the semantic kind" line.  That
one line was copy-pasted a dozen times across the two layers; it lives here once
so both call sites share it.

This module deliberately carries **no result class and no viz import at module
scope** — it is a leaf helper, so importing it never pulls a plotting backend and
it stays off the :class:`~tsdynamics.analysis._result.AnalysisResult` hierarchy
the fake-renderer gate sweeps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsdynamics.viz.spec import PlotKind


def resolve_plot_kind(kind: str | None, default: PlotKind) -> Any:
    """Return ``PlotKind(kind)`` for an explicit override, else ``default``.

    The one-line semantic-kind resolution every ``to_plot_spec`` opens with: a
    caller-supplied ``kind`` string is coerced through the closed
    :class:`~tsdynamics.viz.spec.PlotKind` vocabulary (raising on an unknown
    value, exactly as before), and ``None`` falls back to the result's natural
    ``default`` kind.

    Parameters
    ----------
    kind : str or None
        An explicit semantic-kind override (a ``PlotKind`` value), or ``None``.
    default : PlotKind
        The result's natural kind, used when ``kind`` is ``None``.

    Returns
    -------
    PlotKind
    """
    from tsdynamics.viz.spec import PlotKind

    return PlotKind(kind) if kind is not None else default
