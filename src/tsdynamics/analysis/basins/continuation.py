r"""
Global continuation: track attractors and their basins across a parameter.

Sweep one parameter, recompute the attractors and basin fractions at every value,
and **match** the attractors between consecutive values by state-space distance so
their ids stay consistent — the recurrences-find-and-match idea of

    G. Datseris, K. L. Rossi and A. Wagemakers, "Framework for global stability
    analysis of dynamical systems", *Chaos* **33**, 073151 (2023).

A basin fraction collapsing to zero as the parameter moves is a **tipping** event
(an attractor and its basin annihilating); :func:`tipping_points` reads those off
the continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...data import Ball, Box, Grid, set_distance
from .._result import AnalysisResult, CollectionResult
from ._common import DIVERGED_COLOR, PALETTE, _palette_indices
from .attractors import Attractor, _reject_unsupported
from .basins import basin_fractions

if TYPE_CHECKING:
    from ...data.sampling import _SetMethod

__all__ = [
    "ContinuationResult",
    "continuation",
    "tipping_points",
]


@dataclass(frozen=True)
class ContinuationResult(AnalysisResult):
    """
    Attractors and basin fractions tracked across a parameter sweep.

    Attributes
    ----------
    param : str
        The swept parameter name.
    values : ndarray
        Parameter values, in sweep order.
    fractions : dict[int, ndarray]
        Global attractor id → basin fraction at each value (``nan`` where the
        attractor is absent).
    attractors : list[dict[int, Attractor]]
        Per value, the located attractors keyed by their *global* (matched) id.
    diverged : ndarray
        Diverged-or-untracked fraction at each value: the diverged share **plus**
        the basin mass of any attractor dropped by ``min_fraction``, so the
        tracked bands and this share together tile :math:`[0, 1]`.
    """

    param: str = ""
    values: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)
    fractions: dict[int, np.ndarray] = field(default_factory=dict, compare=False)
    attractors: list[dict[int, Attractor]] = field(default_factory=list, repr=False, compare=False)
    diverged: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)

    @property
    def ids(self) -> list[int]:
        """Sorted global attractor ids seen anywhere in the sweep."""
        return sorted(self.fractions)

    def tipping_points(self, *, threshold: float = 0.0) -> CollectionResult:
        """Tipping events along this continuation (see :func:`tipping_points`)."""
        return tipping_points(self, threshold=threshold)

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the continuation as a backend-agnostic :class:`PlotSpec`.

        Builds a ``CONTINUATION`` spec — the basin fractions **stacked** against
        the swept parameter, one filled ``AREA`` band per global attractor id (its
        ``"lo"`` / ``"hi"`` channels are the cumulative fraction below and above
        the band).  The attractor bands fill the tracked share; the remaining gap
        up to ``1`` is the diverged / untracked mass (:attr:`diverged`), so the
        bands plus that gap tile :math:`[0, 1]`.  Each **tipping** event from
        :meth:`tipping_points` — a basin annihilating (``"disappear"``) or being
        born (``"appear"``) — is drawn as a vertical
        :class:`~tsdynamics.viz.spec.Annotation` at the parameter value where it
        happens.  The bands share the attractor palette (``tab20``, recorded in
        ``meta["palette"]``) so an id keeps its colour across the basin views.  A
        :class:`~tsdynamics.viz.spec.Legend` is attached when more than one
        attractor is tracked.  The :mod:`tsdynamics.viz.spec` import is lazy, so
        building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"continuation"``).  ``None`` uses
            ``CONTINUATION``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import (
            Annotation,
            Axis,
            Colorbar,
            Layer,
            Legend,
            PlotKind,
            PlotSpec,
        )

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.CONTINUATION
        values = np.asarray(self.values, dtype=float)
        ids = self.ids
        swatch = _palette_indices(ids)

        # Stack the (nan-as-zero) fractions so the bands tile [0, 1] per value.
        cumulative = np.zeros(values.size, dtype=float)
        layers: list[Layer] = []
        for gid in ids:
            frac = np.nan_to_num(np.asarray(self.fractions[gid], dtype=float), nan=0.0)
            lo = cumulative.copy()
            cumulative = cumulative + frac
            layers.append(
                Layer(
                    PlotKind.AREA,
                    {"x": values, "y": cumulative.copy(), "lo": lo, "hi": cumulative.copy()},
                    label=f"attractor {gid}",
                    style={"cmap": PALETTE},
                )
            )

        annotations = [
            Annotation(
                kind="vline",
                x=float(event["value"]),
                text=f"{event['kind']} (attractor {event['attractor']})",
                axis="x",
            )
            for event in self.tipping_points()
        ]
        meta = dict(self.meta) if self.meta else {}
        meta.update(palette=PALETTE, diverged_color=DIVERGED_COLOR, palette_index=swatch)
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"continuation over {self.param!r}",
            x=Axis(label=self.param or "parameter"),
            y=Axis(label="basin fraction", limits=(0.0, 1.0)),
            layers=layers,
            legend=Legend() if len(ids) > 1 else None,
            colorbar=Colorbar(label="attractor", cmap=PALETTE, discrete=True),
            annotations=annotations,
            meta=meta,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ContinuationResult(param={self.param!r}, "
            f"values=[{self.values[0]:.3g}..{self.values[-1]:.3g}], "
            f"n_attractors={len(self.fractions)})"
        )


def continuation(
    system: Any,
    param: str,
    values: Any,
    region: Box | Ball | Grid,
    *,
    n: int = 2000,
    resolution: int | tuple[int, ...] = 100,
    seed: int | None = 0,
    dt: float = 1.0,
    max_steps: int = 10000,
    min_fraction: float = 0.0,
    match_method: str = "centroid",
    match_threshold: float | None = None,
    **fsm: Any,
) -> ContinuationResult:
    r"""
    Track attractors and basin fractions as ``param`` sweeps ``values``.

    At each parameter value the attractors are found and their basin fractions
    estimated (:func:`~tsdynamics.analysis.basins.basin_fractions`); consecutive
    value's attractors are matched greedily by nearest state-space distance so a
    persisting attractor keeps its id and a vanishing one drops out (Datseris,
    Rossi & Wagemakers, 2023).

    Parameters
    ----------
    system : System
        A discrete map or continuous flow exposing ``param`` (``with_params``).
    param : str
        Name of the parameter to sweep.
    values : array-like
        Parameter values, in the order to walk them.
    region : Box, Ball, or Grid
        The region whose basin fractions are measured at each value.
    n : int, default 2000
        Initial conditions sampled per value.
    resolution : int or tuple of int, default 100
        Recurrence cells per axis (a Grid uses its own ``counts``).
    seed : int, optional
        Sampler seed (shared across values for a fair comparison).
    dt : float, default 1.0
        Integration step between cell checks for a flow.
    max_steps : int, default 10000
        Per-sample step cap.
    min_fraction : float, default 0.0
        Drop attractors whose basin fraction is below this at a given value before
        matching — a filter for the tiny spurious sets the recurrence finder can
        report near unstable equilibria.  ``0`` keeps everything.  The dropped
        basin mass is folded into the reported ``diverged`` share, so the tracked
        bands plus ``diverged`` still tile :math:`[0, 1]`.
    match_method : {"centroid", "hausdorff", "minimum"}, default "centroid"
        Set distance used to match attractors between values.
    match_threshold : float, optional
        Reject a match farther apart than this (so an attractor that jumps is not
        spuriously tied to a different one).  ``None`` matches the nearest
        regardless of distance.
    **fsm
        Finite-state-machine thresholds forwarded to the recurrence finder.

    Returns
    -------
    ContinuationResult

    Raises
    ------
    TypeError
        If ``system`` is a delay or stochastic system (unsupported by the
        recurrence finder).

    References
    ----------
    G. Datseris, K. L. Rossi and A. Wagemakers, "Framework for global stability
    analysis of dynamical systems", *Chaos* **33**, 073151 (2023).
    """
    _reject_unsupported(system, "continuation")
    values = np.asarray(values, dtype=float)
    fractions: dict[int, list[float]] = {}
    per_value: list[dict[int, Attractor]] = []
    diverged: list[float] = []

    prev_global: dict[int, Attractor] = {}
    next_global = 1

    for k, v in enumerate(values):
        sys_v = system.with_params(**{param: float(v)})
        bf = basin_fractions(
            sys_v, region, n=n, resolution=resolution, seed=seed, dt=dt, max_steps=max_steps, **fsm
        )
        local = {
            lid: att
            for lid, att in bf.attractors.attractors.items()
            if bf.fractions.get(lid, 0.0) >= min_fraction
        }  # local_id -> Attractor, tiny spurious sets dropped
        # Basin mass of the dropped (sub-``min_fraction``) sets is not tracked as a
        # band, so fold it into the "diverged / other" share — otherwise the
        # stacked bands plus diverged would sum below one (the un-tracked spurious
        # mass would silently vanish from the [0, 1] tiling).
        dropped = sum(
            float(bf.fractions.get(lid, 0.0))
            for lid in bf.attractors.attractors
            if lid not in local
        )

        local_to_global, next_global = _match(
            prev_global, local, next_global, method=match_method, threshold=match_threshold
        )

        value_attractors: dict[int, Attractor] = {}
        for lid, att in local.items():
            gid = local_to_global[lid]
            fractions.setdefault(gid, [float("nan")] * len(values))
            fractions[gid][k] = float(bf.fractions.get(lid, 0.0))
            value_attractors[gid] = att
        per_value.append(value_attractors)
        diverged.append(float(bf.diverged) + dropped)
        prev_global = value_attractors

    frac_arrays = {gid: np.asarray(arr) for gid, arr in fractions.items()}
    return ContinuationResult(
        param=param,
        values=values,
        fractions=frac_arrays,
        attractors=per_value,
        diverged=np.asarray(diverged),
        meta=AnalysisResult.build_meta(system, analysis="continuation", param=param),
    )


def _match(
    prev_global: dict[int, Attractor],
    current: dict[int, Attractor],
    next_global: int,
    *,
    method: str,
    threshold: float | None,
) -> tuple[dict[int, int], int]:
    """Greedily match current (local) attractors to the previous global ids."""
    if not prev_global:
        return {lid: next_global + i for i, lid in enumerate(current)}, next_global + len(current)

    pairs = []
    for gid, pa in prev_global.items():
        for lid, ca in current.items():
            d = set_distance(pa.points, ca.points, method=cast("_SetMethod", method))
            pairs.append((d, gid, lid))
    pairs.sort(key=lambda t: t[0])

    mapping: dict[int, int] = {}
    used_global: set[int] = set()
    used_local: set[int] = set()
    for d, gid, lid in pairs:
        if gid in used_global or lid in used_local:
            continue
        if threshold is not None and d > threshold:
            continue
        mapping[lid] = gid
        used_global.add(gid)
        used_local.add(lid)

    for lid in current:
        if lid not in mapping:
            mapping[lid] = next_global
            next_global += 1
    return mapping, next_global


def tipping_points(result: ContinuationResult, *, threshold: float = 0.0) -> CollectionResult:
    r"""
    Read tipping events off a continuation.

    A tipping event is a basin fraction crossing ``threshold`` between consecutive
    parameter values: a **disappear** event (fraction falls to/through it — an
    attractor and its basin annihilating) or an **appear** event (a new attractor
    gaining a basin).  With the default ``threshold=0`` only true
    appearance/annihilation is reported; raise it to flag basins shrinking past a
    safety margin.

    Parameters
    ----------
    result : ContinuationResult
        A continuation from :func:`continuation`.
    threshold : float, default 0.0
        Basin-fraction level whose crossing counts as a tipping event.

    Returns
    -------
    CollectionResult
        Behaves as a ``list of dict``; each event is
        ``{"value", "attractor", "kind", "before", "after"}`` with ``kind`` in
        ``{"appear", "disappear"}``, sorted by parameter value.
    """
    events: list[dict[str, Any]] = []
    vals = result.values
    for gid, frac in result.fractions.items():
        present = np.nan_to_num(np.asarray(frac, dtype=float), nan=0.0)
        for i in range(1, present.size):
            before, after = float(present[i - 1]), float(present[i])
            if before > threshold >= after:
                kind = "disappear"
            elif before <= threshold < after:
                kind = "appear"
            else:
                continue
            events.append(
                {
                    "value": float(vals[i]),
                    "attractor": int(gid),
                    "kind": kind,
                    "before": before,
                    "after": after,
                }
            )
    events.sort(key=lambda e: (e["value"], e["attractor"]))
    return CollectionResult(
        items=tuple(events),
        meta={"analysis": "tipping_points", "threshold": float(threshold)},
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
