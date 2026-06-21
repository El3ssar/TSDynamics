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
from .attractors import Attractor
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
        Diverged fraction at each value.
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
        report near unstable equilibria.  ``0`` keeps everything.
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

    References
    ----------
    G. Datseris, K. L. Rossi and A. Wagemakers, "Framework for global stability
    analysis of dynamical systems", *Chaos* **33**, 073151 (2023).
    """
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
        diverged.append(float(bf.diverged))
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
    list of dict
        Each event is ``{"value", "attractor", "kind", "before", "after"}`` with
        ``kind`` in ``{"appear", "disappear"}``, sorted by parameter value.
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
