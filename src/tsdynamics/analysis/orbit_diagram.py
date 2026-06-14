"""Orbit diagrams — asymptotic states swept across a parameter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["OrbitDiagram", "orbit_diagram"]


@dataclass
class OrbitDiagram:
    """
    Result of :func:`orbit_diagram`.

    Iterate to get ``(value, points)`` pairs, or use :meth:`flat` for the
    scatter-ready arrays.
    """

    param: str
    values: np.ndarray  # (V,)
    points: list[np.ndarray]  # per value: (n, k) recorded components
    components: tuple[int, ...]
    meta: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(zip(self.values, self.points, strict=True))

    def __len__(self) -> int:
        return len(self.values)

    def flat(self, component: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Flatten to scatter-plot arrays ``(x, y)``.

        ``x`` repeats each parameter value once per recorded point; ``y`` is
        the chosen recorded component.
        """
        x = np.concatenate(
            [np.full(p.shape[0], v) for v, p in zip(self.values, self.points, strict=True)]
        )
        y = np.concatenate([p[:, component] for p in self.points])
        return x, y

    def __repr__(self) -> str:
        return (
            f"OrbitDiagram({self.param!r}, {len(self.values)} values, "
            f"{self.points[0].shape[0] if self.points else 0} points/value)"
        )


def orbit_diagram(
    sys: Any,
    param: str,
    values: Any,
    *,
    n: int = 200,
    transient: int = 500,
    carry_state: bool = True,
    components: int | str | tuple = 0,
    ic: Any | None = None,
) -> OrbitDiagram:
    """
    Sweep a parameter and record the asymptotic orbit at each value.

    Works on anything discrete: a :class:`~tsdynamics.families.DiscreteMap`
    directly, or a flow wrapped in a
    :class:`~tsdynamics.derived.PoincareMap` /
    :class:`~tsdynamics.derived.StroboscopicMap` — in which case this *is*
    the bifurcation diagram of the flow.  ODE parameter changes reuse the
    compiled module (control parameters), so flow sweeps stay cheap; DDE
    sweeps recompile per value (their structure depends on all parameters).

    Parameters
    ----------
    sys : System (discrete)
        The system to sweep.  Never mutated — each value gets a fresh
        ``with_params`` copy.
    param : str
        Parameter name to sweep.
    values : iterable of float
        Parameter values, in sweep order.
    n : int
        Points recorded per parameter value.
    transient : int
        Steps discarded before recording, at every value.
    carry_state : bool
        Start each value from the previous value's final state (follows the
        attractor branch; the classic way to draw clean diagrams).  When
        False, every value starts from ``ic`` / the system default.
    components : int, str, or tuple
        Which state components to record (names allowed when the system
        declares ``variables``).
    ic : array-like, optional
        Initial state for the first value (and every value when
        ``carry_state=False``).

    Examples
    --------
    >>> od = orbit_diagram(Logistic(), "r", np.linspace(2.5, 4.0, 600), n=120)
    >>> x, y = od.flat()
    >>> # bifurcation diagram of a flow:
    >>> od = orbit_diagram(PoincareMap(Rossler(), (1, 0.0)), "c", np.linspace(2, 6, 80))
    """
    if not sys.is_discrete:
        raise TypeError(
            "orbit_diagram needs a discrete-time view: a DiscreteMap, or a flow wrapped "
            "in PoincareMap / StroboscopicMap."
        )

    comp = (components,) if isinstance(components, int | str) else tuple(components)
    # Resolve names via the *instance* (not ``type(sys)``): a derived wrapper
    # exposes ``variables`` as a property, so ``type(sys).variables`` returns the
    # descriptor object (truthy) and short-circuits — breaking named components
    # over a PoincareMap/StroboscopicMap.  Instance lookup returns the ClassVar
    # for families and the resolved names for wrappers alike.
    names = getattr(sys, "variables", None)
    idx: list[int] = []
    for c in comp:
        if isinstance(c, str):
            if names is None:
                raise ValueError("named components need the system to declare `variables`")
            idx.append(names.index(c))
        else:
            idx.append(int(c))

    import warnings

    values_arr = np.asarray(list(values), dtype=float)
    points: list[np.ndarray] = []
    state: np.ndarray | None = None

    for v in values_arr:
        current = sys.with_params(**{param: v})
        start = state if (carry_state and state is not None) else ic
        try:
            current.reinit(start)
            for _ in range(transient):
                current.step()
            rec = np.empty((n, len(idx)))
            for i in range(n):
                u = current.step()
                rec[i] = u[idx]
        except RuntimeError as exc:
            # One divergent value must not discard the whole sweep: record an
            # empty point set and restart the next value from `ic`.
            warnings.warn(
                f"orbit_diagram: {param}={v:g} diverged ({exc}); recording an "
                f"empty set for this value.",
                RuntimeWarning,
                stacklevel=2,
            )
            points.append(np.empty((0, len(idx))))
            state = None
            continue
        points.append(rec)
        if carry_state:
            state = current.state()

    meta = {
        "system": type(sys).__name__,
        "param": param,
        "n": n,
        "transient": transient,
        "carry_state": carry_state,
        "components": tuple(idx),
    }
    return OrbitDiagram(
        param=param, values=values_arr, points=points, components=tuple(idx), meta=meta
    )
