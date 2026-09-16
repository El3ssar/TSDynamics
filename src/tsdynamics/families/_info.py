"""Identity: what a system *is*, as opposed to what you *do* to it.

Three things live here.

``Variables`` — the descriptor behind ``system.variables``.  Every system names
its own components in v6, including the five spatially-extended ones whose state
is a flattened field, so ``traj["x"]`` and a labelled axis are never a guess.
It is a *non-data* descriptor so the resolution is paid **lazily and once**:
materialising GrayScott's 4,608 names costs ~770 us, roughly 7x the constructor,
on a class instantiated once per parameter-sweep value.  Read off the class it
still answers with the **declared** tuple, which is what the nine class-level
readers deferred to v6.1 expect.

``SystemInfo`` — the frozen record ``system.info`` prints.  It absorbs
``reference``, ``doi``, ``known_lyapunov``, ``field_labels`` and ``default_ic``
off the tab surface: they are facts about the system, and a fact belongs in the
record you print, not in the list of verbs you can call.

``family_of`` — the ``"ode" | "dde" | "map" | "sde"`` word that replaced
``is_discrete``.  A boolean could not tell a delay system from a stochastic one,
which is why every caller that needed to had to sniff the class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["SystemInfo", "Variables", "family_of", "resolve_variables"]


def family_of(system: Any) -> str:
    """Return ``"ode" | "dde" | "map" | "sde"`` for *system*.

    Walks the MRO, exactly as :func:`tsdynamics.registry.register_class` does,
    so a wrapper delegating to an inner system and a catalogue class answer the
    same way.
    """
    declared = getattr(type(system), "_family", None)
    if isinstance(declared, str):
        return declared
    from tsdynamics.families.continuous import ContinuousSystem
    from tsdynamics.families.delay import DelaySystem
    from tsdynamics.families.discrete import DiscreteMap
    from tsdynamics.families.stochastic import StochasticSystem

    for base, name in (
        (DiscreteMap, "map"),
        (DelaySystem, "dde"),
        (StochasticSystem, "sde"),
        (ContinuousSystem, "ode"),
    ):
        if isinstance(system, base):
            return name
    return "map" if getattr(system, "_is_discrete", False) else "ode"


def resolve_variables(system: Any) -> tuple[str, ...]:
    """Resolve *system*'s component names — the four ordered rules of the v6 contract.

    1. a declared ``variables`` of the right length wins;
    2. a **field** system (``_field_shape``) →  ``f"{block}{cell}"``, block-major
       and row-major inside a block — exactly the layout the kernels pack;
    3. a **repeated-unit** system (``_unit_variables``) → ``f"{name}{unit}"``,
       unit-major;
    4. fallback ``y0 ... y{dim-1}`` — the names ``to_frame()`` already invents.

    Returns
    -------
    tuple of str
        Always exactly ``system.dim`` names.  Never ``None``.
    """
    dim = int(system.dim)
    declared = getattr(type(system), "_declared_variables", None)
    if declared is not None:
        names = tuple(str(n) for n in declared)
        if len(names) == dim:
            return names

    shape = getattr(system, "_field_shape", None)
    if shape:
        cells = 1
        for n in shape:
            cells *= int(n)
        blocks = getattr(type(system), "_field_labels", None) or ("u",)
        if cells > 0 and cells * len(blocks) == dim:
            return tuple(f"{b}{i}" for b in blocks for i in range(cells))

    unit = getattr(type(system), "_unit_variables", None)
    if unit:
        width = len(unit)
        if width > 0 and dim % width == 0:
            return tuple(f"{n}{u}" for u in range(dim // width) for n in unit)

    return tuple(f"y{i}" for i in range(dim))


class Variables:
    """Non-data descriptor: declared names off the class, resolved names off an instance.

    Non-data (only ``__get__``) so the first instance read writes the resolved
    tuple straight into ``obj.__dict__`` and every later read is a plain dict
    hit — the laziness that keeps a 4,608-component field system cheap to
    construct.
    """

    __slots__ = ()

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            # ``type(sys).variables`` — the DECLARED tuple (or None), which is
            # what the class-level readers deferred to v6.1 still expect.
            return getattr(objtype, "_declared_variables", None)
        names = resolve_variables(obj)
        obj.__dict__["variables"] = names
        return names


def _render_equations(system: Any, *, limit: int = 12) -> list[str]:
    """Render ``dx/dt = ...`` (or ``x -> ...``) lines for the info record.

    Best-effort and never fatal: an un-lowerable kernel, a missing symbolic
    frontend or a huge field system all degrade to a one-line summary rather
    than making ``system.info`` raise.
    """
    dim = int(system.dim)
    names = tuple(system.variables)
    if dim > limit:
        return [f"<{dim} equations — too many to print>"]
    kernel = getattr(system, "_equations", None) or getattr(system, "_drift", None)
    is_map = family_of(system) == "map"
    try:
        if is_map:
            import numpy as np
            import symengine as se

            syms = [se.Symbol(n) for n in names]
            exprs = list(
                type(system)._step(np.array(syms, dtype=object), *system.params.as_tuple())
            )
            arrow = "{lhs}' = {rhs}"
        else:
            if kernel is None:
                return []
            import symengine as se

            from tsdynamics.engine.symbols import state_time_symbols

            u, t = state_time_symbols()
            # ODE / SDE kernels take their parameters as KEYWORDS (the catalogue
            # convention), maps positionally.  Pass SYMBOLS, so the printed
            # equations read like the paper rather than like one instance.
            args: dict[str, Any] = {k: se.Symbol(k) for k in system.params}
            args.update(getattr(system, "_structural_vals", dict)())
            exprs = list(kernel(u, t, **args))
            arrow = "d{lhs}/dt = {rhs}"
    except Exception:  # pragma: no cover - defensive: info must never raise
        return []
    lines = []
    for i, expr in enumerate(exprs):
        text = str(expr)
        for j in range(dim - 1, -1, -1):
            text = text.replace(f"y({j})", names[j]).replace(f"y_{j}", names[j])
        lines.append(arrow.format(lhs=names[i], rhs=text))
    return lines


@dataclass(frozen=True)
class SystemInfo:
    """Everything true about one system, in one printable record.

    Received, never constructed — ``system.info`` builds it.  ``print(info)``
    (and a bare ``info`` in a REPL) is the whole point; the fields are there for
    the tooling that used to reach for the ClassVars directly.
    """

    name: str
    family: str
    dim: int
    qualname: str
    equations: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    variables: tuple[str, ...] = ()
    reference: str | None = None
    doi: str | None = None
    default_ic: Any | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    known_lyapunov: dict[str, Any] | None = None
    field_shape: tuple[int, ...] | None = None
    field_labels: tuple[str, ...] | None = None

    _FAMILY_WORD = {
        "ode": "continuous flow",
        "dde": "delay system",
        "map": "discrete map",
        "sde": "stochastic flow",
    }

    @classmethod
    def of(cls, system: Any) -> SystemInfo:
        """Build the record for *system*."""
        t = type(system)
        defaults: dict[str, Any] = {}
        family = family_of(system)
        wanted = (
            ("solver", "_default_method"),
            ("rtol", "_default_rtol"),
            ("atol", "_default_atol"),
            ("dt", "_default_dt"),
            ("backend", "_default_backend"),
        )
        for key, attr in wanted:
            if family == "map" and key in ("solver", "rtol", "atol", "dt"):
                continue  # a map has no solver, no tolerance and no output grid
            if family == "sde" and key in ("rtol", "atol"):
                continue  # a fixed-step Ito scheme has no embedded error estimate
            value = getattr(system, attr, None)
            if value is None and key in ("rtol", "atol"):
                from tsdynamics.utils.tolerances import DEFAULT_ATOL, DEFAULT_RTOL

                value = DEFAULT_RTOL if key == "rtol" else DEFAULT_ATOL
            if value is not None:
                defaults[key] = str(value).lower() if key == "solver" else value
        return cls(
            name=t.__name__,
            family=family_of(system),
            dim=int(system.dim),
            qualname=f"{t.__module__}.{t.__qualname__}",
            equations=tuple(_render_equations(system)),
            parameters=dict(system.params),
            variables=tuple(system.variables),
            reference=getattr(t, "_reference", None),
            doi=getattr(t, "_doi", None),
            default_ic=getattr(t, "_default_ic", None),
            defaults=defaults,
            known_lyapunov=getattr(t, "_known_lyapunov", None),
            field_shape=getattr(system, "_field_shape", None),
            field_labels=getattr(t, "_field_labels", None),
        )

    def _analysis_count(self) -> int | None:
        """How many registered analyses take a subject of **this** family.

        The body used to be ``len(records)`` — the whole registry — so the first
        number a newcomer sees about analyses said ``50`` on a Hénon whose
        ``ts.analysis.find(henon)`` returns 14.  A record card that contradicts
        the call it recommends teaches the wrong thing twice.
        """
        try:
            from tsdynamics.analysis import _discovery

            token = "map" if self.family == "map" else "flow"
            count = _discovery.find_count(token)
        except Exception:  # pragma: no cover - defensive
            return None
        return count or None

    def __call__(self) -> SystemInfo:
        """Return ``self``, so ``system.info()`` works as well as ``system.info``.

        ``info`` reads as a verb among eighteen verbs, so ``lor.info()`` is the
        first thing a newcomer types — and ``TypeError: 'SystemInfo' object is
        not callable`` neither says *drop the parentheses* nor shows the record
        they were after.  This is not a second spelling of a concept: it is the
        same object either way.
        """
        return self

    def __repr__(self) -> str:
        word = self._FAMILY_WORD.get(self.family, self.family)
        head = f"{self.name} — {self.dim}-D {word}"
        pad = max(2, 64 - len(head))
        lines = [f"{head}{' ' * pad}{self.qualname}"]
        for i, eq in enumerate(self.equations):
            label = "  equations   " if i == 0 else "              "
            lines.append(f"{label}{eq}")
        if self.parameters:
            shown = "   ".join(
                f"{k} = {v:g}" if isinstance(v, float) else f"{k} = {v}"
                for k, v in self.parameters.items()
            )
            lines.append(f"  parameters  {shown}")
        if self.variables:
            names = ", ".join(self.variables[:8])
            if len(self.variables) > 8:
                names += f", ... ({len(self.variables)} total)"
            lines.append(f"  variables   {names}")
        if self.reference:
            doi = f"   doi:{self.doi}" if self.doi else ""
            lines.append(f"  reference   {self.reference}{doi}")
        if self.defaults:
            shown = "  ".join(f"{k}={v}" for k, v in self.defaults.items())
            lines.append(f"  defaults    {shown}")
        n = self._analysis_count()
        if n is not None:
            lines.append(f"  analyses    ts.analysis.find(system) → {n}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()
