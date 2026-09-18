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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ._hidden import hide

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


#: Where a resolved (or user-assigned) name tuple is memoised on an instance.
#: Deliberately **not** ``"variables"`` itself: :class:`Variables` is a *data*
#: descriptor, so it wins over the instance ``__dict__`` and a value parked
#: under its own name would never be read again.
_VARIABLES_CACHE = "_variables_resolved"


class Variables:
    """Data descriptor: declared names off the class, resolved names off an instance.

    Reading is still one dict hit after the first time — the resolved tuple is
    memoised under :data:`_VARIABLES_CACHE` — which is the laziness that keeps a
    4,608-component field system cheap to construct.  It used to be a *non-data*
    descriptor (``__get__`` only), memoising under its own name so that Python
    stopped consulting the descriptor at all.

    It grew a ``__set__`` because that same shortcut made ``system.variables``
    an **unguarded** write (``CONTRACT.md`` §11.3 T2, §11.6 defect 1).  Renaming
    a system's components is a legitimate customization and stays legitimate —
    but it has exactly one correct arity, and a wrong one used to be accepted
    in silence and then surface far away, as a raw ``IndexError: tuple index out
    of range`` from inside ``system.info``'s equation renderer.
    """

    __slots__ = ()

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            # ``type(sys).variables`` — the DECLARED tuple (or None), which is
            # what the class-level readers deferred to v6.1 still expect.
            return getattr(objtype, "_declared_variables", None)
        cached = obj.__dict__.get(_VARIABLES_CACHE)
        if cached is not None:
            return cached
        names = resolve_variables(obj)
        obj.__dict__[_VARIABLES_CACHE] = names
        return names

    def __set__(self, obj: Any, value: Any) -> None:
        """Rename this system's components — one name per state component.

        Raises
        ------
        InvalidInputError
            If *value* is not a sequence of strings, or does not name every
            component exactly once.
        """
        from tsdynamics.errors import InvalidInputError, remedy

        dim = int(obj.dim)
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise InvalidInputError(
                f"{type(obj).__name__}.variables must be a sequence of "
                f"{dim} component names, got {type(value).__name__}."
                + remedy(f"system.variables = {_example_names(dim)!r}")
            )
        names = tuple(str(n) for n in value)
        if len(names) != dim:
            raise InvalidInputError(
                f"{type(obj).__name__}.variables names {len(names)} component"
                f"{'' if len(names) == 1 else 's'} {names}, but this system has "
                f"{dim}. Every component is named, or none is — a partial "
                f"listing silently relabels the wrong columns."
                + remedy(f"system.variables = {_example_names(dim)!r}")
            )
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise InvalidInputError(
                f"{type(obj).__name__}.variables repeats {dupes}; a name selects "
                f"a column, so two columns cannot share one."
                + remedy(f"system.variables = {_example_names(dim)!r}")
            )
        obj.__dict__[_VARIABLES_CACHE] = names


def _example_names(dim: int) -> tuple[str, ...]:
    """Return a runnable ``variables`` tuple of the right width for *dim*."""
    if dim <= 3:
        return ("x", "y", "z")[:dim]
    return tuple(f"x{i}" for i in range(dim))


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
            # An SDE is a DRIFT plus a DIFFUSION, and the card used to print only
            # the drift: measured, ``OrnsteinUhlenbeck.info`` rendered
            # ``dx/dt = (mu - x)*theta`` and listed ``sigma = 0.3`` under
            # parameters with no equation using it — so on the one family where
            # the noise IS the model, half the model was invisible and the
            # parameter that carries it looked unused.
            noise = getattr(system, "_diffusion", None)
            if noise is not None:
                coefficients = list(noise(u, t, **args))
                return _render_sde(names, exprs, coefficients, dim)
    except Exception:  # pragma: no cover - defensive: info must never raise
        return []
    return [
        arrow.format(lhs=names[i], rhs=_name_state(str(expr), names, dim))
        for i, expr in enumerate(exprs)
    ]


def _name_state(text: str, names: tuple[str, ...], dim: int) -> str:
    """Replace the engine's ``y(i)`` accessors with the declared component names."""
    for j in range(dim - 1, -1, -1):
        text = text.replace(f"y({j})", names[j]).replace(f"y_{j}", names[j])
    return text


def _render_sde(
    names: tuple[str, ...], drift: list[Any], diffusion: list[Any], dim: int
) -> list[str]:
    """Render an SDE as ``dx = f dt + g dW`` — both halves, on one line each.

    Itô differential form rather than ``dx/dt``, because a stochastic
    differential equation has no derivative: writing it as one would be the
    second thing wrong with the old card.
    """
    lines = []
    for i, (f, g) in enumerate(zip(drift, diffusion, strict=False)):
        rhs = _name_state(str(f), names, dim)
        coeff = _name_state(str(g), names, dim)
        noise = f"dW_{i}" if len(drift) > 1 else "dW"
        term = f"{noise}" if coeff == "1" else f"({coeff}) {noise}"
        lines.append(f"d{names[i]} = ({rhs}) dt + {term}")
    return lines


@dataclass(frozen=True)
@hide("of")
class SystemInfo:
    """Everything true about one system, in one printable record.

    Received, never constructed — ``system.info`` builds it.  ``print(info)``
    (and a bare ``info`` in a REPL) is the whole point; the fields are there for
    the tooling that used to reach for the ClassVars directly.

    ``of`` — the builder ``system.info`` calls — is withheld from ``dir()`` for
    that same reason (``CONTRACT.md`` §11): it is the one member of this record
    that is not a *fact about the system*, and a user holding an ``info`` has
    already built it.  It remains bound and callable.
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


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
