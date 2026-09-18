r"""Discovery for the analysis layer — the registry, the search, and the teaching.

Ruling **A2** took every analysis off every object: there is no ``lor.lyapunov_spectrum()``
and no ``traj.rqa()``, because a second spelling of one function is the
silent-wrong-answer defect (``sys.chaos.zero_one()`` reported **K = -0.026** for
Lorenz where the free function reports **0.999**, because the convenience hid a
sampling choice).  What replaces those bound methods is *discovery*, and it has
to be genuinely good or the capability is unreachable:

``ts.analysis.<TAB>``
    50 analyses, flat and sorted, generated from :data:`tsdynamics.registry.analyses`.
``ts.analysis.__doc__``
    the same 50, grouped by **what you are holding** (:func:`grouped_map`).
:func:`find`
    ``find(traj)`` — what can I measure on THIS?  ``find("chaotic")`` — who
    answers THIS question?
:func:`teach` / :func:`wrong_subject` / :func:`attribute_error`
    one builder for every "that moved" message, so the object door and the
    free-function door cannot drift apart.

Everything here reads :data:`tsdynamics.registry.analyses`; nothing is
hand-listed.
"""

from __future__ import annotations

import difflib
import re
import textwrap
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__: list[str] = []

# ── the vocabulary ───────────────────────────────────────────────────────────

#: The canonical display order of the analysis areas.  One order serves all
#: three subject groups: filtered to a group it reproduces that group's reading
#: order (a system's analyses run lyapunov -> chaos -> orbits -> fixed points ->
#: basins -> fields; a trajectory's run lyapunov -> dimensions -> recurrence ->
#: embedding -> ...).
AREAS: tuple[str, ...] = (
    "lyapunov",
    "chaos",
    "orbits",
    "dimensions",
    "recurrence",
    "embedding",
    "fixedpoints",
    "basins",
    "fields",
    "sampling",
    "geometry",
)

#: How each area is spelled in the map.  Only one differs from its own key.
AREA_LABEL: dict[str, str] = {"fixedpoints": "fixed points"}

#: Subject tokens that stand for a set of narrower ones.  ``system`` is the one
#: an author writes; it expands so that ``find(henon)`` can drop the seven
#: analyses that need a vector field (a map has none).
SUBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "system": ("flow", "map"),
    "data": ("trajectory", "array"),
    "series": ("trajectory", "array"),
}

#: A system is one of these two.
SYSTEM_TOKENS = frozenset({"flow", "map"})

#: Measured data is one of these two.
DATA_TOKENS = frozenset({"trajectory", "array"})

#: The three groups, in order, with the heading each prints.
GROUPS: tuple[tuple[str, str, str], ...] = (
    ("system", "You have a SYSTEM — you have the equations", "ts.analysis.<name>(system, ...)"),
    ("data", "You have a TRAJECTORY or a bare array", "ts.analysis.<name>(data, ...)"),
    ("result", "You have a RESULT another analysis returned", "ts.analysis.<name>(result, ...)"),
)


def expand_subjects(subjects: Iterable[str]) -> tuple[str, ...]:
    """Expand the ``subjects=`` tokens an author declared into narrow ones."""
    out: list[str] = []
    for token in subjects:
        for narrow in SUBJECT_ALIASES.get(token, (token,)):
            if narrow not in out:
                out.append(narrow)
    return tuple(out)


def group_of(subjects: Sequence[str]) -> str:
    """Return which of the three groups an analysis with these subjects is in."""
    if SYSTEM_TOKENS.intersection(subjects):
        return "system"
    if DATA_TOKENS.intersection(subjects):
        return "data"
    return "result"


# ── summaries, derived from the docstring ────────────────────────────────────

_ROLE = re.compile(r":[a-zA-Z]+:`(~?)(?:[^`<]*<)?([^`<>]+)>?`")
#: A LaTeX control word inside ``:math:`` (or a bare docstring): the backslash is
#: markup, the word is the symbol a reader recognises.  ``\tau`` -> ``tau``.
_TEX = re.compile(r"\\([A-Za-z]+)")
#: The rendered width the map is gated at.  Any longer and the two-column
#: listing wraps, which is what makes a generated map unreadable.
SUMMARY_WIDTH = 72


def summarise(doc: str | None) -> str:
    """Render the one-line summary of an analysis from its own docstring.

    First sentence of the first paragraph, with RST roles unwrapped
    (``:math:`D_2``` -> ``D_2``), literals unquoted and the docstring's ``--``
    turned back into ``-``.  Never a second hand-maintained string: a summary
    that can drift from the docstring is a summary nobody updates.

    A sentence longer than :data:`SUMMARY_WIDTH` is shortened by dropping its
    trailing subordinate clause (after the last ``:`` or em dash), then by
    truncating on a word boundary.
    """
    if not doc:
        return ""
    paragraph: list[str] = []
    for raw in doc.strip().splitlines():
        line = raw.strip()
        if not line:
            if paragraph:
                break
            continue
        paragraph.append(line)
    text = " ".join(paragraph)
    # ``:func:`~pkg.mod.name``` renders as ``name`` — the tilde is what says so.
    text = _ROLE.sub(lambda m: m.group(2).rsplit(".", 1)[-1] if m.group(1) else m.group(2), text)
    text = text.replace("``", "").replace("`", "").replace("--", "-").replace("\\ ", " ")
    text = _TEX.sub(r"\1", text).replace("*", "")
    text = re.sub(r"\s+", " ", text).strip()
    # First sentence.  ". " is the separator; a trailing "." is kept.
    head = text.split(". ")[0].rstrip(".")
    for cut in (" — ", " - ", ": "):
        if len(head) + 1 > SUMMARY_WIDTH and cut in head:
            head = head.split(cut)[0].rstrip(" ,;:-")
    if len(head) + 1 > SUMMARY_WIDTH:
        words = head[: SUMMARY_WIDTH - 2].rsplit(" ", 1)[0]
        return f"{words}…"
    return f"{head}."


# ── the search ───────────────────────────────────────────────────────────────

#: Words a user brings that the registry's own vocabulary never uses, mapped to
#: the terms it does.  ``find`` is advertised as "ask in your own words", and an
#: engineer's words are not a dynamicist's: measured, ``find("robust")`` and
#: ``find("safety margin")`` both returned nothing while ``resilience``'s own
#: summary line reads *"Minimal-fatal-shock resilience"*, and ``find("will it
#: tip over")`` returned ``correlation_sum`` and ``dimension_spectrum``.
#:
#: Each row is a word a reader plausibly types -> the words the registry knows.
#: Kept deliberately small and literal: this is a thesaurus, not a ranker.
#: Widened in v6 round 6 with the vocabulary of the FIELD as well as of
#: engineering, after a sweep of 128 plausible questions: 20 of them returned
#: nothing while the analysis that answers them was registered the whole time
#: (``"critical transition"``, ``"early warning"``, ``"hysteresis"``,
#: ``"sensitive dependence"``, ``"bistability"``, ``"intermittency"``,
#: ``"quasiperiodic"``, ``"separatrix"``, ``"reconstruct"``, ``"failure"`` …).
#: A query with genuinely no answer here (``"synchronisation"``,
#: ``"resonance"``, ``"noise"``) is deliberately **still** empty — inventing a
#: row for it would answer a question the library cannot.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "alternative": ("attractors", "basins", "basin_fractions"),
    "annihilate": ("tipping_points",),
    "attractor": ("attractors", "basins"),
    "bifurcate": ("orbit_diagram", "continuation"),
    "bifurcation": ("orbit_diagram", "continuation"),
    "bistability": ("attractors", "basins", "basin_fractions"),
    "bistable": ("attractors", "basins", "basin_fractions"),
    "buffer": ("resilience",),
    "catastrophe": ("tipping_points", "continuation"),
    "coexisting": ("attractors", "basins", "basin_fractions"),
    "collapse": ("tipping_points", "resilience"),
    "complexity": ("rqa", "expansion_entropy", "correlation_dimension"),
    "crisis": ("tipping_points", "continuation"),
    "critical": ("tipping_points", "continuation"),
    "delay": ("embed", "optimal_delay", "mutual_information"),
    "disturbance": ("resilience", "basin_fractions"),
    "divide": ("basins", "uncertainty_exponent"),
    "ergodic": ("invariant_density",),
    "fail": ("resilience", "tipping_points"),
    "failure": ("resilience", "tipping_points"),
    "fold": ("tipping_points", "continuation"),
    "forecast": ("lyapunov_spectrum", "lyapunov_from_data"),
    "fragile": ("resilience",),
    "headroom": ("resilience",),
    "horizon": ("lyapunov_spectrum", "lyapunov_from_data"),
    "hysteresis": ("continuation", "tipping_points"),
    "intermingled": ("wada_property", "basin_entropy"),
    "intermittency": ("rqa", "windowed_rqa", "recurrence_matrix"),
    "intermittent": ("rqa", "windowed_rqa", "recurrence_matrix"),
    "irreversible": ("tipping_points", "continuation"),
    "laminar": ("rqa", "windowed_rqa"),
    "margin": ("resilience",),
    "mixing": ("lyapunov_spectrum", "expansion_entropy"),
    "multistability": ("attractors", "basins", "basin_fractions"),
    "multistable": ("attractors", "basins", "basin_fractions"),
    "perturbation": ("resilience", "basin_fractions"),
    "predictable": ("uncertainty_exponent", "basin_entropy"),
    "quasiperiodic": ("gali", "lyapunov_spectrum"),
    "reconstruct": ("embed", "optimal_delay", "embedding_dimension"),
    "reconstruction": ("embed", "optimal_delay", "embedding_dimension"),
    "regime": ("tipping_points", "continuation"),
    "resilient": ("resilience", "basin_fractions"),
    "riddled": ("wada_property", "basin_entropy", "uncertainty_exponent"),
    "robust": ("resilience", "basin_fractions"),
    "robustness": ("resilience", "basin_fractions"),
    "safety": ("resilience",),
    "sensitive": ("lyapunov_spectrum", "lyapunov_from_data", "uncertainty_exponent"),
    "sensitivity": ("lyapunov_spectrum", "lyapunov_from_data", "uncertainty_exponent"),
    "separatrix": ("basins", "basin_entropy", "uncertainty_exponent"),
    "shock": ("resilience",),
    "stability": ("fixed_points", "basin_fractions", "resilience"),
    "survive": ("resilience",),
    "takens": ("embed", "optimal_delay", "embedding_dimension"),
    "threshold": ("tipping_points", "continuation"),
    "tip": ("tipping_points", "resilience"),
    "tipping": ("tipping_points",),
    "tolerance": ("resilience", "basin_fractions"),
    "torus": ("gali", "lyapunov_spectrum"),
    "unpredictable": ("uncertainty_exponent", "basin_entropy"),
    "warning": ("tipping_points", "continuation"),
    "watershed": ("basins", "uncertainty_exponent"),
    "withstand": ("resilience", "basin_fractions"),
}

_SPLIT = re.compile(r"[^0-9a-z]+")
#: Terms shorter than this cannot match: without the floor, "is" in "is this
#: chaotic?" matches ``set_dIStance`` and the answer is noise.
_MIN_TERM = 3


def _terms(query: str) -> list[str]:
    return [t for t in _SPLIT.split(query.casefold()) if t]


def score(entry: Any, term: str) -> float:
    """Score one registry entry against one search term.

    First rule that fires wins; the weights are the contract's (§5.3).
    """
    name = entry.name.casefold()
    tokens = _SPLIT.split(name)
    if term in tokens:
        return 4.0
    if len(term) < _MIN_TERM:
        return 0.0
    flat = name.replace("_", "")
    if name.startswith(term) or (tokens and term.startswith(tokens[0])):
        return 3.0
    if term in flat:
        return 2.5
    meta = entry.metadata
    if term in _SPLIT.split(str(meta.get("keywords", "")).casefold()):
        return 2.2
    if term == str(meta.get("area", "")).casefold():
        return 2.0
    if term in _SPLIT.split(str(meta.get("summary", "")).casefold()):
        return 1.5
    return 0.0


def search(entries: Sequence[Any], query: str) -> list[Any]:
    """Rank ``entries`` against a free-text ``query``, cutting at the relevance cliff.

    A term the registry's vocabulary does not use is first looked up in
    :data:`SYNONYMS`, so a question asked in the reader's words still lands.
    """
    terms = _terms(query)
    if not terms:
        return list(entries)
    named = {e.name for e in entries}
    wanted = {n for t in terms for n in SYNONYMS.get(t, ()) if n in named}
    scored = [
        (sum(score(e, t) for t in terms) + (4.0 if e.name in wanted else 0.0), e.name, e)
        for e in entries
    ]
    scored = [row for row in scored if row[0] > 0.0]
    if not scored:
        return []
    top = max(row[0] for row in scored)
    kept = [row for row in scored if row[0] >= 0.5 * top]
    kept.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in kept]


def nothing_matched(query: str, total: int) -> str:
    """Explain an empty :func:`~tsdynamics.analysis.find`, instead of shrugging.

    Two kinds of miss, and they need opposite answers.  A word from the layer v6
    **removed** (surrogates, entropy estimators, the signal-transform toolbox)
    must say so, or "nothing matches" reads as *this library cannot do that* and
    a reader goes off to reimplement an FT surrogate test by hand.  Anything else
    gets the areas to browse — a bare ``[]`` is the one repr in the library that
    answers a question with silence.
    """
    from tsdynamics._redirects import OUT_OF_SCOPE_SEARCH_TERMS, SCOPE_SURGERY_REMEDY

    for term in _terms(query):
        what = OUT_OF_SCOPE_SEARCH_TERMS.get(term)
        if what is None:
            continue
        lines = "\n".join(f"    {line}" for line in SCOPE_SURGERY_REMEDY)
        return (
            f"nothing matches {query!r} — {what} were REMOVED from this library. TSDynamics is a "
            f"dynamical systems library: phase-space methods stay, generic series "
            f"statistics go, and that layer moved to a companion time-series package."
            f"\nWhat stayed:\n{lines}"
        )
    return (
        f"nothing matches {query!r}.  The {total} analyses are grouped by what you hold:"
        f"\n    print(ts.analysis.__doc__)        # the map"
        f"\n    ts.analysis.find(subject)         # what can I measure on THIS?"
        f"\n    areas: {', '.join(AREAS)}"
    )


def subject_tokens(what: Any) -> tuple[str, ...] | None:
    """Classify a subject into the tokens an analysis may declare, or ``None``.

    Accepts an instance *or* a class, so ``find(Lorenz)`` and ``find(Lorenz())``
    answer the same.
    """
    from tsdynamics.analysis._common import is_data, is_system

    subject = what
    if isinstance(what, type):
        try:  # a class: read the family off the class where the family declares it
            from tsdynamics.data import Trajectory
            from tsdynamics.families import (  # noqa: F401  # noqa: F401
                ContinuousSystem,
                DelaySystem,
                DiscreteMap,
                StochasticSystem,
                SystemBase,
            )
        except Exception:  # pragma: no cover - defensive
            return None
        if issubclass(what, SystemBase):
            return ("map",) if issubclass(what, DiscreteMap) else ("flow",)
        if issubclass(what, Trajectory):
            return ("trajectory", "array")
        import numpy as np

        if issubclass(what, np.ndarray):
            return ("array",)
        return _result_token_of_class(what)
    if is_system(subject):
        family = getattr(subject, "family", None)
        discrete = family == "map" or bool(getattr(subject, "_is_discrete", False))
        return ("map",) if discrete else ("flow",)
    if is_data(subject):
        return ("trajectory", "array")
    return _result_token_of_class(type(subject))


def _result_token_of_class(cls: type) -> tuple[str, ...] | None:
    """Return a result class's own name and its result-class bases, or ``None``."""
    from tsdynamics.analysis._result_base import AnalysisResult

    if not issubclass(cls, AnalysisResult):
        return None
    return tuple(base.__name__ for base in cls.__mro__ if issubclass(base, AnalysisResult))


# ── the rendered listings ────────────────────────────────────────────────────


def groups_of(subjects: Sequence[str]) -> tuple[str, ...]:
    """Return **every** group an analysis with these subjects belongs to.

    :func:`group_of` picks one — it answers "where does this belong *first*" —
    and a few analyses honestly belong in two: ``zero_one_test`` runs a system
    *or* a measured observable, and its own summary line says so.  Listing it
    once meant the grouped map disagreed with ``find(subject)``, which is the
    thing the map exists to advertise.
    """
    found = tuple(
        token
        for token, tokens in (("system", SYSTEM_TOKENS), ("data", DATA_TOKENS))
        if tokens.intersection(subjects)
    )
    return found or ("result",)


def grouped_map(entries: Sequence[Any], *, indent: str = "  ") -> str:
    """Render entries grouped by what you have to hold, then by area."""
    lines: list[str] = []
    for token, heading, call in GROUPS:
        rows = [e for e in entries if token in groups_of(e.metadata.get("subjects", ()))]
        if not rows:
            continue
        lines.append(f"{indent}{heading}   ({len(rows)})")
        lines.append(f"{indent}    {call}")
        for area in AREAS:
            here = sorted((e for e in rows if e.metadata.get("area") == area), key=lambda e: e.name)
            if not here:
                continue
            lines.append("")
            lines.append(f"{indent}  {AREA_LABEL.get(area, area)}")
            for entry in here:
                summary = str(entry.metadata.get("summary", ""))
                lines.append(f"{indent}    {entry.name:<25s} {summary}")
        lines.append("")
    return "\n".join(lines).rstrip()


class AnalysisList(list):  # type: ignore[type-arg]
    """The list :func:`find` returns — the analysis **functions**, grouped in its repr.

    Holds the callables, so ``find("chaotic")[0](lor)`` runs one and
    ``[f.__name__ for f in find(traj)]`` gives the names.  Received, never
    constructed.
    """

    __slots__ = ("_entries", "_header")

    def __init__(self, entries: Sequence[Any], header: str) -> None:
        super().__init__(e.obj for e in entries)
        self._entries = list(entries)
        self._header = header

    def __repr__(self) -> str:  # noqa: D105
        if not self._entries:
            return self._header
        return f"{self._header}\n{grouped_map(self._entries)}"


# ── the teaching messages ────────────────────────────────────────────────────

#: For a result-first analysis: the analysis that MAKES its subject, and the
#: call that makes it.  Nothing else in the registry holds this.
PRODUCER: dict[str, tuple[str, str]] = {
    "basin_entropy": ("b", "ts.analysis.basins(system, region)"),
    "kaplan_yorke_dimension": ("exps", "ts.analysis.lyapunov_spectrum(system)"),
    "resilience": ("b", "ts.analysis.basins(system, region)"),
    "tipping_points": ("c", "ts.analysis.continuation(system, param, values)"),
    "uncertainty_exponent": ("b", "ts.analysis.basins(system, region)"),
    "wada_property": ("b", "ts.analysis.basins(system, region)"),
}

#: For a system-first analysis: the data-first twin that answers the same
#: question from a measurement.
SIBLING: dict[str, str] = {
    "lyapunov_spectrum": "ts.analysis.lyapunov_from_data(traj)",
}

#: For a system-first analysis with no data-first twin: why there is none.  A
#: clause, appended to "it is a property of the equations, not of a point set".
NO_SIBLING: dict[str, str] = {
    "fixed_points": "they are roots of the equations",
    "periodic_orbits": "a closed orbit is a solution of the equations",
    "flow_field": "the field is the equations themselves",
    "nullclines": "a nullcline is a zero set of the equations",
    "streamlines": "they are solutions of the equations",
    "trace_determinant": "it linearises the equations",
    "ftle_field": "it re-integrates the equations from a lattice of starts",
    "escape_time_field": "it re-integrates the equations from a lattice of starts",
    "transient_time_field": "it re-integrates the equations from a lattice of starts",
    "orbit_diagram": "it re-runs the equations at every parameter value",
    "continuation": "it re-runs the equations at every parameter value",
    "basins": "it re-runs the equations from a lattice of starts",
    "attractors": "it re-runs the equations from a cloud of starts",
    "basin_fractions": "it re-runs the equations from sampled starts",
    "gali": "it evolves the equations' tangent dynamics",
    "expansion_entropy": "it evolves the equations' tangent dynamics",
    "poincare_section": "the crossings are refined against the equations",
    "return_map": "the returns are read off a run of the equations",
}

#: The run-me line each held family actually has, keyed by the family word.
_RUN_LINE = {
    "map": "traj = system.run(20000, transient=1000)",
    "flow": "traj = system.run(200.0, dt=0.02)",
}


def _wants(name: str) -> str:
    """Return ``"system"`` / ``"data"`` / ``"result"`` for a registered analysis."""
    from tsdynamics import registry

    try:
        entry = registry.analyses.entry(name)
    except KeyError:
        return ""
    return group_of(entry.metadata.get("subjects", ()))


#: What ``find()`` is handed, and what it is called, for each held thing.  A
#: flow and a map are counted **separately** because they get different answers:
#: a map has no vector field, so the seven field analyses drop out.
_FIND_SUBJECT: dict[str, tuple[str, str, str]] = {
    "flow": ("system", "flow", "flow"),
    "map": ("system", "map", "map"),
    "system": ("system", "flow", "flow"),
    "data": ("traj", "trajectory", "trajectory"),
    "result": ("traj", "trajectory", "trajectory"),
}


def find_count(token: str) -> int:
    """How many registered analyses accept a subject carrying ``token``.

    The number a :func:`~tsdynamics.analysis.find` call would actually print, read
    from the registry rather than written down — a message that *states* a count
    and then disagrees with the call it recommends teaches the wrong thing twice.
    """
    from tsdynamics import registry

    return sum(1 for e in registry.analyses.all() if token in e.metadata.get("subjects", ()))


def find_line(held: str, *, lead: str = "all") -> str:
    """Return the ``ts.analysis.find(...)`` line, carrying the count it will print."""
    var, token, word = _FIND_SUBJECT.get(held, _FIND_SUBJECT["data"])
    return f"ts.analysis.find({var})   # {lead} {find_count(token)} that take a {word}"


def teach(name: str, *, held: str, has_system: bool = True) -> list[str]:
    """Build the body of every "that moved" message — the one text builder.

    Parameters
    ----------
    name : str
        The analysis that was reached for.
    held : {"flow", "map", "system", "data", "result"}
        What the caller is holding.
    has_system : bool
        Whether the held data knows the system that produced it.  A
        :class:`~tsdynamics.data.Trajectory` with ``system=None`` cannot be sent
        to ``traj.system``, so it is sent to :func:`find` instead.

    Returns
    -------
    list[str]
        ``[clause, *runnable lines]`` — the clause completes a sentence that
        already names what was typed, and every remaining entry is a line the
        reader can run.
    """
    wants = _wants(name)
    holding_system = held in ("flow", "map", "system")

    if wants == "data" and holding_system:
        run = _RUN_LINE.get(held, _RUN_LINE["flow"])
        return [
            "it measures a point set, so it needs data. Run the system first:",
            run,
            f"ts.analysis.{name}(traj)",
        ]
    if wants == "system" and not holding_system:
        clause = "it is a property of the equations, not of a point set"
        extra = NO_SIBLING.get(name)
        clause = f"{clause} — {extra}." if extra else f"{clause}."
        twin = SIBLING.get(name)
        if not has_system:
            # No ``.system`` to forward to — but if a data-driven twin exists it
            # is still the answer, and dropping it sent the one caller who has
            # *only* a measurement to a listing instead of to the function that
            # answers them.
            return [clause, *([twin] if twin else []), find_line(held, lead="the")]
        lines = [clause, f"ts.analysis.{name}(traj.system)"]
        if twin:
            lines.append(twin)
        return lines
    if wants == "result":
        var, producer = PRODUCER.get(name, ("r", "ts.analysis.find(system)"))
        return [
            "it reads what another analysis returns. Compute that first:",
            producer if "=" in producer else f"{var} = {producer}",
            f"ts.analysis.{name}({var})",
        ]
    # The subject is right; only the spelling was.
    subject = "system" if holding_system else "traj"
    return [
        "analyses are free functions in v6, and the subject is the first argument.",
        f"ts.analysis.{name}({subject})",
        find_line(held),
    ]


def _held_word(held: str) -> str:
    return {"data": "measured data", "result": "a result"}.get(held, "a system")


#: The widest a message line may render.  Measured against the **rendered**
#: traceback, which prepends ``AttributeError: `` (16 columns) or
#: ``InvalidInputError: `` (19) to the first line — count the prefix or the
#: first line of most of these messages runs off the terminal.
MESSAGE_WIDTH = 88


def _render(head: str, lines: Sequence[str], prefix: int) -> str:
    """Wrap the sentence to :data:`MESSAGE_WIDTH` and indent the runnable lines."""
    wrapped = textwrap.wrap(head, width=MESSAGE_WIDTH - prefix) or [head]
    rest = textwrap.wrap(" ".join(wrapped[1:]), width=MESSAGE_WIDTH) if len(wrapped) > 1 else []
    return "\n".join([wrapped[0], *rest, *(f"    {line}" for line in lines)])


def wrong_subject(name: str, cls: str, held: str, *, has_system: bool = True) -> Exception:
    """Build the free-function door's error — same body as the object door's."""
    from tsdynamics.errors import InvalidInputError

    wants = _wants(name)
    needs = {
        "system": "a system",
        "data": "data",
        "result": "a result from another analysis",
    }.get(wants, "another subject")
    clause, *lines = teach(name, held=held, has_system=has_system)
    head = f"{name}() needs {needs}, and got {_held_word(held)} ({cls}): {clause}"
    return InvalidInputError(_render(head, lines, len("InvalidInputError: ")))


def attribute_error(name: str, cls: str, held: str, *, has_system: bool = True) -> AttributeError:
    """Build the object door's error — same body as the free-function door's."""
    clause, *lines = teach(name, held=held, has_system=has_system)
    head = f"{cls!r} object has no attribute {name!r}: {clause}"
    return AttributeError(_render(head, lines, len("AttributeError: ")))


def near_miss(name: str, known: Iterable[str]) -> str | None:
    """Return the closest registered analysis name to ``name``, or ``None``."""
    close = difflib.get_close_matches(name, list(known), n=1, cutoff=0.6)
    return close[0] if close else None


# ── the registration door ────────────────────────────────────────────────────


def register(
    func: Callable[..., Any] | None = None,
    /,
    *,
    subjects: Iterable[str],
    area: str,
    returns: type | None = None,
    keywords: str = "",
    cite: str | None = None,
    doi: str | None = None,
    name: str | None = None,
    replace: bool = False,
) -> Any:
    """Register an analysis.  The definition site *is* the registration site.

    Usable as a decorator or called directly.  Six things follow from one call:
    the :data:`tsdynamics.registry.analyses` entry, a name in the generated
    ``ts.analysis.__all__``, a row in the generated ``ts.analysis.__doc__`` map,
    reachability from :func:`find`, the declared return type, and the
    bibliography entry.

    Parameters
    ----------
    func : callable, optional
        The analysis.  Omit it to use ``register`` as a decorator factory.
    subjects : iterable of str
        What the **first argument** is: ``"system"`` (equivalently ``"flow"``
        and/or ``"map"`` when only one applies), ``"trajectory"`` / ``"array"``
        for measured data, or the class name of the result it reads.
    area : str
        One of :data:`AREAS` — the group it lists under.
    returns : type, optional
        The :class:`~tsdynamics.analysis.results.AnalysisResult` subclass it
        returns.
    keywords : str
        Space-separated synonyms **a user types that the docstring does not
        contain** ("chaotic", "fractal", "multistability", "takens").  Measured:
        without them ``find`` scores precision@3 = 63 % and returns nothing for
        "multistability"; with them, 88 % and the three right answers.
    cite, doi : str, optional
        The original paper, and its DOI.
    name : str, optional
        The registered name (default: ``func.__name__``).
    replace : bool
        Overwrite an existing registration under the same name.

    Examples
    --------
    >>> from tsdynamics.analysis import register            # doctest: +SKIP
    >>> @register(subjects=("trajectory", "array"), area="dimensions",
    ...           keywords="fractal attractor scaling grassberger")
    ... def my_dimension(subject, *, radii=None):
    ...     '''My dimension estimator.'''                   # doctest: +SKIP
    """
    if area not in AREAS:
        from tsdynamics.errors import invalid_value

        raise invalid_value(
            "area", area, rule=f"must be one of {', '.join(AREAS)}", hint="pick the closest area"
        )
    tokens = expand_subjects(subjects)
    if not tokens:
        from tsdynamics.errors import invalid_value

        raise invalid_value("subjects", subjects, rule="must name at least one subject")

    def _apply(target: Callable[..., Any]) -> Callable[..., Any]:
        from tsdynamics import registry

        registry.analyses.register(
            name or target.__name__,
            target,
            replace=replace,
            subjects=tokens,
            area=area,
            returns=returns,
            keywords=keywords,
            summary=summarise(target.__doc__),
            cite=cite,
            doi=doi,
        )
        return target

    return _apply if func is None else _apply(func)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
